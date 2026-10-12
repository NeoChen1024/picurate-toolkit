"""Images on disk: which files count as images, how a directory is listed, how an image is identified, and the
metadata kept per image.

Every database keys an image by its item ID, derived with BLAKE3 from ``(source_id, relative path)``. A dataset
project creates a random ``source_id`` per source folder; a standalone image directory has the all-zero one, so both
go through the same code. The file is ``roots[source_id] / path``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field

import PIL.Image
import PIL.ImageFile
from blake3 import blake3

# This is a tool for the user's own collection, not a server decoding untrusted uploads: behave like an image viewer.
# Pillow's defaults reject images above ~179 Mpixel ("decompression bomb") and files whose data ends early, both of
# which ordinary viewers display without complaint. These module globals apply to the process that imports this
# module; worker processes re-import it, so they inherit the same settings.
PIL.Image.MAX_IMAGE_PIXELS = None
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

# Extensions Pillow knows how to open. Anything else in the image dir (sidecar .txt, .json, ...) is ignored silently.
IMAGE_EXTENSIONS = frozenset(ext for ext, fmt in PIL.Image.registered_extensions().items() if fmt in PIL.Image.OPEN)


def is_hidden(name: str) -> bool:
    return name.startswith(".")


def _glob_regex(pattern: str) -> str:
    out, i = [], 0
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith("**", i):
            out.append(".*")
            i += 2
            continue
        if c == "*":
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c == "[" and (end := pattern.find("]", i + 2)) != -1:
            body = pattern[i + 1 : end]
            out.append("[" + ("^" + body[1:] if body[0] in "!^" else body).replace("\\", "\\\\") + "]")
            i = end + 1
            continue
        else:
            out.append(re.escape(c))
        i += 1
    return "".join(out)


@dataclass(slots=True, frozen=True)
class FilterRule:
    include: bool
    pattern: str
    regex: re.Pattern
    dir_only: bool


class PathFilter:
    """rsync-style include/exclude rules over paths relative to a source folder. The first rule that matches decides;
    a path no rule matches is included. An excluded directory is not descended into.

    * A pattern without ``/`` matches the last path component (``*.tmp``); one with ``/`` or ``**`` matches the whole
      path, or with a leading ``/`` only from the folder's root (``/raw/``).
    * ``*`` matches within a component, ``**`` across them, ``?`` one character, ``[...]`` a class.
    * A trailing ``/`` matches directories only; a trailing ``/***`` matches a directory and everything in it.
    """

    def __init__(self, rules: Iterable[tuple[str, str]] = ()):
        self.rules = [self._compile(sign, pattern) for sign, pattern in rules]

    @staticmethod
    def _compile(sign: str, pattern: str) -> FilterRule:
        if sign not in "+-" or len(sign) != 1 or not pattern:
            raise ValueError(f"Invalid filter rule: {sign} {pattern}")
        body, contents = pattern, False
        if body.endswith("/***"):
            body, contents = body[:-4], True
        dir_only = body.endswith("/") and not contents
        body = body.rstrip("/")
        anchored = body.startswith("/")
        regex = _glob_regex(body.lstrip("/")) + ("(?:/.*)?" if contents else "")
        return FilterRule(sign == "+", pattern, re.compile(("^" if anchored else "(?:^|.*/)") + regex + "$"), dir_only)

    @staticmethod
    def parse(line: str) -> tuple[str, str]:
        """``"- *.tmp"`` -> ``("-", "*.tmp")``, as rsync's ``--filter`` writes rules."""
        sign, _, pattern = line.strip().partition(" ")
        PathFilter._compile(sign, pattern.strip())
        return sign, pattern.strip()

    def __bool__(self) -> bool:
        return bool(self.rules)

    def included(self, relative_path: str, *, is_dir: bool = False) -> bool:
        for rule in self.rules:
            if (is_dir or not rule.dir_only) and rule.regex.match(relative_path):
                return rule.include
        return True


def walk_directory_relative(
    directory: str, *, include_hidden: bool = False, path_filter: PathFilter | None = None
) -> Iterator[str]:
    """Yield file paths relative to ``directory``, depth first.

    Unless ``include_hidden``, hidden files and directories (dot names) are skipped: build output, caches, VCS
    metadata and macOS ``._*`` sidecars live there, rarely pictures anyone meant to include. ``path_filter`` drops
    files and prunes directories it excludes.
    """
    for root, dirs, files in os.walk(directory):
        base = os.path.relpath(root, directory)
        base = "" if base == "." else base + "/"
        if not include_hidden:
            dirs[:] = [d for d in dirs if not is_hidden(d)]
        if path_filter:
            dirs[:] = [d for d in dirs if path_filter.included(base + d, is_dir=True)]
        for file in files:
            if (include_hidden or not is_hidden(file)) and (not path_filter or path_filter.included(base + file)):
                yield base + file


def list_directory_relative(
    directory: str, *, recursive: bool, include_hidden: bool = False, path_filter: PathFilter | None = None
) -> Iterator[str]:
    """The files directly in ``directory``, or with ``recursive`` everything below it (see
    :func:`walk_directory_relative`), as paths relative to it. Hidden ones are skipped unless ``include_hidden``,
    and ones ``path_filter`` excludes always."""
    if recursive:
        yield from walk_directory_relative(directory, include_hidden=include_hidden, path_filter=path_filter)
        return
    for entry in os.scandir(directory):
        if not entry.is_file() or not (include_hidden or not is_hidden(entry.name)):
            continue
        if not path_filter or path_filter.included(entry.name):
            yield entry.name


def is_image_path(relative_path: str) -> bool:
    return os.path.splitext(relative_path)[1].lower() in IMAGE_EXTENSIONS


def standalone_items(directory: str, *, recursive: bool = True, include_hidden: bool = False) -> list[Item]:
    """The images of a standalone directory (see :func:`list_directory_relative`), sorted by path."""
    rels = list_directory_relative(directory, recursive=recursive, include_hidden=include_hidden)
    return [Item.standalone(rel) for rel in sorted(r for r in rels if is_image_path(r))]


# The source of a standalone image directory; never generated for a project source.
STANDALONE_SOURCE = bytes(32)
ITEM_ID_CONTEXT = "picurate item_id v1"

# Source folder of every source_id an image list refers to.
Roots = Mapping[bytes, str]


def item_id(source_id: bytes, relative_path: str) -> bytes:
    data = source_id + b"\0" + relative_path.encode("utf-8", "surrogateescape")
    return blake3(data, derive_key_context=ITEM_ID_CONTEXT).digest()


def file_digest(path) -> bytes:
    """BLAKE3 of a file's contents, read sequentially (parallel reads would make spinning disks seek)."""
    hasher = blake3()
    with open(path, "rb") as stream:
        while chunk := stream.read(1 << 20):
            hasher.update(chunk)
    return hasher.digest()


def standalone_roots(image_dir: str) -> dict[bytes, str]:
    return {STANDALONE_SOURCE: image_dir}


@dataclass(slots=True, frozen=True)
class Item:
    """One image to process: its ID and where it is (``roots[source_id] / path``)."""

    id: bytes
    source_id: bytes
    path: str

    @classmethod
    def standalone(cls, relative_path: str) -> Item:
        return cls(item_id(STANDALONE_SOURCE, relative_path), STANDALONE_SOURCE, relative_path)

    def file(self, roots: Roots) -> str:
        return os.path.join(roots[self.source_id], self.path)


@dataclass(slots=True, frozen=True)
class ImageRecord:
    """Metadata of one image as observed when it was read (and as the databases store it). Without an explicit
    ``id`` the record is of a standalone directory and the ID is derived from ``source_id`` and ``path``."""

    path: str  # relative to the source folder
    mtime: float
    size: int
    width: int
    height: int
    format: str  # PIL format name: "JPEG", "PNG", "GIF", "WEBP", ...
    source_id: bytes = STANDALONE_SOURCE
    id: bytes = field(default=b"")
    blake3: bytes | None = None  # of the file's contents, when it was read whole

    def __post_init__(self) -> None:
        if not self.id:
            object.__setattr__(self, "id", item_id(self.source_id, self.path))

    @property
    def item(self) -> Item:
        return Item(self.id, self.source_id, self.path)

    def file(self, roots: Roots) -> str:
        return os.path.join(roots[self.source_id], self.path)

    @property
    def pixels(self) -> int:
        return self.width * self.height
