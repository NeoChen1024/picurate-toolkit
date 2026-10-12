"""Images on disk: which files count as images, how a directory is listed, how an image is identified, and the
metadata kept per image.

Every database keys an image by its item ID, derived with BLAKE3 from ``(source_id, relative path)``. A dataset
project creates a random ``source_id`` per source folder; a standalone image directory has the all-zero one, so both
go through the same code. The file is ``roots[source_id] / path``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
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


def walk_directory_relative(directory: str, *, include_hidden: bool = False) -> Iterator[str]:
    """Yield file paths relative to ``directory``, depth first.

    Unless ``include_hidden``, hidden files and directories (dot names) are skipped: build output, caches, VCS
    metadata and macOS ``._*`` sidecars live there, rarely pictures anyone meant to include.
    """
    for root, dirs, files in os.walk(directory):
        if not include_hidden:
            dirs[:] = [d for d in dirs if not is_hidden(d)]
        for file in files:
            if include_hidden or not is_hidden(file):
                yield os.path.relpath(os.path.join(root, file), directory)


def list_directory_relative(directory: str, *, recursive: bool, include_hidden: bool = False) -> Iterator[str]:
    """The files directly in ``directory``, or with ``recursive`` everything below it (see
    :func:`walk_directory_relative`), as paths relative to it. Hidden ones are skipped unless ``include_hidden``."""
    if recursive:
        yield from walk_directory_relative(directory, include_hidden=include_hidden)
        return
    for entry in os.scandir(directory):
        if entry.is_file() and (include_hidden or not is_hidden(entry.name)):
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
