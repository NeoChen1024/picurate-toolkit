"""Import of a deepghs ``*_full`` booru mirror (written for ``deepghs/konachan_full``) into a pack (see
:mod:`clip_training.pack`).

The dataset is one row per post in ``table.parquet`` (``id``, ``filename``, ``mimetype``, ``rating``, a
space-separated ``tags`` string, ...), the tag dictionary in ``tags.parquet`` (``name``, Moebooru ``type`` numbers),
and the original files in ``images/<n>.tar`` with a same-stem ``.json`` listing every member's offset, size and
SHA-256 within the archive. A post's file is read straight from the archive at that offset (no tar parsing) and
checked against its SHA-256 before it is decoded. A post is keyed ``<prefix>_<id>`` (``konachan_391069``), so a pack
can hold posts of several boorus.

Tags are stored by category in the order the captions of :mod:`clip_training.data` expect, names first: character,
copyright, artist, circle, style, then general and tags of unknown type (category ``None``); alphabetical within a
category, as the dataset lists them. The post's rating is stored first as the word ``safe``, ``questionable`` or
``explicit`` with the category ``training``, the conditioning category the captions leave out. GIF posts (animations)
are skipped.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

# Moebooru tag types (konachan, yande.re).
TAG_TYPES = {0: "general", 1: "artist", 3: "copyright", 4: "character", 5: "style", 6: "circle"}
CATEGORY_ORDER = ("character", "copyright", "artist", "circle", "style", "general", None)
RATINGS = {"s": "safe", "q": "questionable", "e": "explicit"}
IMAGES = "images"


@dataclass(slots=True)
class Post:
    key: str
    archive: str  # path of the tar holding the file
    offset: int
    size: int
    sha256: str
    tags: list[tuple[str, str | None]]


def archive_index(dataset: str) -> dict[str, tuple[str, int, int, str]]:
    """``{file name: (archive path, offset, size, sha256)}`` over every ``images/<n>.json``."""
    index = {}
    directory = Path(dataset) / IMAGES
    for listing in sorted(directory.glob("*.json")):
        archive = str(listing.with_suffix(".tar"))
        with open(listing, encoding="utf-8") as stream:
            for name, entry in json.load(stream)["files"].items():
                index[name] = (archive, int(entry["offset"]), int(entry["size"]), entry["sha256"])
    return index


def tag_categories(dataset: str) -> dict[str, str | None]:
    """``{tag: category}`` from ``tags.parquet``."""
    import pyarrow.parquet as pq

    table = pq.read_table(os.path.join(dataset, "tags.parquet"), columns=["name", "type"]).to_pydict()
    return {name: TAG_TYPES.get(kind) for name, kind in zip(table["name"], table["type"])}


def post_tags(tags: str, rating: str, categories: dict[str, str | None]) -> list[tuple[str, str | None]]:
    """The stored tags of a post: the rating word, then the tags by category (see the module)."""
    rank = {category: k for k, category in enumerate(CATEGORY_ORDER)}
    out = [(name, categories.get(name)) for name in tags.split()]
    out.sort(key=lambda tag: rank.get(tag[1], len(rank)))
    return [(RATINGS[rating], "training")] + out if rating in RATINGS else out


def posts(dataset: str, prefix: str, counts: dict[str, int]) -> Iterator[Post]:
    """Every importable post of the dataset; ``counts`` receives what was left out and why."""
    import pyarrow.parquet as pq

    index = archive_index(dataset)
    categories = tag_categories(dataset)
    columns = ["id", "filename", "mimetype", "rating", "tags"]
    table = pq.read_table(os.path.join(dataset, "table.parquet"), columns=columns).to_pydict()
    for post_id, filename, mimetype, rating, tags in zip(*(table[column] for column in columns)):
        if mimetype == "image/gif":
            counts["gif"] = counts.get("gif", 0) + 1
            continue
        entry = index.get(filename)
        if entry is None:
            counts["not in the archives"] = counts.get("not in the archives", 0) + 1
            continue
        yield Post(f"{prefix}_{post_id}", *entry, post_tags(tags or "", rating or "", categories))


def read_member(archive: str, offset: int, size: int, sha256: str) -> bytes:
    """The file at ``offset`` in ``archive``, checked against its SHA-256."""
    fd = os.open(archive, os.O_RDONLY)
    try:
        data = bytearray()
        while len(data) < size:
            chunk = os.pread(fd, size - len(data), offset + len(data))
            if not chunk:
                raise OSError(f"{archive}: member at {offset} is cut short")
            data += chunk
    finally:
        os.close(fd)
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError(f"{archive}: member at {offset} does not match its SHA-256")
    return bytes(data)


def encode_post(key: str, archive: str, offset: int, size: int, sha256: str, tags: list, budget: int):
    """Worker: the encoded image and ``tags`` as given, or the exception."""
    from .pack import encode_file

    try:
        return encode_file(io.BytesIO(read_member(archive, offset, size, sha256)), budget), tags
    except Exception as error:  # noqa: BLE001 - reported per post
        return error, tags


def check_post(key: str, archive: str, offset: int, size: int, sha256: str, expected: tuple[int, int], decode: bool) -> str | None:
    """Worker: what is wrong with a post's file (cut short, SHA-256 mismatch, undecodable, or another size than the
    table says), or None."""
    import PIL.Image
    import PIL.ImageFile

    try:
        data = read_member(archive, offset, size, sha256)
        if decode:
            PIL.ImageFile.LOAD_TRUNCATED_IMAGES = False
            with PIL.Image.open(io.BytesIO(data)) as image:
                image.load()
                if image.size != expected:
                    return f"decodes to {image.size[0]}x{image.size[1]}, the table says {expected[0]}x{expected[1]}"
    except Exception as error:  # noqa: BLE001 - reported per post
        return f"{type(error).__name__}: {error}"
    return None


def check_listing(dataset: str) -> tuple[list[tuple[str, str]], list[tuple[str, str, int, int, str, tuple[int, int]]]]:
    """Problems of the dataset's tables and archives that need no reading of images (``(what, problem)``), and the
    posts to check: ``(key, archive, offset, size, sha256, (width, height))`` in archive order."""
    import pyarrow.parquet as pq

    problems = []
    index = archive_index(dataset)
    for listing in sorted((Path(dataset) / IMAGES).glob("*.json")):
        with open(listing, encoding="utf-8") as stream:
            expected = int(json.load(stream)["filesize"])
        archive = listing.with_suffix(".tar")
        actual = archive.stat().st_size if archive.exists() else None
        if actual != expected:
            problems.append((str(archive), f"{actual if actual is not None else 'missing'} bytes, the listing says {expected}"))
    categories = tag_categories(dataset)
    columns = ["id", "filename", "width", "height", "tags"]
    table = pq.read_table(os.path.join(dataset, "table.parquet"), columns=columns).to_pydict()
    seen = set()
    posts = []
    for post_id, filename, width, height, tags in zip(*(table[column] for column in columns)):
        if post_id in seen:
            problems.append((filename, "id listed twice"))
            continue
        seen.add(post_id)
        unknown = [tag for tag in (tags or "").split() if tag not in categories]
        if unknown:
            problems.append((filename, f"tags not in tags.parquet: {' '.join(unknown)}"))
        if not (tags or "").split():
            problems.append((filename, "no tags"))
        entry = index.get(filename)
        if entry is None:
            problems.append((filename, "not in the archives"))
            continue
        posts.append((filename, *entry, (int(width), int(height))))
    posts.sort(key=lambda post: (post[1], post[2]))
    return problems, posts
