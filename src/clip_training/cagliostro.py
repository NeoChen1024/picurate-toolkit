"""Import of the cagliostrolab-860k-ordered Danbooru dataset into a pack (see :mod:`clip_training.pack`).

The dataset is image/caption pairs, ``danbooru_<post id>_<md5>.{jpg,png,webp}`` with a same-stem ``.txt``, spread over
rating directories that were imported in stages (the top level and ``continued/``); every directory below the
dataset root is read. A Danbooru post's key is its file stem, so a post found in two stages is imported once.
``continued/screencap/<series>/<character>/`` holds anime frames named by number, where one number can be another
frame in another directory, or the same frame captioned for another character: their key is the path below the
stage directory without the extension, ``screencap/<series>/<character>/<number>`` (identical frames are then two
records with identical stored bytes). A Danbooru file name ends in the MD5 of the original post file. About 60% of
the PNGs no longer match it: the dataset's makers saved them again (RGB, no metadata chunks, Pillow's 64 KiB IDAT
chunks), as did a few dozen JPEG and WebP files. So a mismatch alone is no damage; such a file is accepted when it
decodes completely, which PNG's checksums make a real integrity check. Some JPEGs were saved cut off at a whole
number of MiB and are rejected for not decoding completely. A caption reads::

    1girl, <characters>, <copyrights>, <artists>, ||| <rating>, [nsfw,] <general tags>, <year>, <meta>, <quality>, <aesthetic>

The tags are taken from the caption, in its order, without the ``|||``. Their categories are not in it; they come
from a tag database converted once from ``cagliostrolab-860k-ordered-tags.json`` (:func:`convert_tag_json`), a
separate dataset of the same kind of posts: only the categories are taken from it, never its tags. The words the
dataset adds for conditioning (the rating word and ``nsfw`` at the start of the second part, the year range, quality
and aesthetic words) get the category ``training``. A tag the database lacks (it has no tags of three characters or
fewer) is ``name`` before the ``|||`` (a character, copyright or artist) and ``general`` after it.

The tag database, a SQLite file:

* ``meta``: schema version and source file name.
* ``posts``: id, md5, score, year, rating letter, aesthetic words (comma-separated) and image URL. Where the JSON has
  an id twice, the first entry is kept.
* ``tags``: name and category. The category is the post field (``character``, ``copyright``, ``artist``,
  ``general``, ``meta``) a tag is listed in most often, or ``training`` for conditioning words, or by position as
  above for tags in no field.
* ``post_tags``: the tags of ``ordered_tags`` per post, in order, with ``head`` set before the ``|||``.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sqlite3
from collections import Counter
from collections.abc import Iterator

RATINGS = {"safe", "general", "sensitive", "questionable", "explicit"}
TRAINING_WORDS = {
    "nsfw",
    *("oldest", "early", "mid", "recent", "newest"),
    *("worst quality", "low quality", "normal quality", "good quality", "great quality", "best quality", "masterpiece"),
    *("very displeasing", "displeasing", "aesthetic", "very aesthetic"),
}
CATEGORY_FIELDS = ("character", "copyright", "artist", "general", "meta")
TAG_DB_VERSION = "1"
_TAG_DB_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE posts (
    id INTEGER PRIMARY KEY, md5 TEXT, score INTEGER, year INTEGER, rating TEXT, aesthetic TEXT, image TEXT
);
CREATE TABLE tags (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, category TEXT NOT NULL);
CREATE TABLE post_tags (
    post INTEGER NOT NULL, position INTEGER NOT NULL, tag INTEGER NOT NULL, head INTEGER NOT NULL,
    PRIMARY KEY (post, position)
) WITHOUT ROWID;
"""
_CHUNK = 1 << 22
STAGE_DIRECTORY = "continued"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def read_json_list(path: str) -> Iterator[dict]:
    """The objects of a JSON file holding one list, one at a time instead of the whole file in memory."""
    decoder = json.JSONDecoder()
    skip = re.compile(r"[\s,\[]*")
    with open(path, encoding="utf-8") as stream:
        buffer, position = "", 0
        while True:
            position = skip.match(buffer, position).end()
            if buffer.startswith("]", position):
                return
            try:
                item, position = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                more = stream.read(_CHUNK)
                if not more:
                    raise
                buffer, position = buffer[position:] + more, 0
                continue
            yield item


def convert_tag_json(json_path: str, db_path: str) -> dict[str, int]:
    """Write the tag database (see the module) for the JSON list of posts at ``json_path``; returns counts."""
    split = lambda text: [tag for tag in (t.strip() for t in (text or "").split(",")) if tag]  # noqa: E731
    db = sqlite3.connect(db_path)
    db.executescript("PRAGMA journal_mode = OFF; PRAGMA synchronous = OFF;" + _TAG_DB_SCHEMA)
    db.executemany("INSERT INTO meta VALUES (?, ?)", [("schema_version", TAG_DB_VERSION), ("source", os.path.basename(json_path))])
    tag_ids: dict[str, int] = {}
    votes: dict[str, Counter] = {}
    fallback: dict[str, Counter] = {}
    counts = Counter()
    for post in read_json_list(json_path):
        url = post.get("image") or ""
        aesthetic = post.get("aesthetic")
        row = (
            post["id"],
            os.path.splitext(url.rsplit("/", 1)[-1])[0] or None,
            post.get("score"),
            post.get("year"),
            post.get("rating"),
            ", ".join(aesthetic) if isinstance(aesthetic, list) else aesthetic,
            url or None,
        )
        if db.execute("INSERT OR IGNORE INTO posts VALUES (?, ?, ?, ?, ?, ?, ?)", row).rowcount == 0:
            counts["duplicate ids"] += 1
            continue
        counts["posts"] += 1
        fields = {tag: field for field in CATEGORY_FIELDS for tag in split(post.get(field))}
        ordered = parse_caption(post.get("ordered_tags") or "")
        second = [tag for tag, head in ordered if not head]
        rows = []
        for position, (tag, head) in enumerate(ordered):
            tag_id = tag_ids.setdefault(tag, len(tag_ids) + 1)
            rows.append((post["id"], position, tag_id, int(head)))
            if tag in fields:
                votes.setdefault(tag, Counter())[fields[tag]] += 1
            elif not head and ((tag in RATINGS and second.index(tag) == 0) or tag in TRAINING_WORDS):
                votes.setdefault(tag, Counter())["training"] += 1
            else:
                fallback.setdefault(tag, Counter())["name" if head else "general"] += 1
        listed = {tag for tag, _ in ordered}
        for tag, field in fields.items():
            if tag not in listed:
                counts["field tags missing from ordered_tags"] += 1
                tag_ids.setdefault(tag, len(tag_ids) + 1)
                votes.setdefault(tag, Counter())[field] += 1
        counts["post tags"] += len(rows)
        db.executemany("INSERT INTO post_tags VALUES (?, ?, ?, ?)", rows)
    tags = []
    for tag, tag_id in tag_ids.items():
        tally = votes.get(tag) or fallback[tag]
        counts["tags with conflicting categories"] += len(tally) > 1
        tags.append((tag_id, tag, tally.most_common(1)[0][0]))
    db.executemany("INSERT INTO tags VALUES (?, ?, ?)", tags)
    counts["tags"] = len(tags)
    db.commit()
    db.close()
    return dict(counts)


def category_dictionary(db_path: str) -> dict[str, str]:
    """``{tag: category}`` from the tag database."""
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        version = db.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        if version != (TAG_DB_VERSION,):
            raise ValueError(f"{db_path} is not a tag database of schema version {TAG_DB_VERSION}; convert the JSON again")
        return dict(db.execute("SELECT name, category FROM tags"))
    finally:
        db.close()


def parse_caption(text: str) -> list[tuple[str, bool]]:
    """``(tag, before the |||)`` in caption order."""
    head, separator, tail = text.partition("|||")
    if not separator:
        head, tail = "", head
    return [(tag, k == 0) for k, part in enumerate((head, tail)) for tag in (t.strip() for t in part.split(",")) if tag]


def categorize(tags: list[tuple[str, bool]], dictionary: dict[str, str]) -> list[tuple[str, str]]:
    """``(tag, category)``: conditioning words are ``training``, the rest by the dictionary or by position."""
    out = []
    second = [tag for tag, before in tags if not before]
    for tag, before in tags:
        if not before and ((tag in RATINGS and second.index(tag) == 0) or tag in TRAINING_WORDS):
            out.append((tag, "training"))
        else:
            out.append((tag, dictionary.get(tag) or ("name" if before else "general")))
    return out


def stage_path(path: str) -> str:
    """``path`` without the stage directory: ``continued/screencap/a/b`` -> ``screencap/a/b``."""
    return "/".join(part for part in path.split(os.sep) if part != STAGE_DIRECTORY)


def find_pairs(root: str, unpaired: list[str] | None = None) -> Iterator[tuple[str, str, str]]:
    """``(key, image path, caption path)`` for every image with a caption anywhere below ``root``, each key once.
    Images without a caption and captions without an image are appended to ``unpaired``."""
    seen = set()
    for directory, dirs, files in os.walk(root):
        dirs.sort()
        names = set(files)
        images = set()
        for name in sorted(files):
            stem, extension = os.path.splitext(name)
            if extension.lower() not in IMAGE_EXTENSIONS:
                continue
            images.add(stem)
            if stem + ".txt" not in names:
                if unpaired is not None:
                    unpaired.append(os.path.join(directory, name))
                continue
            key = stem if stem.startswith("danbooru_") else stage_path(os.path.relpath(os.path.join(directory, stem), root))
            if key not in seen:
                seen.add(key)
                yield key, os.path.join(directory, name), os.path.join(directory, stem + ".txt")
        if unpaired is not None:
            unpaired.extend(
                os.path.join(directory, name) for name in sorted(files) if name.endswith(".txt") and name[:-4] not in images
            )


def read_image(key: str, image_path: str) -> tuple[bytes, bool | None]:
    """The file's bytes, and whether they match the MD5 in a Danbooru file name (None for other files)."""
    with open(image_path, "rb") as stream:
        data = stream.read()
    if not key.startswith("danbooru_"):
        return data, None
    return data, hashlib.md5(data).hexdigest() == key.rsplit("_", 1)[1]


def read_caption(caption_path: str) -> list[tuple[str, bool]]:
    with open(caption_path, encoding="utf-8") as stream:
        tags = parse_caption(stream.read())
    if not tags:
        raise ValueError("empty caption")
    return tags


def encode_pair(key: str, image_path: str, caption_path: str, budget: int):
    """Worker: the encoded image and the parsed caption, or the exception."""
    from .pack import encode_file

    try:
        tags = read_caption(caption_path)
        return encode_file(io.BytesIO(read_image(key, image_path)[0]), budget), tags
    except Exception as error:  # noqa: BLE001 - reported per file
        return error, []


def check_pair(key: str, image_path: str, caption_path: str, decode: bool) -> tuple[str | None, str]:
    """Worker: what is wrong with a pair (unreadable, empty caption, undecodable image) or None, and how the image was
    checked: ``original`` (matches its MD5), ``resaved`` (does not) or ``unnamed`` (no MD5 in the name). An image
    without a matching MD5 is always decoded."""
    import PIL.Image
    import PIL.ImageFile

    status = "unread"
    try:
        read_caption(caption_path)
        data, matches = read_image(key, image_path)
        status = {True: "original", False: "resaved", None: "unnamed"}[matches]
        if decode or not matches:
            PIL.ImageFile.LOAD_TRUNCATED_IMAGES = False
            with PIL.Image.open(io.BytesIO(data)) as image:
                image.load()
    except Exception as error:  # noqa: BLE001 - reported per file
        return f"{type(error).__name__}: {error}", status
    return None, status
