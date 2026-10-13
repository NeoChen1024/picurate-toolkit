"""Training image packs: images resized to the training budget, stored as lossless WebP and concatenated into shards,
with an SQLite index.

Layout of a pack directory:

* ``pack-00000.bin``, ``pack-00001.bin``, ...: the encoded images back to back, each shard at most ``shard_bytes``
  (16 GiB by default). Reading one image is one ``pread`` of its bytes, which suits a network share: no memory mapping
  and no locks.
* ``index.sqlite``: ``meta`` (schema version, image format, pixel budget, shard size); ``images`` (key, shard, offset,
  length, stored size, source size, BLAKE3 of the stored bytes); ``tags`` (name, category) and ``image_tags`` (each
  image's tags in their source order).

A truncated or unreadable source is reported and skipped, never stored partly. Images are stored upright (EXIF orientation applied), RGB (transparency composited on white), and downscaled with
Lanczos so that width x height <= ``budget`` pixels; smaller ones are kept as they are. The default budget is twice
the pixels of SigLIP2's 1024 patches of 16x16, room for the crops and rescales of augmentation. Lossless WebP at
method 4, quality 50 is about 26% smaller than optimized PNG and decodes faster; higher efforts cost 4-50 times the
encoding time for 1-3% (measured on library images).

Writing is resumable: shard bytes are fsynced before the index rows that point at them are committed, every
``commit_every`` images, and opening a pack for writing truncates the shards to what the index holds. Keys already in
the index are skipped by the importer, so an interrupted import is simply run again.
"""

from __future__ import annotations

import io
import logging
import os
import signal
import sqlite3
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
import PIL.Image
import PIL.ImageFile
import PIL.ImageOps
from blake3 import blake3
from tqdm import tqdm

from picurate.common.images import is_image_path, list_directory_relative
from picurate.common.log import setup_logging

logger = logging.getLogger(__name__)

PIL.Image.MAX_IMAGE_PIXELS = None

SCHEMA_VERSION = "1"
INDEX = "index.sqlite"
FORMAT = "webp-lossless"
DEFAULT_BUDGET = 2 * 1024 * 16 * 16
DEFAULT_SHARD_BYTES = 16 * 2**30
WEBP = {"format": "WEBP", "lossless": True, "method": 4, "quality": 50}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS images (
    id            INTEGER PRIMARY KEY,
    key           TEXT NOT NULL UNIQUE,   -- the source's name for the image (a relative path, a post ID, ...)
    shard         INTEGER NOT NULL,
    offset        INTEGER NOT NULL,
    length        INTEGER NOT NULL,
    width         INTEGER NOT NULL,       -- as stored
    height        INTEGER NOT NULL,
    source_width  INTEGER NOT NULL,
    source_height INTEGER NOT NULL,
    blake3        BLOB NOT NULL           -- of the stored bytes
);
CREATE TABLE IF NOT EXISTS tags (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, category TEXT);
CREATE TABLE IF NOT EXISTS image_tags (
    image    INTEGER NOT NULL REFERENCES images (id),
    position INTEGER NOT NULL,
    tag      INTEGER NOT NULL REFERENCES tags (id),
    PRIMARY KEY (image, position)
) WITHOUT ROWID;
"""


def shard_path(directory: Path, shard: int) -> Path:
    return directory / f"pack-{shard:05d}.bin"


@dataclass(slots=True)
class Encoded:
    data: bytes
    width: int
    height: int
    source_width: int
    source_height: int


def encode(image: PIL.Image.Image, budget: int = DEFAULT_BUDGET) -> Encoded:
    """``image`` upright, RGB on white, downscaled to at most ``budget`` pixels, as lossless WebP."""
    # picurate.common.images turns this on for the deduper's leniency; a pack stores only complete images.
    PIL.ImageFile.LOAD_TRUNCATED_IMAGES = False
    image = PIL.ImageOps.exif_transpose(image)
    source = image.size
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        image = PIL.Image.new("RGB", rgba.size, (255, 255, 255))
        image.paste(rgba, mask=rgba.getchannel("A"))
    else:
        image = image.convert("RGB")
    scale = min(1.0, (budget / (image.width * image.height)) ** 0.5)
    if scale < 1.0:
        size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        image = image.resize(size, PIL.Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, **WEBP)
    return Encoded(buffer.getvalue(), image.width, image.height, *source)


def encode_file(path, budget: int) -> Encoded:
    """``path`` is a file name or a binary file object. Every image is decoded at full resolution (no JPEG draft scaling) before the Lanczos downscale."""
    with PIL.Image.open(path) as image:
        return encode(image, budget)


class PackWriter:
    """Appends images to a pack. One writer at a time."""

    def __init__(self, directory, *, budget: int = DEFAULT_BUDGET, shard_bytes: int = DEFAULT_SHARD_BYTES, commit_every: int = 500):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.directory / INDEX)
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.executescript(_SCHEMA)
        meta = dict(self.conn.execute("SELECT key, value FROM meta"))
        wanted = {"schema_version": SCHEMA_VERSION, "format": FORMAT, "budget": str(budget), "shard_bytes": str(shard_bytes)}
        if meta and meta != wanted:
            self.conn.close()
            raise ValueError(f"{self.directory} is a pack with other settings: {meta}")
        with self.conn:
            self.conn.executemany("INSERT OR REPLACE INTO meta VALUES (?, ?)", wanted.items())
        self.budget, self.shard_bytes, self.commit_every = budget, shard_bytes, commit_every
        self.tag_ids = dict(self.conn.execute("SELECT name, id FROM tags"))
        # Resume: the last shard ends where its last committed image ends; anything after it was never committed.
        row = self.conn.execute(
            "SELECT shard, MAX(offset + length) FROM images WHERE shard = (SELECT MAX(shard) FROM images)"
        ).fetchone()
        self.shard, end = (row[0], row[1]) if row[0] is not None else (0, 0)
        for path in sorted(self.directory.glob("pack-*.bin")):
            number = int(path.stem.split("-")[1])
            if number > self.shard:
                os.truncate(path, 0)
        self.stream = open(shard_path(self.directory, self.shard), "ab")
        self.stream.truncate(end)
        self.stream.seek(end)
        self.pending = 0
        self.conn.execute("BEGIN")

    def __contains__(self, key: str) -> bool:
        return self.conn.execute("SELECT 1 FROM images WHERE key = ?", (key,)).fetchone() is not None

    def keys(self) -> set[str]:
        return {row[0] for row in self.conn.execute("SELECT key FROM images")}

    def add(self, key: str, encoded: Encoded, tags: Sequence[tuple[str, str | None]] = ()) -> None:
        """Append one image with its tags (``(name, category)`` in source order)."""
        offset = self.stream.tell()
        if offset and offset + len(encoded.data) > self.shard_bytes:
            self._sync()
            self.stream.close()
            self.shard += 1
            self.stream = open(shard_path(self.directory, self.shard), "wb")
            offset = 0
        self.stream.write(encoded.data)
        cursor = self.conn.execute(
            "INSERT INTO images (key, shard, offset, length, width, height, source_width, source_height, blake3)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (key, self.shard, offset, len(encoded.data), encoded.width, encoded.height, encoded.source_width,
             encoded.source_height, blake3(encoded.data).digest()),
        )  # fmt: skip
        image = cursor.lastrowid
        for position, (name, category) in enumerate(tags):
            tag = self.tag_ids.get(name)
            if tag is None:
                tag = self.tag_ids[name] = self.conn.execute(
                    "INSERT INTO tags (name, category) VALUES (?, ?)", (name, category)
                ).lastrowid
            self.conn.execute("INSERT INTO image_tags VALUES (?, ?, ?)", (image, position, tag))
        self.pending += 1
        if self.pending >= self.commit_every:
            self.commit()

    def _sync(self) -> None:
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def commit(self) -> None:
        """Make everything added so far durable: shard bytes first, then the index rows pointing at them."""
        self._sync()
        self.conn.execute("COMMIT")
        self.conn.execute("BEGIN")
        self.pending = 0

    def close(self) -> None:
        self.commit()
        self.conn.execute("COMMIT")
        self.stream.close()
        self.conn.close()


class Pack:
    """Random access to a pack, image ``i`` being the ``i``-th key in byte order: two packs of the same images number
    them alike, whatever order parallel encoding stored them in. Picklable, and safe in forked worker processes: files
    and the index connection are opened per process on first use."""

    def __init__(self, directory):
        self.directory = Path(directory)
        conn = self.connect()
        try:
            meta = dict(conn.execute("SELECT key, value FROM meta"))
            if meta.get("schema_version") != SCHEMA_VERSION or meta.get("format") != FORMAT:
                raise ValueError(f"{self.directory} is not a pack this version reads: {meta}")
            self.meta = meta
            rows = conn.execute("SELECT id, key, shard, offset, length, width, height FROM images ORDER BY key").fetchall()
        finally:
            conn.close()
        self.ids = np.array([r[0] for r in rows], dtype=np.int64)
        self.keys = [r[1] for r in rows]
        self.shards, self.offsets, self.lengths, self.widths, self.heights = (
            np.array([r[k] for r in rows], dtype=np.int64) for k in range(2, 7)
        )
        self._pid = None

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(f"{(self.directory / INDEX).resolve().as_uri()}?mode=ro", uri=True, check_same_thread=False)

    def _local(self) -> None:
        if self._pid != os.getpid():
            self._pid = os.getpid()
            self._files: dict[int, int] = {}
            self._conn = self.connect()

    def __getstate__(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k not in ("_files", "_conn", "_pid")} | {"_pid": None}

    def __len__(self) -> int:
        return len(self.ids)

    def read(self, i: int) -> bytes:
        self._local()
        shard = int(self.shards[i])
        fd = self._files.get(shard)
        if fd is None:
            fd = self._files[shard] = os.open(shard_path(self.directory, shard), os.O_RDONLY)
        length = int(self.lengths[i])
        data = os.pread(fd, length, int(self.offsets[i]))
        if len(data) != length:
            raise OSError(f"{shard_path(self.directory, shard)}: image {self.keys[i]} is cut short")
        return data

    def image(self, i: int) -> PIL.Image.Image:
        with PIL.Image.open(io.BytesIO(self.read(i))) as image:
            return image.convert("RGB")

    def tags(self, i: int) -> list[tuple[str, str | None]]:
        self._local()
        return self._conn.execute(
            "SELECT t.name, t.category FROM image_tags it JOIN tags t ON t.id = it.tag WHERE it.image = ? ORDER BY it.position",
            (int(self.ids[i]),),
        ).fetchall()

    def digest(self, i: int) -> bytes:
        self._local()
        return self._conn.execute("SELECT blake3 FROM images WHERE id = ?", (int(self.ids[i]),)).fetchone()[0]


# -- command line ---------------------------------------------------------------------------------------------------


def _encode_job(path: str, budget: int, tags: list[tuple[str, str | None]]):
    try:
        return encode_file(path, budget), tags
    except Exception as error:  # noqa: BLE001 - reported per file
        return error, tags


def _txt_tags(path: str) -> list[tuple[str, str | None]]:
    caption = os.path.splitext(path)[0] + ".txt"
    if not os.path.exists(caption):
        return []
    with open(caption, encoding="utf-8") as stream:
        return [(tag.strip(), None) for tag in stream.read().split(",") if tag.strip()]


def _ignore_sigint() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl-C is the main process's: it commits what is done and stops


def bounded_map(function, jobs: Iterable[tuple], workers: int) -> Iterator[tuple[tuple, object]]:
    """``(job, function(*job))`` for every job, in completion order, with at most ``2 * workers`` in flight."""
    with ProcessPoolExecutor(workers, initializer=_ignore_sigint) as pool:
        running = {}
        jobs = iter(jobs)
        exhausted = False
        while running or not exhausted:
            while not exhausted and len(running) < 2 * workers:
                job = next(jobs, None)
                if job is None:
                    exhausted = True
                else:
                    running[pool.submit(function, *job)] = job
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                yield running.pop(future), future.result()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--verbose", "-v", is_flag=True, help="Debug logging.")
def main(verbose: bool) -> None:
    """Build and check training image packs (lossless WebP shards with an SQLite index)."""
    setup_logging(verbose, ("picurate", "clip_training"))


@main.command("import-dir")
@click.argument("pack_dir", type=click.Path(file_okay=False))
@click.argument("source", type=click.Path(exists=True, file_okay=False))
@click.option("--recursive", "-r", is_flag=True, help="Include subdirectories.")
@click.option("--txt-tags", is_flag=True, help="Read comma-separated tags from a same-stem .txt next to each image.")
@click.option(
    "--budget", type=click.IntRange(1024), default=DEFAULT_BUDGET, show_default=True, help="Maximum pixels per stored image."
)
@click.option("--shard-gib", type=click.FloatRange(0.001), default=16.0, show_default=True, help="Maximum shard size.")
@click.option("--workers", "-j", type=click.IntRange(1), default=os.cpu_count(), show_default=True, help="Encoding processes.")
def import_dir(pack_dir, source, recursive, txt_tags, budget, shard_gib, workers):
    """Add the images under SOURCE to the pack at PACK_DIR (created if needed), keyed by their relative path. Images
    already in the pack are skipped, so an interrupted run is simply run again."""
    writer = PackWriter(pack_dir, budget=budget, shard_bytes=int(shard_gib * 2**30))
    try:
        done = writer.keys()
        paths = [p for p in list_directory_relative(source, recursive=recursive) if is_image_path(p) and p not in done]
        click.echo(f"{len(paths)} images to add, {len(done)} already in the pack", err=True)
        jobs = ((os.path.join(source, p), budget, _txt_tags(os.path.join(source, p)) if txt_tags else []) for p in paths)
        failed = 0
        with tqdm(total=len(paths), unit="img", desc="Packing") as progress:
            for (path, _, _), (result, tags) in bounded_map(_encode_job, jobs, workers):
                if isinstance(result, Exception):
                    failed += 1
                    logger.warning("%s: %s", path, result)
                else:
                    writer.add(os.path.relpath(path, source), result, tags)
                progress.update()
    finally:
        writer.close()
    click.echo(f"Done; {failed} images could not be read" if failed else "Done", err=True)


@main.command("import-cagliostro")
@click.argument("pack_dir", type=click.Path(file_okay=False))
@click.argument("dataset", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--categories",
    "categories_db",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Tag database to take tag categories from (made by cagliostro-tags).",
)
@click.option(
    "--budget", type=click.IntRange(1024), default=DEFAULT_BUDGET, show_default=True, help="Maximum pixels per stored image."
)
@click.option("--shard-gib", type=click.FloatRange(0.001), default=16.0, show_default=True, help="Maximum shard size.")
@click.option("--workers", "-j", type=click.IntRange(1), default=os.cpu_count(), show_default=True, help="Encoding processes.")
@click.option(
    "--settle",
    type=click.FloatRange(0),
    default=300,
    show_default=True,
    help="Skip pairs with a file modified less than this many seconds ago (still being extracted).",
)
@click.option("--limit", type=click.IntRange(1), default=None, help="Add at most this many posts (a trial run).")
def import_cagliostro(pack_dir, dataset, categories_db, budget, shard_gib, workers, settle, limit):
    """Add the image/caption pairs of the cagliostrolab-860k-ordered dataset at DATASET (every directory below it)
    to the pack at PACK_DIR, with categorized tags (see clip_training.cagliostro). Posts already in
    the pack are skipped, so an interrupted run, or one over a partly extracted dataset, is simply run again."""
    from .cagliostro import categorize, category_dictionary, encode_pair, find_pairs

    try:
        dictionary = category_dictionary(categories_db)
    except (sqlite3.Error, ValueError) as error:
        raise click.ClickException(f"{categories_db}: {error}") from error
    click.echo(f"{len(dictionary)} tag categories from {categories_db}", err=True)
    writer = PackWriter(pack_dir, budget=budget, shard_bytes=int(shard_gib * 2**30))
    try:
        done = writer.keys()
        pairs, unsettled = [], 0
        cutoff = time.time() - settle
        for pair in find_pairs(dataset):
            if pair[0] in done:
                continue
            if max(os.stat(pair[1]).st_mtime, os.stat(pair[2]).st_mtime) > cutoff:
                unsettled += 1
            else:
                pairs.append(pair)
            if limit is not None and len(pairs) >= limit:
                break
        click.echo(
            f"{len(pairs)} posts to add, {len(done)} already in the pack, {unsettled} modified in the last {settle:g} s left for later",
            err=True,
        )
        failed = 0
        with tqdm(total=len(pairs), unit="img", desc="Packing") as progress:
            jobs = ((key, image, caption, budget) for key, image, caption in pairs)
            for (key, image, _, _), (result, tags) in bounded_map(encode_pair, jobs, workers):
                if isinstance(result, Exception):
                    failed += 1
                    logger.warning("%s: %s", image, result)
                else:
                    writer.add(key, result, categorize(tags, dictionary))
                progress.update()
    finally:
        writer.close()
    click.echo(f"Done; {failed} posts could not be read" if failed else "Done", err=True)


@main.command("check-cagliostro")
@click.argument("dataset", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--decode/--no-decode",
    default=True,
    show_default=True,
    help="Also decode images that match their MD5 (the others are always decoded).",
)
@click.option("--workers", "-j", type=click.IntRange(1), default=os.cpu_count(), show_default=True, help="Checking processes.")
def check_cagliostro(dataset, decode, workers):
    """Check the cagliostrolab-860k-ordered dataset at DATASET without writing anything: every caption readable and
    not empty, every image paired with a caption and decodable (one that matches the MD5 in its name only with
    --decode). Problems are printed to stdout, one ``path<TAB>problem`` per line; images that were saved again since
    (no longer matching their MD5) but decode are counted, not printed."""
    from .cagliostro import check_pair, find_pairs

    unpaired = []
    pairs = list(tqdm(find_pairs(dataset, unpaired), unit=" pairs", desc="Listing"))
    for path in unpaired:
        click.echo(f"{path}\tno {'image' if path.endswith('.txt') else 'caption'}")
    bad, statuses = 0, Counter()
    with tqdm(total=len(pairs), unit="img", desc="Checking") as progress:
        for (_, image, _, _), (problem, status) in bounded_map(check_pair, ((*pair, decode) for pair in pairs), workers):
            statuses[status] += 1
            if problem is not None:
                bad += 1
                click.echo(f"{image}\t{problem}")
            progress.update()
    checked = ", ".join(f"{count} {status}" for status, count in statuses.most_common())
    click.echo(f"{len(pairs)} pairs ({checked}), {bad} damaged, {len(unpaired)} files without their pair", err=True)
    if bad or unpaired:
        raise SystemExit(1)


@main.command("cagliostro-tags")
@click.argument("json_path", metavar="JSON", type=click.Path(exists=True, dir_okay=False))
@click.argument("output", type=click.Path(dir_okay=False))
def cagliostro_tags(json_path, output):
    """Convert the post list JSON that comes with cagliostrolab-860k-ordered (cagliostrolab-860k-ordered-tags.json)
    into a SQLite tag database at OUTPUT, which import-cagliostro takes tag categories from (see
    clip_training.cagliostro). OUTPUT must not exist; it is written next to it and renamed when complete."""
    from .cagliostro import convert_tag_json

    if os.path.exists(output):
        raise click.ClickException(f"{output} already exists")
    partial = output + ".partial"
    if os.path.exists(partial):
        os.remove(partial)
    counts = convert_tag_json(json_path, partial)
    os.replace(partial, output)
    for name, count in counts.items():
        click.echo(f"{name}: {count}", err=True)


@main.command()
@click.argument("pack_dir", type=click.Path(exists=True, file_okay=False))
def info(pack_dir):
    """Print what the pack at PACK_DIR holds."""
    pack = Pack(pack_dir)
    size = int(pack.lengths.sum())
    pixels = pack.widths * pack.heights
    click.echo(f"{len(pack)} images in {len(set(pack.shards.tolist()))} shards, {size / 2**30:.2f} GiB")
    if len(pack):
        click.echo(
            f"{size / len(pack) / 1024:.0f} KiB and {pixels.mean() / 1e3:.0f} kpx per image, {8 * size / pixels.sum():.2f} bit/px"
        )
    click.echo(f"Settings: {pack.meta}")


_packs: dict[str, Pack] = {}


def check_images(pack_dir: str, indices: list[int], decode: bool) -> list[tuple[int, str]]:
    """Worker: ``(index, problem)`` for the images of ``indices`` that do not match their BLAKE3 or (``decode``) do not
    decode to their indexed size. The pack is opened once per process."""
    pack = _packs.get(pack_dir) or _packs.setdefault(pack_dir, Pack(pack_dir))
    problems = []
    for i in indices:
        try:
            data = pack.read(i)
            if blake3(data).digest() != pack.digest(i):
                problems.append((i, "checksum mismatch"))
            elif decode:
                with PIL.Image.open(io.BytesIO(data)) as image:
                    image.load()
                    if image.size != (pack.widths[i], pack.heights[i]):
                        problems.append((i, f"size {image.size} differs from the index"))
        except Exception as error:  # noqa: BLE001 - reported per image
            problems.append((i, str(error)))
    return problems


@main.command()
@click.argument("pack_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--decode/--no-decode", default=True, show_default=True, help="Also decode every image.")
@click.option("--workers", "-j", type=click.IntRange(1), default=os.cpu_count(), show_default=True, help="Checking processes.")
def verify(pack_dir, decode, workers):
    """Read every image of the pack at PACK_DIR and check it against its BLAKE3 (and that it decodes)."""
    pack = Pack(pack_dir)
    # In storage order, so every process reads its shard stretch by stretch.
    order = np.lexsort((pack.offsets, pack.shards)).tolist()
    jobs = ((pack_dir, order[start : start + 256], decode) for start in range(0, len(order), 256))
    bad = 0
    with tqdm(total=len(pack), unit="img", desc="Verifying") as progress:
        for (_, indices, _), problems in bounded_map(check_images, jobs, workers):
            for i, problem in problems:
                logger.error("%s: %s", pack.keys[i], problem)
            bad += len(problems)
            progress.update(len(indices))
    if bad:
        raise click.ClickException(f"{bad} of {len(pack)} images are damaged")
    click.echo(f"All {len(pack)} images are intact", err=True)


if __name__ == "__main__":
    main()
