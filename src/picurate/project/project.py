"""Dataset projects: a directory that references external images and owns everything derived from them.

``project.json`` holds the settings and the source folders. ``project.sqlite`` holds the items (one per imported
image) and the human decisions about them, stored per field: a current-state table, and an append-only history in
which every change belongs to an operation, so that one editing session or one batch decision can be rolled back as a
unit. A field without a row has its default (for the caption: the derived one). Source images and their caption
files are only ever read.

An item is identified by ``(source_id, relative path)``: ``source_id`` is 256 random bits created when a source folder
is first imported (all zero for a standalone image directory), and the item ID is derived from both with BLAKE3.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from blake3 import blake3

from ..common.images import list_directory_relative

SCHEMA_VERSION = 3
# What a project imports, and what an export refuses to overwrite: static formats a training pipeline reads.
EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
# The source of a standalone image directory; never generated for a project source.
STANDALONE_SOURCE = bytes(32)
ITEM_ID_CONTEXT = "picurate item_id v1"
REVIEW_STATUSES = ("unreviewed", "approved", "skip")
# Human decision fields and their defaults. The caption's default is the derived caption, see Project.state.
DEFAULTS = {"caption": None, "crop": None, "fills": [], "review_status": "unreviewed", "exclusion": None}
COMMIT_EVERY = 1000

SCHEMA = """
CREATE TABLE items (
    id BLOB PRIMARY KEY,
    source_id BLOB NOT NULL,
    path TEXT NOT NULL,             -- relative to the source folder, '/'-separated
    size INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    blake3 BLOB NOT NULL,           -- of the file's contents
    caption TEXT,                   -- imported from caption_path, NULL if there was none
    caption_path TEXT,
    output_stem TEXT NOT NULL
);
CREATE TABLE ops (
    id INTEGER PRIMARY KEY,
    at REAL NOT NULL,
    kind TEXT NOT NULL,             -- edit, ...
    description TEXT
);
CREATE TABLE state (
    item BLOB NOT NULL,
    field TEXT NOT NULL,
    value TEXT NOT NULL,            -- JSON
    op INTEGER NOT NULL,
    PRIMARY KEY (item, field)
) WITHOUT ROWID;
CREATE TABLE history (
    seq INTEGER PRIMARY KEY,
    op INTEGER NOT NULL,
    item BLOB NOT NULL,
    field TEXT NOT NULL,
    value TEXT                      -- JSON; NULL = back to the default
);
CREATE INDEX history_item ON history (item, field, seq);
CREATE INDEX history_op ON history (op);
"""


def item_id(source_id: bytes, relative_path: str) -> bytes:
    data = source_id + b"\0" + relative_path.encode("utf-8", "surrogateescape")
    return blake3(data, derive_key_context=ITEM_ID_CONTEXT).digest()


def new_source_id() -> bytes:
    while (value := os.urandom(32)) == STANDALONE_SOURCE:
        pass
    return value


def file_digest(path) -> bytes:
    """BLAKE3 of a file's contents, read sequentially (parallel reads would make spinning disks seek)."""
    hasher = blake3()
    with open(path, "rb") as stream:
        while chunk := stream.read(1 << 20):
            hasher.update(chunk)
    return hasher.digest()


def safe_relative(value) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"Invalid project-relative path: {value}")
    return path


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class Project:
    """A project directory. Safe to use from one worker thread at a time besides the GUI thread."""

    def __init__(self, directory, create: bool = False):
        self.directory = Path(directory).expanduser().resolve()
        config = self.directory / "project.json"
        database = self.directory / "project.sqlite"
        if create:
            if self.directory.exists() and any(self.directory.iterdir()):
                raise ValueError("Choose an empty directory for a new project")
            self.directory.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(database) as db:
                db.executescript(SCHEMA)
            db.close()
            atomic_json(
                config,
                {
                    "schema_version": SCHEMA_VERSION,
                    "project_id": os.urandom(32).hex(),
                    "sources": {},
                    "format": "PNG",
                    "spaces": True,
                    "escape": False,
                },
            )
        if not config.exists():
            raise ValueError("No project.json found. Create a project with New Project, then import images.")
        self.settings = json.loads(config.read_text(encoding="utf-8"))
        if self.settings.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported project version {self.settings.get('schema_version')!r}; it was left untouched.")
        if not database.exists():
            raise ValueError(f"{database} is missing")
        self._lock = threading.RLock()
        self._lock_file = None
        self.db = sqlite3.connect(database, check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA journal_mode = WAL")
        self.db.execute("PRAGMA synchronous = NORMAL")
        self.sources = {bytes.fromhex(key): Path(value) for key, value in self.settings["sources"].items()}
        self.items = {}
        for row in self.db.execute(
            "SELECT id, source_id, path, size, mtime_ns, blake3, caption, caption_path, output_stem FROM items ORDER BY rowid"
        ):
            self.items[row[0]] = self._item(row)
        for name in ("masks", "patches", "output"):
            (self.directory / name).mkdir(parents=True, exist_ok=True)
        # Recover an interrupted directory swap without discarding the prior export.
        backup = self.directory / ".output-previous"
        if backup.exists():
            output = self.directory / "output"
            if not any(output.iterdir()):
                output.rmdir()
                backup.rename(output)
            else:
                shutil.rmtree(backup)

    @staticmethod
    def _item(row) -> dict:
        safe_relative(row[2])
        safe_relative(row[8])
        return dict(zip(("id", "source_id", "path", "size", "mtime_ns", "blake3", "caption", "caption_path", "output_stem"), row))

    def lock(self) -> None:
        """Hold the project for one editor. The OS releases the lock when the process ends, however it ends."""
        stream = open(self.directory / "project.lock", "a")
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.close()
            raise ValueError("The project is open in another editor") from None
        self._lock_file = stream

    def close(self) -> None:
        if self._lock_file is not None:
            self._lock_file.close()
            self._lock_file = None
        self.db.close()

    def save_settings(self) -> None:
        self.settings["sources"] = {key.hex(): str(value) for key, value in self.sources.items()}
        atomic_json(self.directory / "project.json", self.settings)

    def source_path(self, item: bytes) -> Path:
        row = self.items[item]
        return self.sources[row["source_id"]] / safe_relative(row["path"])

    def import_images(
        self,
        paths: Iterable,
        *,
        captions: bool = True,
        recursive: bool = False,
        include_hidden: bool = False,
        progress: Callable[[int, int], None] = lambda done, total: None,
    ) -> tuple[int, int]:
        """Import the images in directories (with ``recursive``, also in their subdirectories) and individual files. With ``captions``, a same-stem ``.txt`` becomes the
        image's imported caption. An image already in the project, through any source folder that contains it, is
        skipped and keeps its edits. Hidden files and directories inside an imported directory are skipped unless
        ``include_hidden``. Every new image is hashed, so this reads all of them once; progress is committed every
        1000 images, and running the import again resumes it. Returns ``(imported, skipped)``."""
        known = set(self.items)
        names = {row["output_stem"].casefold() for row in self.items.values()}
        candidates = []
        new_sources = {}  # recorded once an image is actually imported from them
        for entry in paths:
            entry = Path(entry).expanduser().resolve()
            root = entry if entry.is_dir() else entry.parent
            if entry.is_dir():
                listing = list_directory_relative(str(entry), recursive=recursive, include_hidden=include_hidden)
                files = sorted(entry / name for name in listing)
            else:
                files = [entry]
            known_sources = {**self.sources, **{key: value for value, key in new_sources.items()}}
            source_id = next((key for key, value in known_sources.items() if value == root), None)
            if source_id is None:
                source_id = new_sources[root] = new_source_id()
            candidates += [(root, source_id, path) for path in files if path.suffix.lower() in EXTENSIONS]

        def present(path: Path) -> bool:
            return any(
                path.is_relative_to(root) and item_id(sid, path.relative_to(root).as_posix()) in known
                for sid, root in self.sources.items()
            )

        imported = skipped = 0
        rows = []
        for done, (root, source_id, path) in enumerate(candidates):
            progress(done, len(candidates))
            # Never ingest generated artifacts from this project.
            if path.is_relative_to(self.directory) or not path.is_file():
                continue
            if present(path):
                skipped += 1
                continue
            relative = path.relative_to(root).as_posix()
            identifier = item_id(source_id, relative)
            stem = str(Path(relative).with_suffix(""))
            if stem.casefold() in names:
                stem += "__" + identifier.hex()[:12]
            while stem.casefold() in names:
                stem += "_"
            caption = caption_path = None
            if captions and (candidate := path.with_suffix(".txt")).exists():
                try:
                    caption = candidate.read_text(encoding="utf-8-sig")
                except UnicodeDecodeError as error:
                    raise ValueError(f"Caption is not UTF-8: {candidate}") from error
                caption_path = candidate.relative_to(root).as_posix()
            if source_id not in self.sources:
                # Publishing a source folder before its items is safe if interrupted.
                self.sources[source_id] = root
                self.save_settings()
            stat = path.stat()
            rows.append(
                (identifier, source_id, relative, stat.st_size, stat.st_mtime_ns, file_digest(path), caption, caption_path, stem)
            )
            names.add(stem.casefold())
            known.add(identifier)
            imported += 1
            if len(rows) == COMMIT_EVERY:
                self._insert_items(rows)
                rows = []
        self._insert_items(rows)
        progress(len(candidates), len(candidates))
        return imported, skipped

    def _insert_items(self, rows: list[tuple]) -> None:
        with self._lock:
            self.db.execute("BEGIN")
            self.db.executemany("INSERT INTO items VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
            self.db.execute("COMMIT")
            for row in rows:
                self.items[row[0]] = self._item(row)

    def relink(self, source_id: bytes, root) -> None:
        """Point a source folder elsewhere, after checking the contents of every image in it."""
        root = Path(root).expanduser().resolve()
        for row in self.items.values():
            if row["source_id"] == source_id:
                path = root / row["path"]
                if not path.is_file() or file_digest(path) != row["blake3"]:
                    raise ValueError(f"Missing or different source image: {path}")
        self.sources[source_id] = root
        self.save_settings()

    # Human decisions.

    def derived_caption(self, item: bytes) -> str:
        return self.items[item]["caption"] or ""

    def _decisions(self, item: bytes | None = None) -> dict[bytes, dict]:
        query = "SELECT item, field, value FROM state" + (" WHERE item = ?" if item is not None else "")
        decisions = {}
        with self._lock:
            for key, field, value in self.db.execute(query, () if item is None else (item,)):
                decisions.setdefault(key, {})[field] = json.loads(value)
        return decisions

    def _state(self, item: bytes, decisions: dict) -> dict:
        row = self.items[item]
        state = {
            "item_id": item,
            "image": row["path"],
            "source": {"size": row["size"], "mtime_ns": row["mtime_ns"], "blake3": row["blake3"].hex()},
            **{field: json.loads(json.dumps(default)) for field, default in DEFAULTS.items()},
            **decisions,
        }
        if state["caption"] is None:
            state["caption"] = self.derived_caption(item)
        state["excluded"] = (state["exclusion"] or {}).get("decision") == "exclude"
        return state

    def state(self, item: bytes) -> dict:
        """The effective state of an item: its decisions over the defaults, with the derived caption if unedited."""
        return self._state(item, self._decisions(item).get(item, {}))

    def states(self) -> list[dict]:
        decisions = self._decisions()
        return [self._state(item, decisions.get(item, {})) for item in self.items]

    def begin_op(self, kind: str, description: str = "") -> int:
        with self._lock:
            cursor = self.db.execute("INSERT INTO ops (at, kind, description) VALUES (?, ?, ?)", (time.time(), kind, description))
            assert cursor.lastrowid is not None
            return cursor.lastrowid

    def save(self, state: dict, op: int) -> list[str]:
        """Store the fields of ``state`` that differ from what is stored, as part of operation ``op``. A field set back
        to its default (the caption: to the derived one) loses its row. Within one operation, repeated saves of a field
        update the operation's history entry instead of adding one. Returns the changed fields."""
        item = state["item_id"]
        if item not in self.items:
            raise ValueError("Cannot save an unknown item")
        stored = self._decisions(item).get(item, {})
        changed = []
        with self._lock:
            self.db.execute("BEGIN")
            try:
                for field, default in DEFAULTS.items():
                    value = state[field]
                    if field == "caption" and value == self.derived_caption(item):
                        value = None
                    is_default = value == default
                    if (None if is_default else value) == stored.get(field):
                        continue
                    encoded = None if is_default else json.dumps(value, ensure_ascii=False)
                    if is_default:
                        self.db.execute("DELETE FROM state WHERE item = ? AND field = ?", (item, field))
                    else:
                        self.db.execute("INSERT OR REPLACE INTO state VALUES (?, ?, ?, ?)", (item, field, encoded, op))
                    last = self.db.execute(
                        "SELECT seq, op FROM history WHERE item = ? AND field = ? ORDER BY seq DESC LIMIT 1", (item, field)
                    ).fetchone()
                    if last is not None and last[1] == op:
                        self.db.execute("UPDATE history SET value = ? WHERE seq = ?", (encoded, last[0]))
                    else:
                        self.db.execute(
                            "INSERT INTO history (op, item, field, value) VALUES (?, ?, ?, ?)", (op, item, field, encoded)
                        )
                    changed.append(field)
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise
        return changed

    def export_snapshot(self, states, fmt: str) -> dict:
        return {
            "project_id": self.settings["project_id"],
            "format": fmt,
            "items": [
                {
                    "state": {**state, "item_id": state["item_id"].hex()},
                    "source_path": str(self.source_path(state["item_id"])),
                    "output_stem": self.items[state["item_id"]]["output_stem"],
                }
                for state in states
            ],
        }


def validate_source(path, state: dict) -> None:
    if file_digest(path).hex() != state["source"]["blake3"]:
        raise ValueError("Source file changed; review the edit recipe before continuing")
