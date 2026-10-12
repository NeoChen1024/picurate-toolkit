"""Dataset projects: a directory that references external images and owns everything derived from them.

``project.json`` holds the settings and the source folders with how they were imported, so that a sync can find
new, changed and vanished images. ``project.sqlite`` holds the items (one per imported
image) and the human decisions about them, stored per field: a current-state table, and a history in which every
change belongs to an operation and records the value before and after it, so that one editing session or one batch
decision can be rolled back as a unit, and old history can be pruned without breaking that. A field without a row has its default (for the caption: the derived one). Source images and their caption
files are only ever read. The project keeps no content hash: an import only stats the files. Once the images are
encoded for deduplication, export and relink check them against the content hashes in ``embeddings.sqlite``.

Other stages write only their own files in the project directory (``embeddings.sqlite``, ``review.sqlite``); the
project reads them and combines them with the human decisions (:meth:`Project.state`).

``gc --trash-excluded`` moves excluded images into ``trash/`` in the project, as an operation whose ``trashed``
field changes roll back like any other (the rollback moves the files back). The field is written before a file is
moved; :meth:`Project.reconcile_trash` finishes moves an interruption left halfway.

An item is identified by ``(source_id, relative path)`` (see :mod:`picurate.common.images`); a project's
``source_id`` is 256 random bits created when a source folder is first imported.
"""

from __future__ import annotations

import errno
import fcntl
import json
import logging
import os
import shutil
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from ..common.images import STANDALONE_SOURCE, PathFilter, file_digest, item_id, list_directory_relative
from ..dedupe.db_store import SCHEMA_VERSION as EMBEDDING_SCHEMA
from ..dedupe.review import SCHEMA_VERSION as REVIEW_SCHEMA
from ..dedupe.review import losers

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 5
# What a project imports, and what an export refuses to overwrite: static formats a training pipeline reads.
EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
REVIEW_STATUSES = ("unreviewed", "approved", "skip")
# Human decision fields and their defaults. The caption's default is the derived caption, see Project.state.
# image_size: the (EXIF-oriented) size the crop and fills were made on; a source of another size refuses them.
DEFAULTS = {"caption": None, "crop": None, "fills": [], "image_size": None, "review_status": "unreviewed", "exclusion": None}
COMMIT_EVERY = 1000
ITEM_COLUMNS = ("id", "source_id", "path", "size", "mtime_ns", "caption", "caption_path", "output_stem", "missing")
ARTIFACT_DIRS = ("masks", "patches")
TRASH = "trash"
TRASHED = "trashed"  # history field of gc --trash-excluded: "true" while the image is in the trash, else no row
MOVE_TYPES = ("dedupe", "human", "all")
EMBEDDINGS = "embeddings.sqlite"  # written by picurate-dedupe; only read here, for the content hashes
REVIEW = "review.sqlite"  # written by picurate-dedupe and the curation GUI; only read here, for the exclusions
LOCKS = {
    "project": "The project is open in another editor",
    "review": "The review database is in use (picurate-curate or a dedupe run)",
}

SCHEMA = """
CREATE TABLE items (
    id BLOB PRIMARY KEY,
    source_id BLOB NOT NULL,
    path TEXT NOT NULL,             -- relative to the source folder, '/'-separated
    size INTEGER NOT NULL,          -- as of the import or the last sync
    mtime_ns INTEGER NOT NULL,
    caption TEXT,                   -- imported from caption_path, NULL if there was none
    caption_path TEXT,
    output_stem TEXT NOT NULL,
    missing INTEGER NOT NULL DEFAULT 0  -- not found by the last sync
);
CREATE TABLE ops (
    id INTEGER PRIMARY KEY,
    at REAL NOT NULL,
    kind TEXT NOT NULL,             -- edit, rollback, ...
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
    previous TEXT,                  -- JSON before the operation; NULL = the default
    value TEXT                      -- JSON after it; NULL = the default
);
CREATE INDEX history_item ON history (item, field, seq);
CREATE INDEX history_op ON history (op);
"""


def new_source_id() -> bytes:
    while (value := os.urandom(32)) == STANDALONE_SOURCE:
        pass
    return value


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
        self._lock_files = {}
        self.db = sqlite3.connect(database, check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA journal_mode = WAL")
        self.db.execute("PRAGMA synchronous = NORMAL")
        self.sources = {bytes.fromhex(key): Path(value["path"]) for key, value in self.settings["sources"].items()}
        # How each source was imported: directory (else single files), recursive, include_hidden, captions.
        self.source_modes = {
            bytes.fromhex(key): {k: v for k, v in value.items() if k != "path"} for key, value in self.settings["sources"].items()
        }
        self.items = {}
        for row in self.db.execute(f"SELECT {', '.join(ITEM_COLUMNS)} FROM items ORDER BY rowid"):
            self.items[row[0]] = self._item(row)
        self.trashed = {row[0] for row in self.db.execute("SELECT item FROM state WHERE field = ?", (TRASHED,))}
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
        safe_relative(row[7])
        return dict(zip(ITEM_COLUMNS, row))

    def lock(self, name: str = "project") -> None:
        """Hold the project's decisions (``project``) or its review database (``review``) for one writer. The OS
        releases the lock when the process ends, however it ends."""
        if name in self._lock_files:
            return
        stream = open(self.directory / f"{name}.lock", "a")
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.close()
            raise ValueError(LOCKS[name]) from None
        self._lock_files[name] = stream

    def unlock(self, name: str) -> None:
        stream = self._lock_files.pop(name, None)
        if stream is not None:
            stream.close()

    def close(self) -> None:
        for name in list(self._lock_files):
            self.unlock(name)
        self.db.close()

    def save_settings(self) -> None:
        self.settings["sources"] = {
            key.hex(): {"path": str(value), **self.source_modes[key]} for key, value in self.sources.items()
        }
        atomic_json(self.directory / "project.json", self.settings)

    def source_path(self, item: bytes) -> Path:
        """Where the image is now: in its source folder, or in the project's trash."""
        return self.trash_path(item) if item in self.trashed else self.original_path(item)

    def original_path(self, item: bytes) -> Path:
        row = self.items[item]
        return self.sources[row["source_id"]] / safe_relative(row["path"])

    def trash_path(self, item: bytes) -> Path:
        """``trash/<source folder name>-<first 8 hex digits of its ID>/<path>``."""
        row = self.items[item]
        source = row["source_id"]
        return self.directory / TRASH / f"{self.sources[source].name}-{source.hex()[:8]}" / safe_relative(row["path"])

    def _caption_paths(self, item: bytes) -> tuple[Path, Path] | None:
        """The imported caption file in the source folder and its place in the trash, which it follows the image to."""
        row = self.items[item]
        if row["caption_path"] is None:
            return None
        relative = safe_relative(row["caption_path"])
        source = row["source_id"]
        return (
            self.sources[source] / relative,
            self.directory / TRASH / f"{self.sources[source].name}-{source.hex()[:8]}" / relative,
        )

    def import_images(
        self,
        paths: Iterable,
        *,
        captions: bool = True,
        recursive: bool = False,
        include_hidden: bool = False,
        filters: Sequence[tuple[str, str]] = (),
        progress: Callable[[int, int], None] = lambda done, total: None,
    ) -> tuple[int, int]:
        """Import the images in directories (with ``recursive``, also in their subdirectories) and individual files.
        With ``captions``, a same-stem ``.txt`` becomes the image's imported caption. An image already in the project,
        through any source folder that contains it, is skipped and keeps its edits. Hidden files and directories inside
        an imported directory are skipped unless ``include_hidden``, and so is whatever the rsync-style ``filters``
        (``("+" | "-", pattern)``, see :class:`~picurate.common.images.PathFilter`) exclude below a directory; files
        given by name are not filtered. Images are only statted, not read; progress is
        committed every 1000 images, and running the import again resumes it. How a source folder was imported is
        recorded for :meth:`sync` (a directory import widens what an earlier one recorded, and its filters replace
        earlier ones). Returns
        ``(imported, skipped)``."""
        known = set(self.items)
        names = {row["output_stem"].casefold() for row in self.items.values()}
        candidates = []
        new_sources = {}  # recorded once an image is actually imported from them
        modes = {}
        filters = [list(rule) for rule in filters]
        path_filter = PathFilter(filters)
        for entry in paths:
            entry = Path(entry).expanduser().resolve()
            root = entry if entry.is_dir() else entry.parent
            if entry.is_dir():
                listing = list_directory_relative(
                    str(entry), recursive=recursive, include_hidden=include_hidden, path_filter=path_filter
                )
                files = sorted(entry / name for name in listing)
            else:
                files = [entry]
            known_sources = {**self.sources, **{key: value for value, key in new_sources.items()}}
            source_id = next((key for key, value in known_sources.items() if value == root), None)
            if source_id is None:
                source_id = new_sources[root] = new_source_id()
            old = modes.get(source_id) or self.source_modes.get(source_id) or {}
            directory = entry.is_dir()
            modes[source_id] = {
                "directory": old.get("directory", False) or directory,
                "recursive": old.get("recursive", False) or (directory and recursive),
                "include_hidden": old.get("include_hidden", False) or (directory and include_hidden),
                "captions": captions,
                "filters": filters if directory else old.get("filters", []),
            }
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
                self.source_modes[source_id] = modes[source_id]
                self.save_settings()
            stat = path.stat()
            rows.append((identifier, source_id, relative, stat.st_size, stat.st_mtime_ns, caption, caption_path, stem, 0))
            names.add(stem.casefold())
            known.add(identifier)
            imported += 1
            if len(rows) == COMMIT_EVERY:
                self._insert_items(rows)
                rows = []
        self._insert_items(rows)
        changed = {key: mode for key, mode in modes.items() if key in self.sources and self.source_modes.get(key) != mode}
        if changed:
            self.source_modes.update(changed)
            self.save_settings()
        progress(len(candidates), len(candidates))
        return imported, skipped

    def sync(self, progress: Callable[[int, int], None] = lambda done, total: None) -> dict:
        """Bring the project up to date with its source folders: import new images of directory sources (as they
        were imported), record the size and mtime of changed images, and mark images that are gone as missing (or no
        longer missing). A source folder that is not there at all (unmounted, moved) is skipped and listed; relink it.
        Imported captions and every decision are kept. Images in the trash are not looked for in their source folders.
        Returns counts of ``added``, ``changed``, ``missing`` and ``back``, the unavailable source folders, and the
        trash files of trashed images that are gone (``trash_gone``; gc removes those items)."""
        counts = {"added": 0, "changed": 0, "missing": 0, "back": 0}
        unavailable = {key for key, root in self.sources.items() if not root.is_dir()}
        updates = []
        for done, (key, row) in enumerate(list(self.items.items())):
            progress(done, len(self.items))
            if row["source_id"] in unavailable or key in self.trashed:
                continue
            try:
                stat = (self.sources[row["source_id"]] / row["path"]).stat()
            except OSError:
                if not row["missing"]:
                    counts["missing"] += 1
                    updates.append((row["size"], row["mtime_ns"], 1, key))
                continue
            if row["missing"]:
                counts["back"] += 1
            elif (stat.st_size, stat.st_mtime_ns) != (row["size"], row["mtime_ns"]):
                counts["changed"] += 1
            else:
                continue
            updates.append((stat.st_size, stat.st_mtime_ns, 0, key))
        with self._lock:
            self.db.execute("BEGIN")
            self.db.executemany("UPDATE items SET size = ?, mtime_ns = ?, missing = ? WHERE id = ?", updates)
            self.db.execute("COMMIT")
        for size, mtime_ns, missing, key in updates:
            self.items[key].update(size=size, mtime_ns=mtime_ns, missing=missing)
        for key, mode in list(self.source_modes.items()):
            if mode["directory"] and key not in unavailable:
                added, _ = self.import_images(
                    [self.sources[key]],
                    captions=mode["captions"],
                    recursive=mode["recursive"],
                    include_hidden=mode["include_hidden"],
                    filters=mode["filters"],
                )
                counts["added"] += added
        progress(len(self.items), len(self.items))
        return {
            **counts,
            "unavailable": [self.sources[key] for key in unavailable],
            "trash_gone": [self.trash_path(item) for item in self.trashed if not self.trash_path(item).exists()],
        }

    def _insert_items(self, rows: list[tuple]) -> None:
        with self._lock:
            self.db.execute("BEGIN")
            self.db.executemany(
                f"INSERT INTO items ({', '.join(ITEM_COLUMNS)}) VALUES ({', '.join('?' * len(ITEM_COLUMNS))})", rows
            )
            self.db.execute("COMMIT")
            for row in rows:
                self.items[row[0]] = self._item(row)

    def relink(self, source_id: bytes, root) -> None:
        """Point a source folder elsewhere, after checking that every image in it is there with the same file size and,
        if the embedding database has it, the same content hash (images already marked missing may stay missing, but at
        least one image must be found)."""
        root = Path(root).expanduser().resolve()
        hashes = self.content_hashes()
        found = 0
        for item, row in self.items.items():
            if row["source_id"] == source_id and item not in self.trashed:
                path = root / row["path"]
                if path.is_file() and path.stat().st_size == row["size"]:
                    if item in hashes and file_digest(path) != hashes[item][1]:
                        raise ValueError(f"Source image with other contents than when it was encoded: {path}")
                    found += 1
                elif path.exists() or not row["missing"]:
                    raise ValueError(f"Missing or different source image: {path}")
        if not found:
            raise ValueError(f"None of the source's images are in {root}")
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

    def _review_losers(self, item: bytes | None = None) -> dict[bytes, tuple[int, list[bytes]]]:
        path = self.directory / REVIEW
        if not path.exists():
            return {}
        conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
        try:
            version = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
            if version is None or version[0] != REVIEW_SCHEMA:
                logger.warning("%s has another schema version; its decisions are ignored until matching runs again", path)
                return {}
            return losers(conn, item)
        finally:
            conn.close()

    def human_excluded(self) -> set[bytes]:
        """Items a person excluded (force exclude); they take no part in matching."""
        return {item for item, fields in self._decisions().items() if (fields.get("exclusion") or {}).get("decision") == "exclude"}

    def _state(self, item: bytes, decisions: dict) -> dict:
        row = self.items[item]
        state = {
            "item_id": item,
            "image": row["path"],
            "source": {"size": row["size"], "mtime_ns": row["mtime_ns"]},
            **{field: json.loads(json.dumps(default)) for field, default in DEFAULTS.items()},
            **decisions,
        }
        if state["caption"] is None:
            state["caption"] = self.derived_caption(item)
        return state

    @staticmethod
    def _human_exclusion(fields: dict) -> str | None:
        return (fields.get("exclusion") or {}).get("decision")

    def _combine_exclusion(self, state: dict, losers: dict, decisions: dict | None = None) -> None:
        """Excluded: in the trash; else the human decision (force keep / force exclude) if any, else a loser of a
        decided duplicate group, unless a person excluded every copy the group keeps. Sets ``excluded`` and ``excluded_by`` (None, or
        what excluded it). ``decisions`` holds every item's decisions, or None to look up the kept copies."""
        human = self._human_exclusion(state)
        state["excluded_by"] = None
        state["trashed"] = state["item_id"] in self.trashed
        if state["trashed"]:
            state["excluded_by"] = {"by": "trash"}
        elif human == "exclude":
            state["excluded_by"] = {"by": "human", "reason": state["exclusion"].get("reason", "")}
        elif human is None and state["item_id"] in losers:
            group, kept = losers[state["item_id"]]
            if decisions is None:
                decisions = {k: self._decisions(k).get(k, {}) for k in kept}
            if not kept or any(self._human_exclusion(decisions.get(k, {})) != "exclude" for k in kept):
                state["excluded_by"] = {"by": "dedupe", "group": group, "kept": [k.hex() for k in kept]}
        state["excluded"] = state["excluded_by"] is not None

    def update_exclusion(self, state: dict) -> None:
        """Recompute ``excluded`` and ``excluded_by`` after ``state["exclusion"]`` changed."""
        self._combine_exclusion(state, self._review_losers(state["item_id"]))

    def state(self, item: bytes) -> dict:
        """The effective state of an item: its decisions over the defaults, with the derived caption if unedited, and
        whether it is excluded (``excluded``, with ``excluded_by``)."""
        state = self._state(item, self._decisions(item).get(item, {}))
        self._combine_exclusion(state, self._review_losers(item))
        return state

    def states(self) -> list[dict]:
        decisions = self._decisions()
        losers = self._review_losers()
        states = [self._state(item, decisions.get(item, {})) for item in self.items]
        for state in states:
            self._combine_exclusion(state, losers, decisions)
        return states

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
        changed = []
        with self._lock:
            self.db.execute("BEGIN")
            try:
                for field, default in DEFAULTS.items():
                    value = state[field]
                    if field == "caption" and value == self.derived_caption(item):
                        value = None
                    encoded = None if value == default else json.dumps(value, ensure_ascii=False)
                    if self._write(item, field, encoded, op):
                        changed.append(field)
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise
        return changed

    def _stored(self, item: bytes, field: str) -> str | None:
        row = self.db.execute("SELECT value FROM state WHERE item = ? AND field = ?", (item, field)).fetchone()
        return None if row is None else row[0]

    def _write(self, item: bytes, field: str, encoded: str | None, op: int) -> bool:
        """Set one field to ``encoded`` (JSON, ``None`` for the default) as part of ``op``, inside the caller's
        transaction. Returns whether it changed."""
        before = self._stored(item, field)
        if encoded == before:
            return False
        if encoded is None:
            self.db.execute("DELETE FROM state WHERE item = ? AND field = ?", (item, field))
        else:
            self.db.execute("INSERT OR REPLACE INTO state VALUES (?, ?, ?, ?)", (item, field, encoded, op))
        if field == TRASHED:
            (self.trashed.add if encoded is not None else self.trashed.discard)(item)
        last = self.db.execute(
            "SELECT seq, op, previous FROM history WHERE item = ? AND field = ? ORDER BY seq DESC LIMIT 1", (item, field)
        ).fetchone()
        if last is not None and last[1] == op:
            if encoded == last[2]:  # back where the operation started: it did not change this field after all
                self.db.execute("DELETE FROM history WHERE seq = ?", (last[0],))
            else:
                self.db.execute("UPDATE history SET value = ? WHERE seq = ?", (encoded, last[0]))
        else:
            self.db.execute(
                "INSERT INTO history (op, item, field, previous, value) VALUES (?, ?, ?, ?, ?)", (op, item, field, before, encoded)
            )
        return True

    # Operations: listing, rollback and pruning.

    def ops(self, limit: int | None = None) -> list[dict]:
        """The most recent operations first, with the number of field changes each still has in the history."""
        rows = self.db.execute(
            "SELECT o.id, o.at, o.kind, o.description, COUNT(h.seq) FROM ops o LEFT JOIN history h ON h.op = o.id"
            " GROUP BY o.id ORDER BY o.id DESC" + (" LIMIT ?" if limit is not None else ""),
            () if limit is None else (limit,),
        )
        return [dict(zip(("id", "at", "kind", "description", "changes"), row)) for row in rows]

    def rollback(self, op: int) -> tuple[int, list[tuple[bytes, str]], int]:
        """Restore every field ``op`` changed to its value before ``op``, as a new operation. A field changed again
        since is left alone and listed. Returns ``(new operation, skipped (item, field) pairs, restored count)``."""
        with self._lock:
            row = self.db.execute("SELECT kind, description FROM ops WHERE id = ?", (op,)).fetchone()
            if row is None:
                raise ValueError(f"No operation {op}")
            changes = self.db.execute(
                "SELECT item, field, previous, value FROM history WHERE op = ? ORDER BY seq", (op,)
            ).fetchall()
            if not changes:
                raise ValueError(f"Operation {op} has no changes left in the history")
            self.reconcile_trash()
            self.db.execute("BEGIN")
            try:
                new = self.begin_op("rollback", f"rollback of {op} ({row[0]}{': ' + row[1] if row[1] else ''})")
                skipped = []
                restored = 0
                for item, field, before, value in changes:
                    if self._stored(item, field) != value or (field == TRASHED and not self._can_untrash(item, before)):
                        skipped.append((item, field))
                    elif self._write(item, field, before, new):
                        restored += 1
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                self.trashed = {r[0] for r in self.db.execute("SELECT item FROM state WHERE field = ?", (TRASHED,))}
                raise
            moved = self.reconcile_trash()
        return new, skipped, restored, moved

    def _can_untrash(self, item: bytes, before: str | None) -> bool:
        """Whether a rollback can set ``trashed`` back to ``before``: moving out of the trash needs the trash file
        (one deleted by hand means the image is gone; gc removes it) and a free place in the source folder."""
        if item not in self.items:
            return False
        if before is not None:  # rolling back a rollback: into the trash again
            return self.original_path(item).exists() or self.trash_path(item).exists()
        return self.trash_path(item).exists() and not self.original_path(item).exists()

    # The trash.

    def reconcile_trash(self) -> int:
        """Move every image (and its imported caption file) to where its ``trashed`` field says it is, when it is
        only at the other place: finishes the moves of an interrupted trash run or rollback. Returns the number
        moved."""
        moved = 0
        candidates = set(self.trashed) | self._in_trash_directory()
        for item in candidates:
            if item not in self.items:
                continue
            inside = item in self.trashed
            source, target = (self.original_path(item), self.trash_path(item))
            if not inside:
                source, target = target, source
            if source.exists() and not target.exists():
                move_file(source, target)
                moved += 1
            captions = self._caption_paths(item)
            if captions is not None:
                source, target = captions if inside else captions[::-1]
                if source.exists() and not target.exists():
                    move_file(source, target)
        if moved:
            logger.warning("Finished %d moves into or out of the trash that an interrupted run left undone", moved)
        return moved

    def _in_trash_directory(self) -> set[bytes]:
        """Items with a file in the trash, found by walking the trash rather than statting every item."""
        found = set()
        for source, root in self.sources.items():
            directory = self.directory / TRASH / f"{root.name}-{source.hex()[:8]}"
            for parent, _, files in os.walk(directory):
                for name in files:
                    relative = (Path(parent) / name).relative_to(directory).as_posix()
                    if (item := item_id(source, relative)) in self.items:
                        found.add(item)
        return found

    def trash_candidates(self, move_type: str = "all") -> list[dict]:
        """States of the excluded images ``gc --trash-excluded`` would move: excluded as duplicates (``dedupe``), by a
        person (``human``) or either (``all``), present in their source folder and not in the trash yet."""
        if move_type not in MOVE_TYPES:
            raise ValueError(f"Unknown move type {move_type!r}")
        wanted = {"dedupe", "human"} if move_type == "all" else {move_type}
        return [
            state
            for state in self.states()
            if state["excluded"] and state["excluded_by"]["by"] in wanted and self.original_path(state["item_id"]).is_file()
        ]

    def trash_excluded(self, move_type: str = "all", progress: Callable[[int, int], None] = lambda done, total: None) -> int:
        """Move the :meth:`trash_candidates` into the project's trash as one operation (see :meth:`rollback`), with
        their imported caption files. Returns the operation, or 0 when there was nothing to move."""
        self.reconcile_trash()
        candidates = self.trash_candidates(move_type)
        if not candidates:
            return 0
        op = self.begin_op("trash", f"{len(candidates)} excluded images ({move_type})")
        for done, state in enumerate(candidates):
            progress(done, len(candidates))
            item = state["item_id"]
            with self._lock:
                self.db.execute("BEGIN")
                self._write(item, TRASHED, "true", op)
                self.db.execute("COMMIT")
            try:
                move_file(self.original_path(item), self.trash_path(item))
            except OSError as error:
                logger.error("Cannot move %s to the trash: %s", self.original_path(item), error)
                with self._lock:
                    self.db.execute("BEGIN")
                    self._write(item, TRASHED, None, op)  # back where the operation started: no history entry
                    self.db.execute("COMMIT")
                continue
            captions = self._caption_paths(item)
            if captions is not None and captions[0].exists() and not captions[1].exists():
                move_file(*captions)
        progress(len(candidates), len(candidates))
        return op

    def trash_gone(self) -> list[bytes]:
        """Trashed items whose file was removed from the trash by hand (and is not back in the source folder)."""
        return [item for item in self.trashed if not self.trash_path(item).exists() and not self.original_path(item).exists()]

    def remove_items(self, items: Iterable[bytes]) -> None:
        """Forget items with their decisions and history (the image no longer exists)."""
        items = list(items)
        with self._lock:
            self.db.execute("BEGIN")
            for table, column in (("state", "item"), ("history", "item"), ("items", "id")):
                self.db.executemany(f"DELETE FROM {table} WHERE {column} = ?", [(item,) for item in items])
            self.db.execute("COMMIT")
        for item in items:
            self.items.pop(item, None)
            self.trashed.discard(item)

    def artifacts_in_use(self) -> set[str]:
        """Project-relative paths of the LaMa masks and patches that the current state or the history refers to."""
        used = set()
        rows = self.db.execute(
            "SELECT value FROM state WHERE field = 'fills' UNION ALL"
            " SELECT previous FROM history WHERE field = 'fills' UNION ALL SELECT value FROM history WHERE field = 'fills'"
        )
        for (encoded,) in rows:
            for fill in json.loads(encoded) if encoded else []:
                used.update(fill[key] for key in ("mask", "patch") if key in fill)
        return used

    def gc(self, *, keep_revisions: int | None = None, older_than: float | None = None, dry_run: bool = False) -> dict:
        """Prune history entries that are beyond the newest ``keep_revisions`` of their field and/or belong to an
        operation older than ``older_than`` seconds (both rules must hold when both are given), the operations left
        without entries, and the LaMa artifacts nothing refers to afterwards. Items whose image was removed from the
        trash by hand go first, with their decisions and history. Returns what was (or would be) removed."""
        gone = self.trash_gone()
        removed = [self.trash_path(item) for item in gone]
        if gone and not dry_run:
            self.remove_items(gone)
        conditions, args = [], []
        if keep_revisions is not None:
            conditions.append("(SELECT COUNT(*) FROM history n WHERE n.item = h.item AND n.field = h.field AND n.seq > h.seq) >= ?")
            args.append(keep_revisions)
        if older_than is not None:
            conditions.append("(SELECT at FROM ops WHERE id = h.op) < ?")
            args.append(time.time() - older_than)
        prune = "SELECT seq FROM history h WHERE " + (" AND ".join(conditions) if conditions else "0")
        empty_ops = "SELECT id FROM ops WHERE id NOT IN (SELECT op FROM history) AND id NOT IN (SELECT op FROM state)"
        with self._lock:
            self.db.execute("BEGIN")
            try:
                entries = self.db.execute(f"SELECT COUNT(*) FROM ({prune})", args).fetchone()[0]
                self.db.execute(f"DELETE FROM history WHERE seq IN ({prune})", args)
                ops = [row[0] for row in self.db.execute(empty_ops)]
                self.db.execute(f"DELETE FROM ops WHERE id IN ({empty_ops})")
                used = self.artifacts_in_use()
                self.db.execute("ROLLBACK" if dry_run else "COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise
        artifacts = sorted(
            path
            for name in ARTIFACT_DIRS
            for file in (self.directory / name).iterdir()
            if file.is_file() and (path := f"{name}/{file.name}") not in used
        )
        if not dry_run:
            for path in artifacts:
                (self.directory / path).unlink()
            self.db.execute("VACUUM")
        return {"entries": entries, "ops": ops, "artifacts": artifacts, "removed": removed}

    def content_hashes(self) -> dict[bytes, tuple[float, bytes]]:
        """``{item: (mtime, BLAKE3)}`` as of encoding, from the project's embedding database; empty without one."""
        path = self.directory / EMBEDDINGS
        if not path.exists():
            return {}
        conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
        try:
            version = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
            if version is None or version[0] != EMBEDDING_SCHEMA:
                logger.warning("%s has another schema version; source images are not checked against it", path)
                return {}
            return {row[0]: (row[1], row[2]) for row in conn.execute("SELECT id, mtime, blake3 FROM embeddings")}
        finally:
            conn.close()

    def export_snapshot(self, states, fmt: str) -> dict:
        hashes = self.content_hashes()
        return {
            "project_id": self.settings["project_id"],
            "format": fmt,
            "items": [
                {
                    "state": {**state, "item_id": state["item_id"].hex()},
                    "source_path": str(self.source_path(state["item_id"])),
                    "output_stem": self.items[state["item_id"]]["output_stem"],
                    "content": (
                        [hashes[state["item_id"]][0], hashes[state["item_id"]][1].hex()] if state["item_id"] in hashes else None
                    ),
                }
                for state in states
            ],
        }


def move_file(source: Path, target: Path) -> None:
    """Move a file, also across filesystems, without ever leaving neither copy complete: a cross-device move copies
    to a temporary name next to the target, checks the size, renames it into place and only then removes the
    source. Refuses to replace an existing target."""
    if target.exists():
        raise FileExistsError(errno.EEXIST, "Target exists", str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.rename(source, target)
        return
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
    temporary = target.with_name(f".{target.name}.picurate-partial")
    shutil.copy2(source, temporary)
    if temporary.stat().st_size != source.stat().st_size:
        temporary.unlink()
        raise OSError(f"Copy of {source} is incomplete")
    os.replace(temporary, target)
    source.unlink()


def validate_source(path, state: dict) -> str | None:
    """Raise if the source image is missing; return a warning if its size or mtime differs from the recorded one.
    Whether the edits still fit is decided by the image size (see :func:`check_image_size`)."""
    try:
        stat = Path(path).stat()
    except OSError as error:
        raise ValueError(f"Source image missing: {path}") from error
    if (stat.st_size, stat.st_mtime_ns) != (state["source"]["size"], state["source"]["mtime_ns"]):
        return f"Source file changed since import or the last sync: {path}"
    return None


def check_image_size(size: tuple[int, int], state: dict) -> None:
    """Refuse edits made on an image of another size than ``size``."""
    recorded = state.get("image_size")
    if recorded is not None and list(size) != list(recorded):
        raise ValueError(
            f"The source image is now {size[0]}×{size[1]}, but its crop and fills were made on {recorded[0]}×{recorded[1]}"
        )
