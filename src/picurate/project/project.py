"""Dataset projects: a directory that references external images and owns everything derived from them.

``project.json`` holds the settings and the source folders with how they were imported, so that a sync can find
new, changed and vanished images. ``project.sqlite`` holds the items (one per imported
image) and the human decisions about them, stored per field: a current-state table, and a history in which every
change belongs to an operation and records the value before and after it, so that one editing session or one batch
decision can be rolled back as a unit, and old history can be pruned without breaking that. A field without a row has its default (for the caption: the derived one). Source images and their caption
files are only ever read. The project keeps no content hash: an import only stats the files.

An item is identified by ``(source_id, relative path)`` (see :mod:`picurate.common.images`); a project's
``source_id`` is 256 random bits created when a source folder is first imported.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from blake3 import blake3

from ..common.images import STANDALONE_SOURCE, PathFilter, item_id, list_directory_relative

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
        self.sources = {bytes.fromhex(key): Path(value["path"]) for key, value in self.settings["sources"].items()}
        # How each source was imported: directory (else single files), recursive, include_hidden, captions.
        self.source_modes = {
            bytes.fromhex(key): {k: v for k, v in value.items() if k != "path"} for key, value in self.settings["sources"].items()
        }
        self.items = {}
        for row in self.db.execute(f"SELECT {', '.join(ITEM_COLUMNS)} FROM items ORDER BY rowid"):
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
        safe_relative(row[7])
        return dict(zip(ITEM_COLUMNS, row))

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
        self.settings["sources"] = {
            key.hex(): {"path": str(value), **self.source_modes[key]} for key, value in self.sources.items()
        }
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
        Imported captions and every decision are kept. Returns counts of ``added``, ``changed``, ``missing`` and
        ``back``, and the unavailable source folders."""
        counts = {"added": 0, "changed": 0, "missing": 0, "back": 0}
        unavailable = {key for key, root in self.sources.items() if not root.is_dir()}
        updates = []
        for done, (key, row) in enumerate(list(self.items.items())):
            progress(done, len(self.items))
            if row["source_id"] in unavailable:
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
        return {**counts, "unavailable": [self.sources[key] for key in unavailable]}

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
        """Point a source folder elsewhere, after checking that every image in it is there with the same file size
        (images already marked missing may stay missing, but at least one image must be found)."""
        root = Path(root).expanduser().resolve()
        found = 0
        for row in self.items.values():
            if row["source_id"] == source_id:
                path = root / row["path"]
                if path.is_file() and path.stat().st_size == row["size"]:
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
            self.db.execute("BEGIN")
            try:
                new = self.begin_op("rollback", f"rollback of {op} ({row[0]}{': ' + row[1] if row[1] else ''})")
                skipped = []
                restored = 0
                for item, field, before, value in changes:
                    if self._stored(item, field) != value:
                        skipped.append((item, field))
                    elif self._write(item, field, before, new):
                        restored += 1
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise
        return new, skipped, restored

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
        without entries, and the LaMa artifacts nothing refers to afterwards. Returns what was (or would be) removed."""
        if keep_revisions is None and older_than is None:
            raise ValueError("Give a number of revisions to keep, an age, or both")
        conditions, args = [], []
        if keep_revisions is not None:
            conditions.append("(SELECT COUNT(*) FROM history n WHERE n.item = h.item AND n.field = h.field AND n.seq > h.seq) >= ?")
            args.append(keep_revisions)
        if older_than is not None:
            conditions.append("(SELECT at FROM ops WHERE id = h.op) < ?")
            args.append(time.time() - older_than)
        prune = "SELECT seq FROM history h WHERE " + " AND ".join(conditions)
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
        return {"entries": entries, "ops": ops, "artifacts": artifacts}

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
