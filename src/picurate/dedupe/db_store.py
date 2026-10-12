"""SQLite-backed embedding store.

One database file holds every embedding of an image directory. Each row is keyed by the image's path relative to
that directory and stores the file metadata observed at encoding time (mtime, size, width, height, format) plus the
raw float16 embedding bytes. CLIP models run in fp16 on CUDA, so their outputs are exactly representable in fp16 and
storing them as fp32 would just double the size.

A ``meta`` table records the model id, embedding dimension and schema version so a database is never silently reused
with a different model or layout.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterable

import numpy as np

from ..common.images import ImageRecord

SCHEMA_VERSION = "2"
EMBEDDING_DTYPE = np.float16

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS embeddings (
    path      TEXT PRIMARY KEY,
    mtime     REAL NOT NULL,
    size      INTEGER NOT NULL,
    width     INTEGER NOT NULL,
    height    INTEGER NOT NULL,
    format    TEXT NOT NULL,
    embedding BLOB NOT NULL
);
"""
# Note: deliberately a rowid table. WITHOUT ROWID stores rows in an index b-tree whose small local payload limit pushes
# every ~2.5 KiB embedding into overflow pages (~60% file bloat and 4x slower writes).


class ModelMismatchError(RuntimeError):
    """The database was encoded with a different model or schema than requested."""


class EmbeddingDB:
    """Thin wrapper around a SQLite connection holding fixed-dimension embeddings."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def __enter__(self) -> EmbeddingDB:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    # -- meta ------------------------------------------------------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else row[0]

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, str(value)))

    @property
    def model_id(self) -> str | None:
        return self.get_meta("model_id")

    @property
    def dim(self) -> int | None:
        value = self.get_meta("dim")
        return None if value is None else int(value)

    def _bind(self, model_id: str) -> None:
        self.set_meta("schema_version", SCHEMA_VERSION)
        self.set_meta("model_id", model_id)
        self.set_meta("dtype", np.dtype(EMBEDDING_DTYPE).name)
        self.conn.commit()

    def ensure_model(self, model_id: str, *, reset_on_mismatch: bool = False) -> None:
        """Bind the database to ``model_id`` and the current schema.

        A fresh database adopts them. An existing database must match both, unless ``reset_on_mismatch`` is set, in
        which case all rows are dropped and the database is re-bound (used by ``--force-update``).
        """
        current_model, current_schema = self.model_id, self.get_meta("schema_version")
        if current_model is None:
            self._bind(model_id)
            return
        if current_model == model_id and current_schema == SCHEMA_VERSION:
            return
        if not reset_on_mismatch:
            if current_model != model_id:
                why = f"was encoded with model '{current_model}', but '{model_id}' was requested"
            else:
                why = f"uses schema version {current_schema}, but this version of the tool needs {SCHEMA_VERSION}"
            raise ModelMismatchError(
                f"Database {self.path} {why}. Pass --force-update to re-encode everything, or use a different database file."
            )
        self.conn.execute("DELETE FROM embeddings")
        self.conn.execute("DELETE FROM meta")
        self._bind(model_id)

    # -- rows ------------------------------------------------------------------------------------------------------

    def __len__(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]

    def load_index(self) -> dict[str, float]:
        """``{relative_path: mtime}`` for every stored embedding; what the incremental update compares against."""
        return dict(self.conn.execute("SELECT path, mtime FROM embeddings"))

    def upsert(self, rows: Iterable[tuple[ImageRecord, np.ndarray]]) -> None:
        """Insert or replace ``(record, embedding)`` pairs in a single transaction."""
        prepared = []
        for rec, embedding in rows:
            vec = np.ascontiguousarray(embedding, dtype=EMBEDDING_DTYPE).reshape(-1)
            dim = self.dim
            if dim is None:
                self.set_meta("dim", str(vec.shape[0]))
            elif vec.shape[0] != dim:
                raise ValueError(f"Embedding for {rec.path} has dimension {vec.shape[0]}, database expects {dim}")
            prepared.append((rec.path, float(rec.mtime), int(rec.size), int(rec.width), int(rec.height), rec.format, vec.tobytes()))
        if prepared:
            self.conn.executemany(
                "INSERT OR REPLACE INTO embeddings (path, mtime, size, width, height, format, embedding) VALUES (?, ?, ?, ?, ?, ?, ?)",
                prepared,
            )
        self.conn.commit()

    def delete(self, paths: Iterable[str]) -> int:
        """Delete rows by path; returns the number of rows removed."""
        paths = list(paths)
        if not paths:
            return 0
        cur = self.conn.executemany("DELETE FROM embeddings WHERE path = ?", ((p,) for p in paths))
        self.conn.commit()
        return cur.rowcount if cur.rowcount >= 0 else len(paths)

    def load_all(self) -> tuple[list[ImageRecord], np.ndarray]:
        """Every row as ``(records, (N, D) float16 array)``, ordered by path, with one memory copy for the matrix."""
        dim = self.dim
        rows = self.conn.execute(
            "SELECT path, mtime, size, width, height, format, embedding FROM embeddings ORDER BY path"
        ).fetchall()
        if not rows:
            return [], np.empty((0, dim or 0), dtype=EMBEDDING_DTYPE)
        if dim is None:
            raise RuntimeError(f"Database {self.path} has rows but no recorded dimension; it is corrupt.")
        records = [ImageRecord(*row[:6]) for row in rows]
        # bytearray.join yields a mutable buffer, so frombuffer below is a writable zero-copy view.
        blob = bytearray().join(row[6] for row in rows)
        expected = len(rows) * dim * np.dtype(EMBEDDING_DTYPE).itemsize
        if len(blob) != expected:
            raise RuntimeError(f"Database {self.path} embedding blobs total {len(blob)} bytes, expected {expected}; it is corrupt.")
        return records, np.frombuffer(blob, dtype=EMBEDDING_DTYPE).reshape(len(rows), dim)


def load_database(db_path: str) -> tuple[list[ImageRecord], np.ndarray]:
    """Load a database file, or an empty result if it does not exist yet."""
    if not os.path.exists(db_path):
        return [], np.empty((0, 0), dtype=EMBEDDING_DTYPE)
    with EmbeddingDB(db_path) as db:
        return db.load_all()
