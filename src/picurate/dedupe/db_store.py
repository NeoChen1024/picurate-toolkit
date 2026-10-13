"""SQLite-backed embedding store.

One database file holds every embedding of an image directory or project. Each row is keyed by the item ID (see
:mod:`picurate.common.images`), keeps the source ID and relative path it was derived from, and stores the file metadata
observed at encoding time (mtime, size, width, height, format, BLAKE3 of the contents) plus the raw float16 embedding
bytes. The content hash lets another database's rows be reused for the same file elsewhere, and lets a project verify
its sources. CLIP models run in fp16 on CUDA, so their outputs are exactly representable in fp16 and
storing them as fp32 would just double the size.

A ``meta`` table records the model id, embedding dimension and schema version so a database is never silently reused
with a different model or layout. A database of another schema version is refused; ``--force-update`` rebuilds it.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from ..common.images import ImageRecord

SCHEMA_VERSION = "4"
EMBEDDING_DTYPE = np.float16

_META = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""
_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    id        BLOB PRIMARY KEY,
    source_id BLOB NOT NULL,
    path      TEXT NOT NULL,
    mtime     REAL NOT NULL,
    size      INTEGER NOT NULL,
    width     INTEGER NOT NULL,
    height    INTEGER NOT NULL,
    format    TEXT NOT NULL,
    blake3    BLOB NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS embeddings_blake3 ON embeddings (blake3);
"""
# Note: deliberately a rowid table. WITHOUT ROWID stores rows in an index b-tree whose small local payload limit pushes
# every ~2.5 KiB embedding into overflow pages (~60% file bloat and 4x slower writes).


class ModelMismatchError(RuntimeError):
    """The database was encoded with a different model or schema than requested."""


class EmbeddingDB:
    """Thin wrapper around a SQLite connection holding fixed-dimension embeddings."""

    def __init__(self, path: str, *, rebuild: bool = False):
        """Open or create ``path``. A database of another schema version raises :class:`ModelMismatchError`, or with
        ``rebuild`` is emptied."""
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.executescript(_META)
        version = self.get_meta("schema_version")
        if version not in (None, SCHEMA_VERSION):
            if not rebuild:
                self.conn.close()
                raise ModelMismatchError(
                    f"Database {path} uses schema version {version}, but this version of the tool needs {SCHEMA_VERSION}. "
                    "Pass --force-update to re-encode everything, or use a different database file."
                )
            self.conn.execute("DROP TABLE IF EXISTS embeddings")
            self.conn.execute("DELETE FROM meta")
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
        current_model = self.model_id
        if current_model is None:
            self._bind(model_id)
            return
        if current_model == model_id:
            return
        if not reset_on_mismatch:
            raise ModelMismatchError(
                f"Database {self.path} was encoded with model '{current_model}', but '{model_id}' was requested. "
                "Pass --force-update to re-encode everything, or use a different database file."
            )
        self.conn.execute("DELETE FROM embeddings")
        self.conn.execute("DELETE FROM meta")
        self._bind(model_id)

    # -- rows ------------------------------------------------------------------------------------------------------

    def __len__(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]

    def load_index(self) -> dict[bytes, float]:
        """``{item_id: mtime}`` for every stored embedding; what the incremental update compares against."""
        return dict(self.conn.execute("SELECT id, mtime FROM embeddings"))

    def upsert(self, rows: Iterable[tuple[ImageRecord, np.ndarray]]) -> None:
        """Insert or replace ``(record, embedding)`` pairs in a single transaction."""
        prepared = []
        dim = self.dim
        for rec, embedding in rows:
            vec = np.ascontiguousarray(embedding, dtype=EMBEDDING_DTYPE).reshape(-1)
            if dim is None:
                dim = vec.shape[0]
                self.set_meta("dim", str(dim))
            elif vec.shape[0] != dim:
                raise ValueError(f"Embedding for {rec.path} has dimension {vec.shape[0]}, database expects {dim}")
            prepared.append(
                (
                    rec.id,
                    rec.source_id,
                    rec.path,
                    float(rec.mtime),
                    int(rec.size),
                    int(rec.width),
                    int(rec.height),
                    rec.format,
                    rec.blake3,
                    vec.tobytes(),
                )
            )
        if prepared:
            self.conn.executemany(
                "INSERT OR REPLACE INTO embeddings (id, source_id, path, mtime, size, width, height, format, blake3, embedding)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                prepared,
            )
        self.conn.commit()

    def delete(self, ids: Iterable[bytes]) -> int:
        """Delete rows by item ID; returns the number of rows removed."""
        ids = list(ids)
        if not ids:
            return 0
        cur = self.conn.executemany("DELETE FROM embeddings WHERE id = ?", ((i,) for i in ids))
        self.conn.commit()
        return cur.rowcount if cur.rowcount >= 0 else len(ids)

    def by_content(self, digests: Iterable[bytes]) -> dict[bytes, tuple[int, int, str, bytes]]:
        """``{blake3: (width, height, format, embedding bytes)}`` for the given content hashes that have a row."""
        out = {}
        for digest in digests:
            row = self.conn.execute(
                "SELECT width, height, format, embedding FROM embeddings WHERE blake3 = ? LIMIT 1", (digest,)
            ).fetchone()
            if row is not None:
                out[digest] = row
        return out

    def load_all(self) -> tuple[list[ImageRecord], np.ndarray]:
        """Every row as ``(records, (N, D) float16 array)``, ordered by source and path, with one memory copy for the
        matrix."""
        dim = self.dim
        rows = self.conn.execute(
            "SELECT path, mtime, size, width, height, format, source_id, id, blake3, embedding FROM embeddings"
            " ORDER BY source_id, path"
        ).fetchall()
        if not rows:
            return [], np.empty((0, dim or 0), dtype=EMBEDDING_DTYPE)
        if dim is None:
            raise RuntimeError(f"Database {self.path} has rows but no recorded dimension; it is corrupt.")
        records = [ImageRecord(*row[:9]) for row in rows]
        # bytearray.join yields a mutable buffer, so frombuffer below is a writable zero-copy view.
        blob = bytearray().join(row[9] for row in rows)
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


def load_rows(db_path: str, items: Sequence[bytes]) -> tuple[str, np.ndarray, np.ndarray]:
    """``(model id, (len(items), D) float16 matrix, bool mask of the rows found)``: the stored embeddings of ``items``
    in their order, read without writing the database and without a second copy of the matrix. Rows of other items are
    skipped; an item without a row gets zeros."""
    conn = sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)
    try:
        meta = dict(conn.execute("SELECT key, value FROM meta"))
        if meta.get("schema_version") != SCHEMA_VERSION:
            raise ModelMismatchError(f"Database {db_path} uses schema version {meta.get('schema_version')}, not {SCHEMA_VERSION}")
        if "model_id" not in meta or "dim" not in meta:
            raise ModelMismatchError(f"Database {db_path} holds no embeddings yet")
        dim = int(meta["dim"])
        position = {item: i for i, item in enumerate(items)}
        matrix = np.zeros((len(items), dim), dtype=EMBEDDING_DTYPE)
        found = np.zeros(len(items), dtype=bool)
        cursor = conn.execute("SELECT id, embedding FROM embeddings")
        while rows := cursor.fetchmany(4096):
            for item, blob in rows:
                i = position.get(item)
                if i is not None:
                    matrix[i] = np.frombuffer(blob, dtype=EMBEDDING_DTYPE)
                    found[i] = True
        return meta["model_id"], matrix, found
    finally:
        conn.close()
