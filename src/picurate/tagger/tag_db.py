"""SQLite store of tagger scores, so captions can be rendered again (other style, thresholds, filters) without the GPU.

Layout:

* ``meta``: key/value. The binding (model id with revision, dtype, score floor) is checked on open, like the embedding
  database's model id: scores of a different model, precision or floor are never mixed into one file. The checkpoint's
  per-category thresholds are kept too, so exporting needs neither the model nor the network.
* ``tags``: the model's vocabulary, ``id`` = output index, Danbooru name, category (general/character/copyright/
  artist/meta/rating).
* ``images``: one row per tagged image, the same file metadata as the embedding database (relative path, mtime, size,
  width, height, format).
* ``scores``: ``(image_id, tag_id, prob)`` for every tag scoring at least the floor, plus all rating tags (the rating is
  an argmax). Small rows, so ``WITHOUT ROWID``; ``(tag_id, prob)`` is indexed for "which images have tag X" queries.

Each batch is written in one transaction (image rows and their scores together), so an interrupted update leaves only
complete images behind and the next update, comparing mtimes, carries on where it stopped.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from ..common.images import ImageRecord
from .captions import Vocabulary

SCHEMA_VERSION = "1"
default_score_floor = 0.05

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tags (
    id       INTEGER PRIMARY KEY,
    name     TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS images (
    id     INTEGER PRIMARY KEY,
    path   TEXT NOT NULL UNIQUE,
    mtime  REAL NOT NULL,
    size   INTEGER NOT NULL,
    width  INTEGER NOT NULL,
    height INTEGER NOT NULL,
    format TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS scores (
    image_id INTEGER NOT NULL REFERENCES images (id) ON DELETE CASCADE,
    tag_id   INTEGER NOT NULL REFERENCES tags (id),
    prob     REAL NOT NULL,
    PRIMARY KEY (image_id, tag_id)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS scores_tag ON scores (tag_id, prob);
"""

# Keys that must match for scores to be comparable.
_BINDING_KEYS = ("schema_version", "model_id", "dtype", "score_floor")


class TagDBMismatchError(RuntimeError):
    """The database holds scores of a different model, precision, floor or schema than requested."""


@dataclass(frozen=True, slots=True)
class Binding:
    model_id: str
    dtype: str
    score_floor: float

    def as_meta(self) -> dict[str, str]:
        return {"schema_version": SCHEMA_VERSION, "model_id": self.model_id, "dtype": self.dtype, "score_floor": repr(self.score_floor)}


class TagDB:
    def __init__(self, path: str, *, create: bool = True):
        if not create and not os.path.exists(path):
            raise FileNotFoundError(f"Tag database {path} does not exist")
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def __enter__(self) -> TagDB:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    # -- meta / binding --------------------------------------------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else row[0]

    def _set_meta(self, items: dict[str, str]) -> None:
        self.conn.executemany("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", items.items())

    @property
    def score_floor(self) -> float | None:
        value = self.get_meta("score_floor")
        return None if value is None else float(value)

    def bind(self, binding: Binding, vocab: Vocabulary, *, reset_on_mismatch: bool = False) -> None:
        """Bind a fresh database to ``binding`` and store the vocabulary; check an existing one against it.

        On a mismatch, ``reset_on_mismatch`` drops every row and rebinds (``--force-update``); otherwise it raises.
        """
        wanted = binding.as_meta()
        current = {k: self.get_meta(k) for k in _BINDING_KEYS}
        if current["model_id"] is not None:
            diff = {k: (current[k], wanted[k]) for k in _BINDING_KEYS if current[k] != wanted[k]}
            if not diff:
                return
            if not reset_on_mismatch:
                why = "; ".join(f"{k} is {have!r}, {want!r} requested" for k, (have, want) in diff.items())
                raise TagDBMismatchError(
                    f"Tag database {self.path}: {why}. Pass --force-update to re-tag everything, or use a different database file."
                )
        with self.conn:
            self.conn.execute("DELETE FROM scores")
            self.conn.execute("DELETE FROM images")
            self.conn.execute("DELETE FROM tags")
            self.conn.execute("DELETE FROM meta")
            category = {i: cat for cat, sl in vocab.slices.items() for i in range(sl.start, sl.stop)}
            self.conn.executemany("INSERT INTO tags (id, name, category) VALUES (?, ?, ?)", ((i, t, category[i]) for i, t in enumerate(vocab.tags)))
            self._set_meta({**wanted, "category_thresholds": json.dumps(vocab.thresholds)})

    def vocabulary(self) -> Vocabulary:
        """The stored vocabulary in the checkpoint-config shape ``Vocabulary`` reads."""
        rows = self.conn.execute("SELECT id, name, category FROM tags ORDER BY id").fetchall()
        if not rows:
            raise RuntimeError(f"Tag database {self.path} is empty; run 'picurate-tag update' first.")
        split: list[list] = []
        for i, (tag_id, _, cat) in enumerate(rows):
            if tag_id != i:
                raise RuntimeError(f"Tag database {self.path}: tag ids are not contiguous; it is corrupt.")
            if split and split[-1][0] == cat:
                split[-1][1] += 1
            else:
                split.append([cat, 1])
        thresholds = json.loads(self.get_meta("category_thresholds") or "{}")
        return Vocabulary.from_config({"tags": [r[1] for r in rows], "tags_split": split, "category_best_threshold": thresholds})

    # -- images ----------------------------------------------------------------------------------------------------

    def __len__(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]

    def load_index(self) -> dict[str, float]:
        """``{relative_path: mtime}`` of every tagged image."""
        return dict(self.conn.execute("SELECT path, mtime FROM images"))

    def store(self, rows: Iterable[tuple[ImageRecord, np.ndarray]], vocab: Vocabulary) -> None:
        """Replace the scores of these images with ``probs`` (one full output vector each), in one transaction."""
        floor = self.score_floor
        if floor is None:
            raise RuntimeError("Database is not bound to a model")
        rating = np.zeros(len(vocab.tags), dtype=bool)
        rating[vocab.slices["rating"]] = True
        with self.conn:
            for rec, probs in rows:
                self.conn.execute("DELETE FROM images WHERE path = ?", (rec.path,))
                cur = self.conn.execute(
                    "INSERT INTO images (path, mtime, size, width, height, format) VALUES (?, ?, ?, ?, ?, ?)",
                    (rec.path, float(rec.mtime), int(rec.size), int(rec.width), int(rec.height), rec.format),
                )
                idx = np.flatnonzero((probs >= floor) | rating)
                self.conn.executemany(
                    "INSERT INTO scores (image_id, tag_id, prob) VALUES (?, ?, ?)",
                    ((cur.lastrowid, int(i), float(probs[i])) for i in idx),
                )

    def delete(self, paths: Sequence[str]) -> None:
        with self.conn:
            self.conn.executemany("DELETE FROM images WHERE path = ?", ((p,) for p in paths))

    def iter_scores(self, num_tags: int) -> Iterator[tuple[ImageRecord, np.ndarray]]:
        """Every image ordered by path with its scores as a dense vector (0 below the floor)."""
        images = self.conn.execute("SELECT id, path, mtime, size, width, height, format FROM images ORDER BY path").fetchall()
        for image_id, *meta in images:
            probs = np.zeros(num_tags, dtype=np.float32)
            rows = self.conn.execute("SELECT tag_id, prob FROM scores WHERE image_id = ?", (image_id,)).fetchall()
            if rows:
                ids, values = zip(*rows)
                probs[list(ids)] = values
            yield ImageRecord(*meta), probs
