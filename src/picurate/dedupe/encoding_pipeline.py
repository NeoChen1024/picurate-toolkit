"""Incremental encoding: find images that are new or changed, encode them through the shared decode pipeline, and
store the embeddings plus file metadata in the database.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence

import numpy as np

from ..common.images import ImageRecord, Item, Roots, file_digest
from ..common.pipeline import run_pipeline
from .db_store import EmbeddingDB, ModelMismatchError
from .encoder import CLIPImageEncoder, PendingBatch

logger = logging.getLogger(__name__)

__all__ = ["encode_images", "find_candidates", "reuse_embeddings", "update_database"]


def find_candidates(roots: Roots, items: Sequence[Item], index: dict[bytes, float], *, force: bool = False) -> list[Item]:
    """The ``items`` that need (re)encoding: no row, or a stored mtime different from the file's (``!=`` rather than
    ``>``, so a file replaced by an older copy is re-encoded too)."""
    candidates: list[Item] = []
    for item in items:
        try:
            mtime = os.path.getmtime(item.file(roots))
        except OSError as e:
            logger.warning("Cannot stat %s: %s", item.path, e)
            continue
        if force or index.get(item.id) != mtime:
            candidates.append(item)
    return candidates


def encode_images(
    encoder: CLIPImageEncoder,
    db: EmbeddingDB,
    roots: Roots,
    candidates: Sequence[Item],
    *,
    batch_size: int,
    workers: int | None = None,
    progress: Callable[[int], object] | None = None,
) -> int:
    """Encode ``candidates`` with ``run_pipeline`` and store their embeddings. Returns #stored."""

    def store(records: list[ImageRecord], pending: PendingBatch) -> None:
        db.upsert(zip(records, encoder.collect(pending)))

    return run_pipeline(
        roots,
        candidates,
        encoder.get_preprocessor(),
        encoder.submit,
        store,
        batch_size=batch_size,
        workers=workers,
        progress=progress,
    )


def reuse_embeddings(db: EmbeddingDB, roots: Roots, candidates: Sequence[Item], source_path: str) -> list[Item]:
    """Copy the embeddings of candidates whose content another database of the same model already holds (found by
    BLAKE3, so each candidate is read once to hash it). Returns the candidates still to encode."""
    if not os.path.exists(source_path):
        raise RuntimeError(f"No database at {source_path} to reuse embeddings from")
    with EmbeddingDB(source_path) as source:
        if source.model_id != db.model_id or (db.dim is not None and source.dim != db.dim):
            raise RuntimeError(f"{source_path} was encoded with model '{source.model_id}', not '{db.model_id}'; nothing to reuse")
        digests = {}
        for item in candidates:
            try:
                digests[item.id] = file_digest(item.file(roots))
            except OSError as e:
                logger.warning("Cannot read %s: %s", item.path, e)
        found = source.by_content(set(digests.values()))
    rows, rest = [], []
    for item in candidates:
        hit = found.get(digests.get(item.id, b""))
        if hit is None:
            rest.append(item)
            continue
        width, height, fmt, embedding = hit
        st = os.stat(item.file(roots))
        record = ImageRecord(item.path, st.st_mtime, st.st_size, width, height, fmt, item.source_id, item.id, digests[item.id])
        rows.append((record, np.frombuffer(embedding, dtype=np.float16)))
    db.upsert(rows)
    logger.info("Reused %d embeddings from %s; %d left to encode", len(rows), source_path, len(rest))
    return rest


def update_database(
    encoder: CLIPImageEncoder,
    roots: Roots,
    items: Sequence[Item],
    db_path: str,
    *,
    force_update: bool = False,
    clean_orphans: bool = True,
    batch_size: int = 4,
    workers: int | None = None,
    progress_factory: Callable[[int], Callable[[int], object]] | None = None,
    reuse_from: str | None = None,
) -> None:
    """Bring ``db_path`` up to date with ``items``.

    ``workers`` is the number of decoder processes (default: one per CPU). ``progress_factory(total)`` may return a
    callable that is invoked with increments as images are processed. With ``clean_orphans``, rows of images not in
    ``items`` are removed. ``reuse_from`` names another database whose rows are copied for files of the same content
    (see :func:`reuse_embeddings`).
    """
    with EmbeddingDB(db_path, rebuild=force_update) as db:
        try:
            db.ensure_model(encoder.model_id, reset_on_mismatch=force_update)
        except ModelMismatchError as e:
            raise RuntimeError(str(e)) from e

        index = db.load_index()
        candidates = find_candidates(roots, items, index, force=force_update)
        logger.info("%d images, %d to encode", len(items), len(candidates))
        if candidates and reuse_from:
            candidates = reuse_embeddings(db, roots, candidates, reuse_from)
        if candidates:
            progress = progress_factory(len(candidates)) if progress_factory else None
            encode_images(encoder, db, roots, candidates, batch_size=batch_size, workers=workers, progress=progress)

        if clean_orphans:
            seen = {item.id for item in items}
            orphans = [i for i in index if i not in seen]
            if orphans:
                logger.info("Removing %d database entries whose image is gone", len(orphans))
            db.delete(orphans)
