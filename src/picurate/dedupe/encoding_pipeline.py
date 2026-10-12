"""Incremental encoding: find images that are new or changed, encode them through the shared decode pipeline, and
store the embeddings plus file metadata in the database.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence

from ..common.images import ImageRecord, Item, Roots
from ..common.pipeline import run_pipeline
from .db_store import EmbeddingDB, ModelMismatchError
from .encoder import CLIPImageEncoder, PendingBatch

logger = logging.getLogger(__name__)

__all__ = ["encode_images", "find_candidates", "update_database"]


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
) -> None:
    """Bring ``db_path`` up to date with ``items``.

    ``workers`` is the number of decoder processes (default: one per CPU). ``progress_factory(total)`` may return a
    callable that is invoked with increments as images are processed. With ``clean_orphans``, rows of images not in
    ``items`` are removed.
    """
    with EmbeddingDB(db_path, rebuild=force_update) as db:
        try:
            db.ensure_model(encoder.model_id, reset_on_mismatch=force_update)
        except ModelMismatchError as e:
            raise RuntimeError(str(e)) from e

        index = db.load_index()
        candidates = find_candidates(roots, items, index, force=force_update)
        logger.info("%d images, %d to encode", len(items), len(candidates))
        if candidates:
            progress = progress_factory(len(candidates)) if progress_factory else None
            encode_images(encoder, db, roots, candidates, batch_size=batch_size, workers=workers, progress=progress)

        if clean_orphans:
            seen = {item.id for item in items}
            orphans = [i for i in index if i not in seen]
            if orphans:
                logger.info("Removing %d database entries whose image is gone", len(orphans))
            db.delete(orphans)
