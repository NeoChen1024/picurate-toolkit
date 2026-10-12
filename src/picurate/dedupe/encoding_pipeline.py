"""Incremental encoding: find images that are new or changed, encode them through the shared decode pipeline, and
store the embeddings plus file metadata in the database.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence

from ..common.images import ImageRecord, is_image_path, walk_directory_relative
from ..common.pipeline import run_pipeline
from .db_store import EmbeddingDB, ModelMismatchError
from .encoder import CLIPImageEncoder, PendingBatch

logger = logging.getLogger(__name__)

__all__ = ["encode_images", "find_candidates", "is_image_path", "update_database"]


def find_candidates(
    image_dir: str, index: dict[str, float], *, force: bool = False, include_hidden: bool = False
) -> tuple[list[str], set[str]]:
    """Images under ``image_dir`` that need (re)encoding, and the set of all image paths seen.

    An image needs encoding when it has no row or its mtime differs from the stored one (``!=`` rather than ``>``,
    so a file replaced by an older copy is re-encoded too).
    """
    candidates: list[str] = []
    seen: set[str] = set()
    for rel in walk_directory_relative(image_dir, include_hidden=include_hidden):
        if not is_image_path(rel):
            continue
        seen.add(rel)
        try:
            mtime = os.path.getmtime(os.path.join(image_dir, rel))
        except OSError as e:
            logger.warning("Cannot stat %s: %s", rel, e)
            continue
        if force or index.get(rel) != mtime:
            candidates.append(rel)
    return candidates, seen


def encode_images(
    encoder: CLIPImageEncoder,
    db: EmbeddingDB,
    image_dir: str,
    candidates: Sequence[str],
    *,
    batch_size: int,
    workers: int | None = None,
    progress: Callable[[int], object] | None = None,
) -> int:
    """Encode ``candidates`` with ``run_pipeline`` and store their embeddings. Returns #stored."""

    def store(records: list[ImageRecord], pending: PendingBatch) -> None:
        db.upsert(zip(records, encoder.collect(pending)))

    return run_pipeline(
        image_dir,
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
    image_dir: str,
    db_path: str,
    *,
    force_update: bool = False,
    clean_orphans: bool = True,
    include_hidden: bool = False,
    batch_size: int = 4,
    workers: int | None = None,
    progress_factory: Callable[[int], Callable[[int], object]] | None = None,
) -> None:
    """Bring ``db_path`` up to date with the images under ``image_dir``.

    ``workers`` is the number of decoder processes (default: one per CPU). ``progress_factory(total)`` may return a
    callable that is invoked with increments as images are processed.
    """
    with EmbeddingDB(db_path) as db:
        try:
            db.ensure_model(encoder.model_id, reset_on_mismatch=force_update)
        except ModelMismatchError as e:
            raise RuntimeError(str(e)) from e

        index = db.load_index()
        candidates, seen = find_candidates(image_dir, index, force=force_update, include_hidden=include_hidden)
        logger.info("%d images, %d to encode", len(seen), len(candidates))
        if candidates:
            progress = progress_factory(len(candidates)) if progress_factory else None
            encode_images(encoder, db, image_dir, candidates, batch_size=batch_size, workers=workers, progress=progress)

        if clean_orphans:
            orphans = [p for p in index if p not in seen]
            for p in orphans:
                logger.info("Removing orphaned database entry: %s", p)
            db.delete(orphans)
