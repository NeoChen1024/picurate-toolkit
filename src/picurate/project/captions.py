"""Captions combined on read: the imported caption and the one rendered from the project's tag scores.

``picurate-tag update`` writes ``tags.sqlite`` in the project and ``picurate-tag captions`` records there how its
captions are rendered (style, thresholds, ...) and combined with an imported caption:

* ``skip``: the imported caption if there is one, else the tags;
* ``overwrite``: the tags, whenever the image has scores;
* ``append``: the imported caption as it is, then the tags it lacks, compared exactly (sd-scripts' ``--append_tags``);
* ``merge``: the same, compared like the editor's duplicate highlighting (case, underscores and escapes ignored).

An image without scores (not tagged yet, excluded) keeps its imported caption. A caption a person edited overrides
all of it (:meth:`picurate.project.project.Project.state`). Nothing here imports torch.
"""

from __future__ import annotations

import logging
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

from ..tagger.captions import Captioner, CaptionOptions, Vocabulary, append_tags, duplicate_key
from ..tagger.tag_db import TagDB, TagDBMismatchError

logger = logging.getLogger(__name__)

TAGS = "tags.sqlite"  # written by picurate-tag; only read here
COMBINE_MODES = ("skip", "overwrite", "append", "merge")
DEFAULT_SETTINGS = {"options": CaptionOptions().to_json(), "combine": "skip"}
# Tolerance between the mtime the tagger saw (float seconds) and the one the project recorded (nanoseconds).
MTIME_SLACK = 1e-6

_vocabularies: dict[tuple, Vocabulary] = {}


@dataclass(slots=True, frozen=True)
class Generated:
    """An item's caption entries rendered from its tag scores, and the source mtime they were computed from."""

    entries: list[str]
    mtime: float


def settings(db: TagDB | None) -> tuple[CaptionOptions, str]:
    """The recorded caption options and combine mode, or the defaults."""
    recorded = (db.caption_options() if db is not None else None) or DEFAULT_SETTINGS
    return CaptionOptions.from_json(recorded["options"]), recorded["combine"]


def generated_captions(path: Path, items: Collection[bytes] | None = None) -> tuple[dict[bytes, Generated], CaptionOptions, str]:
    """Render the captions of ``items`` (None: all) from the tag database at ``path`` with its recorded options.
    Returns ``(captions, options, combine mode)``; without a usable database, no captions and the defaults."""
    if not path.exists():
        return {}, *settings(None)
    try:
        db = TagDB(str(path), readonly=True)
    except TagDBMismatchError as error:
        logger.warning("%s; its captions are ignored until 'picurate-tag update -f' rebuilds it", error)
        return {}, *settings(None)
    try:
        options, combine = settings(db)
        if db.get_meta("model_id") is None:
            return {}, options, combine
        key = (str(path), *db.binding_key())
        vocab = _vocabularies.get(key)
        if vocab is None:
            vocab = _vocabularies[key] = db.vocabulary()
        captioner = Captioner(vocab, options)
        out = {
            rec.id: Generated(captioner.caption(probs, rec.width, rec.height).entries, rec.mtime)
            for rec, probs in db.scores(items, len(vocab.tags))
        }
        return out, options, combine
    finally:
        db.conn.close()


def sidecar_digests(path: Path) -> dict[bytes, bytes]:
    """``{item: BLAKE3}`` of the caption files ``picurate-tag export`` wrote next to the images."""
    if not path.exists():
        return {}
    try:
        db = TagDB(str(path), readonly=True)
    except TagDBMismatchError:
        return {}
    try:
        return db.sidecars()
    finally:
        db.conn.close()


def combine(imported: str | None, generated: Generated | None, mode: str, separator: str) -> tuple[str, str | None]:
    """The derived caption and where it comes from: ``imported``, ``tags``, ``combined`` or None (empty)."""
    if generated is None or not generated.entries:
        return (imported, "imported") if imported else ("", None)
    tags = separator.join(generated.entries)
    if not imported or not imported.strip() or mode == "overwrite":
        return tags, "tags"
    if mode == "skip":
        return imported, "imported"
    text = append_tags(imported, generated.entries, separator, key=duplicate_key if mode == "merge" else str.strip)
    return (imported, "imported") if text == imported.rstrip() else (text, "combined")


def stale(generated: Generated, mtime_ns: int) -> bool:
    """Whether the scores were computed from another version of the file than the project recorded."""
    return abs(generated.mtime - mtime_ns / 1e9) > MTIME_SLACK
