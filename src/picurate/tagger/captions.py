"""Turning tagger scores into caption text, in one of two prompt conventions.

* ``anima``: the section order of circlestone-labs/Anima,
  ``[prefix/meta/safety] [1girl/1boy/1other] [character] [series] [@artist] [general]``, lowercase with spaces.
* ``sdxl``: what sd-scripts' ``tag_images_by_wd14_tagger.py`` writes with a WD14 tagger and ``--remove_underscore
  --use_rating_tags_as_last_tag --character_tags_first --character_tag_expand --always_first_tags "1girl,1boy"``:
  characters first (``name, series``), general tags in vocabulary order (Danbooru popularity: ``1girl, solo, long
  hair, ...``, not by confidence), the rating word last, and the always-first tags moved to the front one at a time
  as sd-scripts does it, so the last one listed ends up first (``1girl,1boy`` -> ``1boy, 1girl, ...``). WD14 has no series/artist/meta categories; here series and artists follow the characters (artists without
  ``@``) and meta tags come just before the rating.

Tags are handled under their Danbooru names (underscores) until the very end, so filters and replacements match one
spelling whatever the output style.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

STYLES = ("anima", "sdxl")
# Our category names; the checkpoint calls artists "style".
CATEGORIES = ("general", "character", "copyright", "artist", "meta", "rating")
_CHECKPOINT_CATEGORY: dict[str, str] = {"style": "artist"}

RATING_WORDS = {
    "anima": {"rating:g": "safe", "rating:s": "sensitive", "rating:q": "nsfw", "rating:e": "explicit"},
    "sdxl": {"rating:g": "general", "rating:s": "sensitive", "rating:q": "questionable", "rating:e": "explicit"},
}

RESOLUTION_TAGS = frozenset({"lowres", "highres", "absurdres", "incredibly_absurdres"})

# Meta tags that describe what the image looks like. The other ~110 meta tags the checkpoint knows are Danbooru
# bookkeeping (md5_mismatch, check_translation, tagme, protected_link, duplicate, ...), noise in a caption.
VISUAL_META_TAGS = frozenset({
    "official_art", "anime_screenshot", "game_screenshot", "jpeg_artifacts", "scan", "magazine_scan", "scan_artifacts",
    "crease", "aliasing", "traditional_media", "mixed_media", "unfinished", "concept_art", "key_visual", "promotional_art",
    "novel_illustration", "visual_novel_cg", "artbook", "official_wallpaper", "production_art", "game_asset", "eyecatch",
    "end_card", "cosplay_photo", "ai-generated", "ai-assisted", "colorized", "upscaled", "waifu2x", "adversarial_noise",
})  # fmt: skip
# *_(medium) tags pass the filter (watercolor, graphite, oil painting, blender, ...) except these: which digital
# painting program was used says little about what the picture looks like, and photoshop_(medium) alone is predicted
# for a large share of all images.
SOFTWARE_MEDIUM_TAGS = frozenset({
    "photoshop_(medium)", "painttool_sai_(medium)", "clip_studio_paint_(medium)", "procreate_(medium)", "paint.net_(medium)",
    "gimp_(medium)", "ibispaint_(medium)", "krita_(medium)", "paint_chat_(medium)",
})  # fmt: skip
META_MODES = ("filtered", "all", "none")
RESOLUTION_MODES = ("size", "predicted", "none")

_COUNT_TAG = re.compile(r"^\d+\+?(girl|boy|other)s?$")


def normalize(tag: str) -> str:
    """User spelling -> Danbooru name: ``"Simple Background "`` -> ``simple_background``."""
    return tag.strip().lower().replace(" ", "_")


def prompt_form(tag: str) -> str:
    """Danbooru name -> prompt spelling. Short tags keep their underscores: they are emoticons (``^_^``, ``o_o``)."""
    return tag.replace("_", " ") if len(tag) > 3 else tag


def _visual_meta(tag: str) -> bool:
    return tag in VISUAL_META_TAGS or tag in RESOLUTION_TAGS or (tag.endswith("_(medium)") and tag not in SOFTWARE_MEDIUM_TAGS)


def split_tag_list(text: str | None) -> list[str]:
    return [t.strip() for t in (text or "").split(",") if t.strip()]


def parse_replacements(text: str | None) -> dict[str, str]:
    """``"src1,dst1;src2,dst2"`` (``\\,`` and ``\\;`` escape literal commas/semicolons) -> {src: dst}, Danbooru names."""
    if not text:
        return {}
    out: dict[str, str] = {}
    escaped = text.replace("\\,", "\0c").replace("\\;", "\0s")
    for pair in escaped.split(";"):
        if not pair.strip():
            continue
        parts = pair.split(",")
        if len(parts) != 2:
            raise ValueError(f"Tag replacement must be 'source,target': {pair.replace(chr(0) + 'c', ',').replace(chr(0) + 's', ';')!r}")
        src, dst = (normalize(p.replace("\0c", ",").replace("\0s", ";")) for p in parts)
        out[src] = dst
    return out


def resolution_tags(width: int, height: int) -> list[str]:
    """Danbooru's automatic resolution tags for an image of this size."""
    tags = []
    if width <= 500 and height <= 500:
        tags.append("lowres")
    if width >= 1600 or height >= 1200:
        tags.append("highres")
    if width >= 3200 or height >= 2400:
        tags.append("absurdres")
    if width >= 10000 or height >= 10000:
        tags.append("incredibly_absurdres")
    return tags


def expand_character(tag: str) -> list[str]:
    """``name_(series)`` -> ``[name, series]``; ``name_(costume)_(series)`` -> ``[name_(costume), series]``."""
    if not tag.endswith(")") or "(" not in tag:
        return [tag]
    head, _, series = tag[:-1].rpartition("(")
    head = head.rstrip("_")
    return [head, series] if head and series else [tag]


@dataclass(frozen=True, slots=True)
class Vocabulary:
    tags: Sequence[str]
    slices: dict[str, slice]  # our category name -> its range of output indices
    thresholds: dict[str, float]  # the checkpoint's per-category F1-optimal thresholds

    @classmethod
    def from_config(cls, config: dict) -> Vocabulary:
        slices: dict[str, slice] = {}
        start = 0
        for name, count in config["tags_split"]:
            name = str(name)
            slices[_CHECKPOINT_CATEGORY.get(name, name)] = slice(start, start + count)
            start += count
        missing = set(CATEGORIES) - set(slices)
        if missing:
            raise ValueError(f"Checkpoint lacks tag categories {sorted(missing)}")
        best = config.get("category_best_threshold") or {}
        thresholds = {_CHECKPOINT_CATEGORY.get(str(k), str(k)): float(v) for k, v in best.items()}
        return cls(config["tags"], slices, thresholds)


@dataclass(slots=True)
class CaptionOptions:
    style: str = "anima"
    # Per-category minimum confidence; a value above 1 disables the category. The rating is always the argmax.
    thresholds: dict[str, float] = field(default_factory=dict)
    rating: bool = True
    meta: str = "filtered"
    resolution: str = "size"
    prefix: list[str] = field(default_factory=list)  # emitted verbatim, first
    always_first: list[str] = field(default_factory=list)  # sdxl: each moved to the front when present, in turn
    override_artists: list[str] | None = None  # Danbooru names: these artists instead of any prediction
    undesired: frozenset[str] = frozenset()  # Danbooru names
    replacements: dict[str, str] = field(default_factory=dict)  # Danbooru name -> Danbooru name
    separator: str = ", "

    def __post_init__(self) -> None:
        if self.style not in STYLES:
            raise ValueError(f"Unknown style {self.style!r}; expected one of {STYLES}")
        if self.meta not in META_MODES:
            raise ValueError(f"Unknown meta mode {self.meta!r}; expected one of {META_MODES}")
        if self.resolution not in RESOLUTION_MODES:
            raise ValueError(f"Unknown resolution mode {self.resolution!r}; expected one of {RESOLUTION_MODES}")


Scored = dict[str, list[tuple[str, float]]]


@dataclass(slots=True)
class Caption:
    entries: list[str]  # final caption tags in order
    scored: Scored  # what passed the thresholds and filters, per category, Danbooru names, for --debug
    separator: str

    @property
    def text(self) -> str:
        return self.separator.join(self.entries)


class Captioner:
    def __init__(self, vocab: Vocabulary, options: CaptionOptions):
        self.vocab = vocab
        self.options = options
        self.thresholds = {c: options.thresholds.get(c, vocab.thresholds.get(c, 0.35)) for c in CATEGORIES if c != "rating"}
        self._order = {t: i for i, t in enumerate(vocab.tags)}

    def score(self, probs: np.ndarray, width: int, height: int) -> Scored:
        """Tags above their category threshold, highest confidence first, after filters and replacements."""
        o = self.options
        tags = self.vocab.tags
        out: Scored = {c: [] for c in CATEGORIES}
        for cat, th in self.thresholds.items():
            if th > 1:
                continue
            p = probs[self.vocab.slices[cat]]
            idx = np.flatnonzero(p >= th)
            start = self.vocab.slices[cat].start
            out[cat] = [(tags[start + int(i)], float(p[i])) for i in idx[np.argsort(-p[idx], kind="stable")]]
        if o.meta == "none":
            out["meta"] = []
        elif o.meta == "filtered":
            out["meta"] = [(t, p) for t, p in out["meta"] if _visual_meta(t)]
        if o.resolution != "predicted":
            out["meta"] = [(t, p) for t, p in out["meta"] if t not in RESOLUTION_TAGS]
            if o.resolution == "size" and o.meta != "none":
                out["meta"] = [(t, 1.0) for t in resolution_tags(width, height)] + out["meta"]
        if o.override_artists is not None:
            out["artist"] = [(a, 1.0) for a in o.override_artists]
        if o.rating:
            sl = self.vocab.slices["rating"]
            i = int(np.argmax(probs[sl]))
            out["rating"] = [(tags[sl.start + i], float(probs[sl][i]))]
        for cat, items in out.items():
            kept = []
            for t, p in items:
                if t in o.undesired:
                    continue
                t = o.replacements.get(t, t)
                if t and t not in o.undesired:
                    kept.append((t, p))
            out[cat] = kept
        return out

    def caption(self, probs: np.ndarray, width: int, height: int) -> Caption:
        scored = self.score(probs, width, height)
        build = self._anima if self.options.style == "anima" else self._sdxl
        entries = [e for e in _dedupe(self.options.prefix + build(scored)) if normalize(e) not in self.options.undesired]
        return Caption(entries, scored, self.options.separator)

    def _rating(self, scored: Scored) -> list[str]:
        words = RATING_WORDS[self.options.style]
        return [words.get(t, prompt_form(t)) for t, _ in scored["rating"]]

    def _anima(self, s: Scored) -> list[str]:
        general = [t for t, _ in s["general"]]
        counts = [t for t in general if _COUNT_TAG.match(t)]
        rest = [t for t in general if not _COUNT_TAG.match(t)]
        return (
            [prompt_form(t) for t, _ in s["meta"]]
            + self._rating(s)
            + [prompt_form(t) for t in counts]
            + [prompt_form(t) for t, _ in s["character"]]
            + [prompt_form(t) for t, _ in s["copyright"]]
            + ["@" + prompt_form(t) for t, _ in s["artist"]]
            + [prompt_form(t) for t in rest]
        )

    def _sdxl(self, s: Scored) -> list[str]:
        last = len(self._order)  # a replaced name is not in the vocabulary: it goes after the others
        general = sorted((t for t, _ in s["general"]), key=lambda t: self._order.get(t, last))
        entries = (
            # one entry per part, so a series shared by two characters or also predicted as a copyright appears once
            [prompt_form(p) for t, _ in s["character"] for p in expand_character(t)]
            + [prompt_form(t) for t, _ in s["copyright"]]
            + [prompt_form(t) for t, _ in s["artist"]]
            + [prompt_form(t) for t in general]
            + [prompt_form(t) for t, _ in s["meta"]]
            + self._rating(s)
        )
        for tag in (prompt_form(normalize(t)) for t in self.options.always_first):
            if tag in entries:
                entries.remove(tag)
                entries.insert(0, tag)
        return entries


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
