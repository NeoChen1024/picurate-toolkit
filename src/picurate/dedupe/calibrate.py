"""Threshold calibration from the data itself.

Two distributions are measured on a reproducible sample of the database:

* **variant distances**: each sampled image is re-encoded on the fly (JPEG re-save, downscale, ...) and the distance
  between the variant's embedding and the stored one is recorded. This is what a lossy copy of a picture looks like
  to the deduper.
* **nearest different image**: the distance from each sampled image to its closest other entry in the database.
  This is the floor a threshold must stay under, or unrelated pictures start merging.

Both are printed as text histograms on a shared log axis, and a threshold is suggested when the two do not overlap.
The default ``--threshold`` of 0.1 deliberately sits far below both: it only catches bit-identical re-encodes. The
suggestion here is for a looser setting that also merges the chosen variants.
"""

from __future__ import annotations

import io
import logging
import math
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import PIL.Image

from ..common.images import ImageRecord, Roots
from .encoder import CLIPImageEncoder, Preprocessed
from .similarity import DistanceIndex

logger = logging.getLogger(__name__)

Progress = Callable[[int], object] | None


def _recompress(img: PIL.Image.Image, fmt: str, **kwargs) -> PIL.Image.Image:
    buf = io.BytesIO()
    img.save(buf, fmt, **kwargs)
    out = PIL.Image.open(io.BytesIO(buf.getvalue()))
    out.load()
    return out.convert("RGB")


def _scale(img: PIL.Image.Image, factor: float) -> PIL.Image.Image:
    w, h = img.size
    return img.resize((max(1, round(w * factor)), max(1, round(h * factor))), PIL.Image.Resampling.LANCZOS)


VARIANTS: dict[str, Callable[[PIL.Image.Image], PIL.Image.Image]] = {
    "jpeg90": lambda im: _recompress(im, "JPEG", quality=90),
    "jpeg75": lambda im: _recompress(im, "JPEG", quality=75),
    "webp80": lambda im: _recompress(im, "WEBP", quality=80),
    "half": lambda im: _scale(im, 0.5),
    "quarter": lambda im: _scale(im, 0.25),
}
default_variants = ("jpeg90", "half", "quarter")


def sample_rows(records: Sequence[ImageRecord], samples: int, seed: int) -> list[int]:
    """Row indices of a reproducible sample: for the same set of paths and seed, the same images are picked
    regardless of database row order."""
    order = sorted(range(len(records)), key=lambda i: records[i].path)
    rng = random.Random(seed)
    return sorted(rng.sample(order, min(samples, len(order))))


def nearest_other_distances(index: DistanceIndex, rows: Sequence[int]) -> np.ndarray:
    """For each row in ``rows``, the L2 distance to the closest *other* row of ``index``."""
    out = np.empty(len(rows), dtype=np.float32)
    for k, row in enumerate(rows):
        d = index.distances(row, row + 1)[0]
        d[row] = math.inf
        out[k] = float(d.min())
    return out


@dataclass
class Calibration:
    series: dict[str, np.ndarray] = field(default_factory=dict)  # name -> distances
    skipped: list[str] = field(default_factory=list)

    @property
    def variant_names(self) -> list[str]:
        return [k for k in self.series if k != "nearest-other"]

    def variant_distances(self) -> np.ndarray:
        parts = [self.series[k] for k in self.variant_names]
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)


def calibrate(
    encoder: CLIPImageEncoder,
    roots: Roots,
    records: Sequence[ImageRecord],
    embeddings: np.ndarray,
    index: DistanceIndex,
    *,
    samples: int,
    seed: int,
    variants: Sequence[str] = default_variants,
    batch_size: int = 4,
    progress: Progress = None,
) -> Calibration:
    """Measure variant and nearest-other distance distributions on a sample of ``records``.

    ``embeddings`` are the stored rows (fp16 or fp32) matching ``records``; ``index`` holds them on the device.
    """
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants {unknown}; available: {', '.join(VARIANTS)}")
    rows = sample_rows(records, samples, seed)
    result = Calibration()
    result.series["nearest-other"] = nearest_other_distances(index, rows)

    per_variant: dict[str, list[float]] = {v: [] for v in variants}
    pending: list[tuple[str, int, Preprocessed]] = []  # (variant, row, preprocessed)

    def flush() -> None:
        if not pending:
            return
        out = encoder.encode_images([p for _, _, p in pending])
        for (variant, row, _), emb in zip(pending, out):
            per_variant[variant].append(float(np.linalg.norm(emb - embeddings[row].astype(np.float32))))
        pending.clear()

    for row in rows:
        path = records[row].file(roots)
        try:
            with PIL.Image.open(path) as img:
                base = img.convert("RGB")
        except Exception as e:  # noqa: BLE001 - a broken file only drops one sample
            logger.warning("Skipping %s: %s", path, e)
            result.skipped.append(records[row].path)
        else:
            for v in variants:
                pending.append((v, row, encoder.preprocess(VARIANTS[v](base))))
                if len(pending) >= batch_size:
                    flush()
        if progress is not None:
            progress(1)
    flush()
    for v in variants:
        result.series[v] = np.asarray(per_variant[v], dtype=np.float32)
    return result


# -- reporting ---------------------------------------------------------------------------------------------------------


def log_edges(values: Iterable[np.ndarray], per_decade: int = 8) -> np.ndarray:
    """Log-spaced bin edges covering every positive value of all series, whole decades, ``per_decade`` bins each."""
    positives = np.concatenate([v[v > 0] for v in values]) if values else np.empty(0)
    if positives.size == 0:
        return np.array([0.1, 1.0])
    lo = math.floor(math.log10(positives.min()))
    hi = math.ceil(math.log10(positives.max()))
    hi = max(hi, lo + 1)
    return np.logspace(lo, hi, (hi - lo) * per_decade + 1)


def render_histogram(name: str, values: np.ndarray, edges: np.ndarray, width: int = 40) -> list[str]:
    """Text histogram: one line per non-empty bin range, bar length proportional to the count."""
    lines = []
    if values.size == 0:
        return [f"{name}: no samples"]
    p = np.percentile(values, [50, 95, 99])
    lines.append(
        f"{name} ({values.size}): min {values.min():.3g}  p50 {p[0]:.3g}  p95 {p[1]:.3g}  p99 {p[2]:.3g}  max {values.max():.3g}"
    )
    zeros = int((values <= 0).sum())
    counts, _ = np.histogram(values[values > 0], bins=edges)
    peak = max(counts.max() if counts.size else 0, zeros, 1)
    if zeros:
        lines.append(f"  {'0':>7} {'':<9} |{'#' * round(width * zeros / peak)} {zeros}")
    nz = np.flatnonzero(counts)
    if nz.size:
        for b in range(nz[0], nz[-1] + 1):
            bar = "#" * round(width * counts[b] / peak)
            lines.append(f"  {edges[b]:>7.3g} -{edges[b + 1]:>7.3g} |{bar} {counts[b]}")
    return lines


def _fmt(x: float) -> str:
    return f"{x:.3g}"


def report(result: Calibration, thresholds: Sequence[float] = (0.1,), width: int = 40) -> list[str]:
    """Histograms of every series on a shared axis, then what each threshold would do and a suggestion."""
    lines: list[str] = []
    edges = log_edges(list(result.series.values()))
    for name, values in result.series.items():
        lines.extend(render_histogram(name, values, edges, width))
        lines.append("")
    if result.skipped:
        lines.append(f"{len(result.skipped)} sampled images could not be read and were skipped.")
        lines.append("")

    variants = result.variant_distances()
    background = result.series.get("nearest-other", np.empty(0))
    if variants.size == 0 or background.size == 0:
        lines.append("Not enough data for a suggestion.")
        return lines

    for t in thresholds:
        caught = float((variants <= t).mean()) * 100
        merged = int((background <= t).sum())
        lines.append(
            f"At threshold {_fmt(t)}: {caught:.1f}% of variants caught, {merged} of {background.size} sampled images would merge with a different image."
        )
    need = float(variants.max())
    floor = float(background.min())
    if need < floor:
        # Round up to two significant digits so the printed number still catches every sampled variant.
        mag = 10 ** (math.floor(math.log10(need)) - 1)
        suggested = math.ceil(need / mag) * mag
        lines.append(
            f"Suggested threshold for {', '.join(result.variant_names)}: {_fmt(suggested)} "
            f"(largest variant distance {_fmt(need)}, nearest different image {_fmt(floor)}, gap {floor / need:.1f}x)."
        )
        if floor / need < 2:
            lines.append("The margin is thin: a larger sample may close it. Consider dropping the loosest variant.")
    else:
        lines.append(
            f"No clean threshold: the largest variant distance ({_fmt(need)}) is not below the nearest different image "
            f"({_fmt(floor)}). Drop the loosest variant or accept missing some copies."
        )
    return lines
