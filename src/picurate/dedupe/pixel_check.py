"""Pixel check: do two images differ only by lossy compression?

Embedding distance cannot tell a re-encoded copy from an edited one (a signature, an artist revision, the next page of
a pixiv set that changes one expression). Their difference images can: compression noise is high-frequency and spread
over the whole picture, an edit is concentrated in one place. So both images are reduced to the smaller one's size
(capped at :data:`WORK_EDGE` on the long side), low-passed, and compared window by window:

* ``peak``: the largest 16x16 window mean of the per-pixel max-over-channels absolute difference,
* ``mean``: the mean of that difference over the whole picture.

A pair is *compression only* when the peak is tiny, or when the peak is small, the mean is small (a global colour
change fails here) and the peak is not far above the mean (a local edit on an otherwise identical copy fails here).

Measured on the author's library (6158 review pairs, 200 sampled images with synthetic variants): JPEG q60-q90, WebP
80, double compression and a halved re-encode pass 94.5-100%; 2%-high text and a 6-level colour shift pass 0%,
1%-high text and 1.5-3% patches 1-2%; none of 90 pairs a reviewer skipped as "not sure" passed.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from collections import OrderedDict
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import PIL.Image
import PIL.ImageFile
import PIL.ImageFilter

logger = logging.getLogger(__name__)

# Same leniency as the encoder.
PIL.Image.MAX_IMAGE_PIXELS = None
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

WORK_EDGE = 1024  # long edge of the comparison, at most
BLUR_SIGMA = 1.5  # removes 8x8 block edges and ringing at that scale
WINDOW = 16  # window side for the local means, stride half of it
ASPECT_TOLERANCE = 0.01  # relative; beyond it the pair is a crop or a reframe, never compression only

IDENTICAL_PEAK = 0.5  # below this the two are the same picture whatever the mean
MAX_PEAK = 12.0
MAX_MEAN = 3.0
MAX_PEAK_TO_MEAN = 20.0  # compression noise peaks at <= 16x its mean, local edits at 50-900x

default_buffer_mib = 2048

SAME = "same"
DIFFERENT = "different"
ASPECT = "aspect"
ERROR = "error"


@dataclass(slots=True, frozen=True)
class PixelVerdict:
    status: str  # SAME, DIFFERENT, ASPECT (aspect ratios differ) or ERROR (unreadable)
    peak: float | None = None
    mean: float | None = None
    message: str = ""

    @property
    def same(self) -> bool:
        return self.status == SAME

    @classmethod
    def measured(cls, peak: float, mean: float) -> PixelVerdict:
        return cls(SAME if compression_only(peak, mean) else DIFFERENT, peak, mean)

    def describe(self) -> str:
        if self.status in (SAME, DIFFERENT):
            return f"{self.status} (peak {self.peak:.2f}, mean {self.mean:.2f})"
        return f"{self.status}{': ' + self.message if self.message else ''}"


def compression_only(peak: float, mean: float) -> bool:
    return peak <= IDENTICAL_PEAK or (peak <= MAX_PEAK and mean <= MAX_MEAN and peak <= MAX_PEAK_TO_MEAN * mean)


def load_rgb(path: str) -> PIL.Image.Image:
    """Decode to RGB; transparency is composited on mid-grey so hidden colours under alpha do not count."""
    with PIL.Image.open(path) as img:
        img.load()
        if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            background = PIL.Image.new("RGBA", rgba.size, (128, 128, 128, 255))
            return PIL.Image.alpha_composite(background, rgba).convert("RGB")
        return img.convert("RGB")


def _window_means(d: np.ndarray, w: int) -> np.ndarray:
    """Means of all ``w`` x ``w`` windows at stride ``w // 2``, from an integral image."""
    h, wd = d.shape
    if h < w or wd < w:
        return np.array([d.mean()])
    ii = np.zeros((h + 1, wd + 1), dtype=np.float64)
    ii[1:, 1:] = d.cumsum(0).cumsum(1)
    s = max(1, w // 2)
    ys = np.arange(0, h - w + 1, s)[:, None]
    xs = np.arange(0, wd - w + 1, s)[None, :]
    total = ii[ys + w, xs + w] - ii[ys, xs + w] - ii[ys + w, xs] + ii[ys, xs]
    return (total / (w * w)).ravel()


def compare_images(a: PIL.Image.Image, b: PIL.Image.Image) -> PixelVerdict:
    if abs((a.width / a.height) / (b.width / b.height) - 1) > ASPECT_TOLERANCE:
        return PixelVerdict(ASPECT, message=f"{a.width}x{a.height} vs {b.width}x{b.height}")
    small = a if a.width * a.height <= b.width * b.height else b
    scale = min(1.0, WORK_EDGE / max(small.size))
    size = (max(1, round(small.width * scale)), max(1, round(small.height * scale)))
    blur = PIL.ImageFilter.GaussianBlur(BLUR_SIGMA)
    pa = a.resize(size, PIL.Image.Resampling.BOX).filter(blur)
    pb = b.resize(size, PIL.Image.Resampling.BOX).filter(blur)
    d = np.abs(np.asarray(pa, dtype=np.int16) - np.asarray(pb, dtype=np.int16)).max(axis=2).astype(np.float32)
    return PixelVerdict.measured(float(_window_means(d, WINDOW).max()), float(d.mean()))


def compare_files(path_a: str, path_b: str) -> PixelVerdict:
    try:
        return compare_images(load_rgb(path_a), load_rgb(path_b))
    except Exception as e:  # noqa: BLE001 - an unreadable file is a verdict, not a crash
        return PixelVerdict(ERROR, message=str(e))


# -- many pairs ----------------------------------------------------------------------------------------------------


class _ImageCache:
    """Decoded images, least recently used dropped first once ``budget`` bytes are exceeded. The two images of the
    pair being compared are always held, even when they alone exceed the budget."""

    def __init__(self, budget: int):
        self.budget = budget
        self.used = 0
        self._d: OrderedDict[str, PIL.Image.Image | Exception] = OrderedDict()

    def get(self, path: str, pinned: str | None = None) -> PIL.Image.Image:
        hit = self._d.get(path)
        if hit is None:
            try:
                hit = load_rgb(path)
            except Exception as e:  # noqa: BLE001
                hit = e
            self._d[path] = hit
            self.used += self._size(hit)
            self._evict(keep={path, pinned})
        else:
            self._d.move_to_end(path)
        if isinstance(hit, Exception):
            raise hit
        return hit

    @staticmethod
    def _size(item: PIL.Image.Image | Exception) -> int:
        return 0 if isinstance(item, Exception) else item.width * item.height * 3

    def _evict(self, keep: set[str | None]) -> None:
        for path in list(self._d):
            if self.used <= self.budget:
                return
            if path not in keep:
                self.used -= self._size(self._d.pop(path))


def _check_component(paths: Sequence[str], pairs: Sequence[tuple[int, int, int]], budget: int) -> list[tuple[int, PixelVerdict]]:
    """Worker: compare ``pairs`` (index into the caller's list, a, b as indices into ``paths``), decoding each image
    once as long as the cache budget allows. Pairs arrive sorted so that one side repeats across consecutive pairs."""
    cache = _ImageCache(budget)
    out = []
    for k, a, b in pairs:
        try:
            ia = cache.get(paths[a])
            ib = cache.get(paths[b], pinned=paths[a])
            out.append((k, compare_images(ia, ib)))
        except Exception as e:  # noqa: BLE001
            out.append((k, PixelVerdict(ERROR, message=str(e))))
    return out


def check_pairs(
    pairs: Sequence[tuple[str, str]],
    *,
    workers: int | None = None,
    buffer_bytes: int = default_buffer_mib << 20,
    progress: Callable[[int], object] | None = None,
) -> list[PixelVerdict]:
    """Verdicts for ``pairs`` of absolute paths, in order.

    Pairs sharing images (the edges of one duplicate group) go to the same worker, which decodes each image once and
    keeps decoded images up to ``buffer_bytes / workers``; ``buffer_bytes`` bounds the decoded pixels held across all
    workers (plus the two images of each pair in progress).
    """
    if not pairs:
        return []
    ids: dict[str, int] = {}
    parent: list[int] = []

    def node(path: str) -> int:
        if path not in ids:
            ids[path] = len(parent)
            parent.append(len(parent))
        return ids[path]

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    indexed = []
    for k, (a, b) in enumerate(pairs):
        ia, ib = node(a), node(b)
        parent[find(ia)] = find(ib)
        indexed.append((k, ia, ib))
    components: dict[int, list[tuple[int, int, int]]] = {}
    for k, ia, ib in indexed:
        components.setdefault(find(ia), []).append((k, ia, ib))
    names = list(ids)

    workers = max(1, workers or os.cpu_count() or 1)
    budget = max(1, buffer_bytes // workers)
    tasks = []
    for edges in components.values():
        local: dict[int, int] = {}
        for _, a, b in edges:
            local.setdefault(a, len(local))
            local.setdefault(b, len(local))
        task_pairs = sorted(((k, local[a], local[b]) for k, a, b in edges), key=lambda t: (min(t[1], t[2]), max(t[1], t[2])))
        task_paths = [names[g] for g in sorted(local, key=local.__getitem__)]
        tasks.append((task_paths, task_pairs))
    tasks.sort(key=lambda t: -len(t[1]))  # big components first so they do not finish last alone

    results: list[PixelVerdict | None] = [None] * len(pairs)
    if workers == 1 or len(tasks) == 1:
        for task_paths, task_pairs in tasks:
            for k, verdict in _check_component(task_paths, task_pairs, buffer_bytes):
                results[k] = verdict
            if progress:
                progress(len(task_pairs))
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks)), mp_context=mp.get_context("spawn")) as pool:
            futures = {pool.submit(_check_component, p, q, budget): len(q) for p, q in tasks}
            for future in as_completed(futures):
                for k, verdict in future.result():
                    results[k] = verdict
                if progress:
                    progress(futures[future])
    assert all(r is not None for r in results)
    return results  # type: ignore[return-value]
