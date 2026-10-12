"""Image loading for the review GUI. Disk reads never happen on the GUI thread; see :class:`picurate.common.qt.Job`."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import PIL.Image
import PIL.ImageFile
from PySide6.QtGui import QImage, QPixmap

from ...common.qt import Job, Signals, qimage

# Same leniency as the encoder: this is the user's own collection, behave like an image viewer.
PIL.Image.MAX_IMAGE_PIXELS = None
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_image(path: str) -> PIL.Image.Image:
    with PIL.Image.open(path) as img:
        return img.convert("RGB")  # animated: first frame


def thumbnail(path: str, size: tuple[int, int]) -> QImage:
    with PIL.Image.open(path) as img:
        img.draft("RGB", size)  # JPEG: decode at a reduced scale
        im = img.convert("RGB")
    im.thumbnail(size)
    return qimage(im)


def diff_images(a: PIL.Image.Image, b: PIL.Image.Image, gain: int = 4) -> tuple[QImage, QImage]:
    """``(overlay, plain)``: the absolute difference of ``a`` and ``b`` (``b`` resampled to ``a``'s size), amplified
    ``gain`` times, once drawn over ``a`` at half brightness so the differences can be located, once on black."""
    if b.size != a.size:
        b = b.resize(a.size, PIL.Image.Resampling.LANCZOS)
    arr_a = np.asarray(a, dtype=np.int16)
    diff = np.abs(arr_a - np.asarray(b, dtype=np.int16)) * gain
    plain = PIL.Image.fromarray(np.clip(diff, 0, 255).astype(np.uint8), "RGB")
    overlay = PIL.Image.fromarray(np.clip(arr_a // 2 + diff, 0, 255).astype(np.uint8), "RGB")
    return qimage(overlay), qimage(plain)


class LRU:
    """A small dict with least-recently-used eviction."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._d: OrderedDict = OrderedDict()

    def get(self, key):
        if key in self._d:
            self._d.move_to_end(key)
            return self._d[key]
        return None

    def put(self, key, value) -> None:
        self._d[key] = value
        self._d.move_to_end(key)
        while len(self._d) > self.capacity:
            self._d.popitem(last=False)

    def __contains__(self, key) -> bool:
        return key in self._d

    def clear(self) -> None:
        self._d.clear()


__all__ = ["Job", "Signals", "LRU", "load_image", "qimage", "thumbnail", "diff_images", "QPixmap"]
