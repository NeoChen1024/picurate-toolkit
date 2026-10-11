"""Images on disk: which files count as images, how a directory is listed, and the metadata kept per image."""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass

import PIL.Image
import PIL.ImageFile

# This is a tool for the user's own collection, not a server decoding untrusted uploads: behave like an image viewer.
# Pillow's defaults reject images above ~179 Mpixel ("decompression bomb") and files whose data ends early, both of
# which ordinary viewers display without complaint. These module globals apply to the process that imports this
# module; worker processes re-import it, so they inherit the same settings.
PIL.Image.MAX_IMAGE_PIXELS = None
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

# Extensions Pillow knows how to open. Anything else in the image dir (sidecar .txt, .json, ...) is ignored silently.
IMAGE_EXTENSIONS = frozenset(ext for ext, fmt in PIL.Image.registered_extensions().items() if fmt in PIL.Image.OPEN)


def walk_directory_relative(directory: str) -> Iterator[str]:
    """Yield file paths relative to ``directory``, depth first."""
    for root, _, files in os.walk(directory):
        for file in files:
            yield os.path.relpath(os.path.join(root, file), directory)


def is_image_path(relative_path: str) -> bool:
    return os.path.splitext(relative_path)[1].lower() in IMAGE_EXTENSIONS


@dataclass(slots=True, frozen=True)
class ImageRecord:
    """Metadata of one image as observed when it was read (and as the databases store it)."""

    path: str  # relative to the image directory
    mtime: float
    size: int
    width: int
    height: int
    format: str  # PIL format name: "JPEG", "PNG", "GIF", "WEBP", ...

    @property
    def pixels(self) -> int:
        return self.width * self.height
