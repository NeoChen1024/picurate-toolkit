"""The image tower that turns preprocessed images into embeddings.

The tower is ``picurate.common.siglip_image`` (SigLIP2 so400m NaFlex), loaded from the base checkpoint or from a
``clip-train export`` file. Images keep their aspect ratio and are resized to at most ``seq_len`` 16x16 patches, so
nothing is squashed or cropped; ``seq_len`` is part of the model identity a database is bound to.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import PIL.Image
import torch

from ..common.siglip_image import DEFAULT_SEQ_LEN, IMAGE_TOWER, ImageTower, load_image_tower, to_patches

logger = logging.getLogger(__name__)

# SigLIP2 so400m NaFlex: the image keeps its aspect ratio and is resized to at most ``seq_len`` 16x16 patches (1024 =
# the pixel budget of the fixed 512px model, same speed and VRAM). Measured on 24 library images (norm ~10.8): a JPEG
# q90 re-save lands within 0.3-0.5 of the original, a 50% downscale within 0.4, a 25% downscale within 2.4, while
# different images sit at 5.6+. The fixed-size vit_so400m_patch16_siglip_512 (norm ~16) was wider on every variant
# relative to its inter-image spread (50% downscale up to 1.4, 25% up to 4.3, different images 8.7+). PE-Core-bigG
# (vit_pe_core_bigG_14_448.fb) was tried first but is so sensitive to compression artifacts that a JPEG q90 re-save
# lands 5-15 away.
default_model_id = IMAGE_TOWER
default_seq_len = DEFAULT_SEQ_LEN

# Accepted spellings for the precision option -> torch dtype
_PRECISIONS: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}
precision_choices = tuple(_PRECISIONS)

Preprocessed = tuple[np.ndarray, np.ndarray]


class _NaFlexPreprocess:
    """PIL -> ``(patches, coords)``: the image resized (aspect ratio kept, bicubic) to at most ``seq_len`` patches,
    normalized to [-1, 1] in FP32 as SigLIP was trained (then cast to ``dtype``) and patchified to ``(N, 16*16*3)``,
    plus ``(N, 2)`` int32 (row, column) patch coordinates. Picklable for worker processes."""

    def __init__(self, seq_len: int, dtype: np.dtype):
        self.seq_len = seq_len
        self.dtype = dtype

    def __call__(self, img: PIL.Image.Image) -> Preprocessed:
        patches, coords = to_patches(img.convert("RGB"), self.seq_len)
        normalized = (patches.astype(np.float32) / 255.0 - 0.5) / 0.5
        return normalized.astype(self.dtype, copy=False), coords.astype(np.int32, copy=False)


@dataclass(slots=True)
class PendingBatch:
    """A forward pass that has been queued on the device but not synchronized yet (see ``CLIPImageEncoder.submit``).

    ``host`` holds the pinned staging buffers; they are kept alive here because the asynchronous copy may still be
    reading from them.
    """

    out: torch.Tensor
    n: int
    host: tuple[torch.Tensor, ...]


class CLIPImageEncoder:
    """Loads the image tower once and encodes batches of preprocessed images.

    ``model_id`` is the base tower's name (``default_model_id``) or the path of a ``clip-train export`` file, whose
    ``name`` then identifies the model. ``dtype`` defaults to fp16 on CUDA and fp32 elsewhere. fp16 inference is what
    makes the stored fp16 embeddings lossless: every value the model emits is already an fp16 number.

    The workers deliver fully preprocessed arrays and ``submit`` only pads, pins, copies and runs. ``compile=True``
    wraps the forward in ``torch.compile(dynamic=True)``, so batches of any size share one graph (20-30 s warm-up).
    """

    def __init__(
        self,
        model_id: str = default_model_id,
        device: str = "cpu",
        dtype: str | None = None,
        *,
        compile: bool = False,
        seq_len: int = default_seq_len,
        attention: str | None = None,
    ):
        if dtype is None:
            dtype = "fp16" if "cuda" in device else "fp32"
        try:
            self.tdtype = _PRECISIONS[dtype.lower()]
        except KeyError:
            raise ValueError(f"Unknown dtype {dtype!r}; expected one of {precision_choices}") from None
        if seq_len < 1:
            raise ValueError("seq_len must be positive")
        self.device = device
        self.seq_len = seq_len
        logger.info("Loading image model %s on %s (%s%s)", model_id, device, dtype, ", compiled" if compile else "")
        if model_id == default_model_id:
            name = model_id
            self.model = load_image_tower(device=device, dtype=self.tdtype, attention=attention)
        elif os.path.isfile(model_id):
            name, trained_seq_len, self.model = _load_export(model_id, attention)
            if trained_seq_len and trained_seq_len != seq_len:
                logger.warning("%s was trained with a patch budget of %d, not %d", model_id, trained_seq_len, seq_len)
            self.model = self.model.to(device=device, dtype=self.tdtype).eval()
        else:
            raise ValueError(f"Unknown model {model_id!r}: expected {default_model_id!r} or the path of a clip-train export")
        # Embeddings depend on the patch budget, so it is part of the identity a database is bound to.
        self.model_id = f"{name}@{seq_len}"
        self.preprocess = _NaFlexPreprocess(seq_len, self.array_dtype)
        self._forward = torch.compile(self.model, dynamic=True) if compile else self.model

    def get_preprocessor(self) -> _NaFlexPreprocess:
        """The full PIL -> numpy preprocessing, picklable so worker processes can run it. Its output is what
        ``submit`` takes."""
        return self.preprocess

    @property
    def array_dtype(self) -> np.dtype:
        """numpy dtype the preprocessor emits: the model's own dtype so the host->device copy is as small as
        possible, except bf16 which numpy lacks (fp32 then, cast on the device)."""
        return np.dtype(np.float16) if self.tdtype == torch.float16 else np.dtype(np.float32)

    @torch.no_grad()
    def submit(self, preprocessed: Sequence[Preprocessed]) -> PendingBatch:
        """Queue one forward pass on the device and return without waiting for it.

        ``preprocessed`` items are ``get_preprocessor`` outputs. They are gathered into pinned host buffers, copied
        asynchronously and run; call ``collect`` for the result. On CUDA this lets the caller go back to feeding the
        decoders while the GPU works.
        """
        n = len(preprocessed)
        if n == 0:
            raise ValueError("Empty batch")
        pin = self.device.startswith("cuda")
        patch_dim = preprocessed[0][0].shape[1]
        patches = torch.zeros((n, self.seq_len, patch_dim), dtype=self.tdtype, pin_memory=pin)
        coords = torch.zeros((n, self.seq_len, 2), dtype=torch.int64, pin_memory=pin)
        valid = torch.zeros((n, self.seq_len), dtype=torch.bool, pin_memory=pin)
        for i, (p, c) in enumerate(preprocessed):
            k = p.shape[0]
            patches[i, :k].copy_(torch.from_numpy(np.ascontiguousarray(p)))
            coords[i, :k].copy_(torch.from_numpy(np.ascontiguousarray(c)))
            valid[i, :k] = True
        dev = {
            "patches": patches.to(self.device, non_blocking=pin),
            "patch_coord": coords.to(self.device, non_blocking=pin),
            "patch_valid": valid.to(self.device, non_blocking=pin),
        }
        return PendingBatch(self._forward(dev), n, (patches, coords, valid))

    def collect(self, pending: PendingBatch) -> np.ndarray:
        """Wait for a submitted batch and return its ``(N, D)`` float32 embeddings."""
        return pending.out.float().cpu().numpy()

    def encode_images(self, preprocessed: Sequence[Preprocessed]) -> np.ndarray:
        """Encode already-preprocessed images synchronously. Returns an ``(N, D)`` float32 array."""
        return self.collect(self.submit(preprocessed))

    def encode_pil_images(self, images: Sequence[PIL.Image.Image]) -> np.ndarray:
        """Preprocess and encode PIL images in one batch (convenience for small inputs)."""
        if not images:
            return np.empty((0, 0), dtype=np.float32)
        return self.encode_images([self.preprocess(img.convert("RGB")) for img in images])

    def close(self) -> None:
        """Drop the model and free device memory."""
        del self._forward
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _load_export(path: str, attention: str | None) -> tuple[str, int, ImageTower]:
    """A ``clip-train export`` file: its ``name``, the patch budget it was trained with (0 if unknown) and the image
    tower with its ``image.*`` weights, on the CPU."""
    from safetensors import safe_open

    with safe_open(path, "pt") as export:
        metadata = export.metadata() or {}
        state = {k[len("image.") :]: export.get_tensor(k) for k in export.keys() if k.startswith("image.")}
    name = metadata.get("name")
    if not name or not state:
        raise ValueError(f"{path} is not a clip-train export (no name or no image.* tensors)")
    with torch.device("meta"):
        tower = ImageTower(attention=attention)
    tower.load_state_dict(state, strict=True, assign=True)
    return name, int(metadata.get("seq_len", 0)), tower
