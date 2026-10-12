"""timm image tower wrapper used to turn preprocessed images into embeddings.

Only the image tower of a CLIP/SigLIP model is needed for deduplication, so the model is loaded straight from timm
(`timm.create_model`), which is also what open_clip uses under the hood for these architectures. Embeddings are
bit-identical to open_clip's ``encode_image`` for the same checkpoint, at 40% of the download size.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import PIL.Image
import timm
import timm.data
import torch
from timm.models.naflexvit import NaFlexVit

logger = logging.getLogger(__name__)

# SigLIP2 so400m NaFlex: the image keeps its aspect ratio and is resized to at most ``seq_len`` 16x16 patches (1024 =
# the pixel budget of the fixed 512px model, same speed and VRAM). Measured on 24 library images (norm ~10.8): a JPEG
# q90 re-save lands within 0.3-0.5 of the original, a 50% downscale within 0.4, a 25% downscale within 2.4, while
# different images sit at 5.6+. The fixed-size vit_so400m_patch16_siglip_512 (norm ~16) was wider on every variant
# relative to its inter-image spread (50% downscale up to 1.4, 25% up to 4.3, different images 8.7+). PE-Core-bigG
# (vit_pe_core_bigG_14_448.fb) was tried first but is so sensitive to compression artifacts that a JPEG q90 re-save
# lands 5-15 away.
default_model_id = "naflexvit_so400m_patch16_siglip.v2_webli"
default_seq_len = 1024

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


class _FixedPreprocess:
    """PIL (any mode) -> ``(C, H, W)`` array in ``dtype``, fully preprocessed for a fixed-resolution tower. Picklable for workers."""

    def __init__(self, transform: Callable, dtype: np.dtype):
        self.transform = transform
        self.dtype = dtype

    def __call__(self, img: PIL.Image.Image) -> np.ndarray:
        return self.transform(img.convert("RGB")).numpy().astype(self.dtype, copy=False)


class _NaFlexPreprocess:
    """PIL -> ``(patches, coords)``: the image resized (aspect ratio kept) to at most ``seq_len`` patches, normalized
    and patchified to ``(N, P*P*C)`` in ``dtype`` plus ``(N, 2)`` int32 (y, x) patch coordinates. Picklable."""

    def __init__(self, transform: Callable, dtype: np.dtype):
        self.transform = transform
        self.dtype = dtype

    def __call__(self, img: PIL.Image.Image) -> tuple[np.ndarray, np.ndarray]:
        d = self.transform(img.convert("RGB"))
        return d["patches"].numpy().astype(self.dtype, copy=False), d["patch_coord"].numpy().astype(np.int32, copy=False)


Preprocessed = np.ndarray | tuple[np.ndarray, np.ndarray]


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
    """Loads a timm image tower once and encodes batches of preprocessed images.

    ``dtype`` defaults to fp16 on CUDA and fp32 elsewhere. fp16 inference is what makes the stored fp16 embeddings
    lossless: every value the model emits is already an fp16 number.

    NaFlex towers (timm ``naflexvit_*``) take the image at its own aspect ratio as a sequence of at most ``seq_len``
    patches; fixed-resolution towers get the whole image squashed to their input size. Either way the workers deliver
    fully preprocessed arrays and ``submit`` only pads, pins, copies and runs.

    ``compile=True`` wraps the forward in ``torch.compile(dynamic=True)``, so batches of any size share one graph
    (20-30 s warm-up). Fixed-resolution towers gain about 17% on an RTX 4080; NaFlex towers gain nothing today
    because timm's NaFlex position-embedding path breaks the graph (measured: 62 vs 63 img/s eager).
    """

    def __init__(
        self,
        model_id: str = default_model_id,
        device: str = "cpu",
        dtype: str | None = None,
        *,
        compile: bool = False,
        seq_len: int = default_seq_len,
    ):
        if dtype is None:
            dtype = "fp16" if "cuda" in device else "fp32"
        try:
            self.tdtype = _PRECISIONS[dtype.lower()]
        except KeyError:
            raise ValueError(f"Unknown dtype {dtype!r}; expected one of {precision_choices}") from None
        self.device = device
        logger.info("Loading image model %s on %s (%s%s)", model_id, device, dtype, ", compiled" if compile else "")
        self.model = timm.create_model(model_id, pretrained=True, num_classes=0).to(device=device, dtype=self.tdtype).eval()
        self.naflex = isinstance(self.model, NaFlexVit)
        cfg = timm.data.resolve_data_config({}, model=self.model)
        if self.naflex:
            if seq_len < 1:
                raise ValueError("seq_len must be positive")
            self.seq_len = seq_len
            self.patch_size = tuple(self.model.embeds.patch_size)
            # Embeddings depend on the patch budget, so it is part of the identity a database is bound to.
            self.model_id = f"{model_id}@{seq_len}"
            transform = timm.data.create_transform(
                input_size=cfg["input_size"],
                interpolation=cfg["interpolation"],
                mean=cfg["mean"],
                std=cfg["std"],
                is_training=False,
                naflex=True,
                patch_size=self.patch_size,
                max_seq_len=seq_len,
                patchify=True,
            )
            self.preprocess: Callable[[PIL.Image.Image], Preprocessed] = _NaFlexPreprocess(transform, self.array_dtype)
        else:
            self.seq_len = None
            self.model_id = model_id
            # CLIP-style models are trained on the whole image squashed to the input size. timm's default eval
            # transform center-crops 90%, which would shift every embedding; override to the exact open_clip one.
            cfg.update(crop_pct=1.0, crop_mode="squash")
            self.preprocess = _FixedPreprocess(timm.data.create_transform(**cfg, is_training=False), self.array_dtype)

        self._forward: Callable[..., torch.Tensor] = torch.compile(self.model, dynamic=True) if compile else self.model

    def get_preprocessor(self) -> Callable[[PIL.Image.Image], Preprocessed]:
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
        if self.naflex:
            first = preprocessed[0]
            assert isinstance(first, tuple), "NaFlex tower expects (patches, coords) pairs"
            patch_dim = first[0].shape[1]
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
        host = torch.empty((n, *preprocessed[0].shape), dtype=self.tdtype, pin_memory=pin)
        for i, a in enumerate(preprocessed):
            host[i].copy_(torch.from_numpy(np.ascontiguousarray(a)))
        batch = host.to(self.device, non_blocking=pin)
        return PendingBatch(self._forward(batch), n, (host,))

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
