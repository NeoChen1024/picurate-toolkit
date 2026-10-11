"""Runs the tagger on batches of preprocessed images, asynchronously, for ``encoding_pipeline.run_pipeline``."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from .model import TaggerPreprocess, default_repo_id, load_model, model_identity

logger = logging.getLogger(__name__)

PRECISIONS = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


@dataclass(slots=True)
class PendingProbs:
    out: torch.Tensor
    n: int
    host: torch.Tensor  # pinned staging buffer, kept alive until the copy has run


class Tagger:
    """Loads the tagger once and turns batches of ``TaggerPreprocess`` arrays into ``(N, num_tags)`` probabilities.

    ``dtype`` defaults to fp16 on CUDA (probabilities within ~1e-3 of fp32) and fp32 elsewhere.

    ``compile=True`` compiles for one static input shape, and every batch, the last one included, is padded to
    ``batch_size`` so it never recompiles. Measured on an RTX 4080 SUPER, fp16, batch 4: 9.7 img/s eager, 14.4 img/s
    compiled static, 12.7 img/s with ``dynamic=True``; max-autotune adds nothing for a 90 s warm-up.
    """

    def __init__(
        self,
        model: str = default_repo_id,
        *,
        revision: str | None = None,
        device: str = "cpu",
        dtype: str | None = None,
        compile: bool = False,
        batch_size: int = 4,
    ):
        if dtype is None:
            dtype = "fp16" if device.startswith("cuda") else "fp32"
        try:
            self.tdtype = PRECISIONS[dtype]
        except KeyError:
            raise ValueError(f"Unknown dtype {dtype!r}; expected one of {tuple(PRECISIONS)}") from None
        self.device = device
        self.dtype_name = dtype
        self.model_id = model_identity(model, revision)
        self.batch_size = batch_size
        self.compiled = compile
        logger.info("Loading tagger %s on %s (%s%s)", model, device, dtype, ", compiled" if compile else "")
        loaded = load_model(model, revision=revision, device=device, dtype=self.tdtype)
        self.model = loaded.model
        self.config = loaded.config
        self.image_size = self.model.cfg.img_size
        self._forward = torch.compile(self.model, dynamic=False) if compile else self.model
        # numpy has no bf16: the workers emit fp32 then and the cast happens on the device.
        self.array_dtype = np.dtype(np.float16) if self.tdtype == torch.float16 else np.dtype(np.float32)

    def get_preprocessor(self) -> TaggerPreprocess:
        return TaggerPreprocess(self.image_size, self.array_dtype)

    @torch.inference_mode()
    def submit(self, preprocessed: Sequence[np.ndarray]) -> PendingProbs:
        """Queue one forward pass and return without waiting for it; ``collect`` gets the result."""
        n = len(preprocessed)
        if n == 0:
            raise ValueError("Empty batch")
        rows = max(n, self.batch_size) if self.compiled else n
        pin = self.device.startswith("cuda")
        host = torch.zeros((rows, *preprocessed[0].shape), dtype=torch.from_numpy(preprocessed[0][:0]).dtype, pin_memory=pin)
        for i, a in enumerate(preprocessed):
            host[i].copy_(torch.from_numpy(np.ascontiguousarray(a)))
        x = host.to(self.device, non_blocking=pin).to(self.tdtype)
        return PendingProbs(self._forward(x)[:n].float().sigmoid(), n, host)

    def collect(self, pending: PendingProbs) -> np.ndarray:
        return pending.out.cpu().numpy()

    def close(self) -> None:
        del self._forward
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
