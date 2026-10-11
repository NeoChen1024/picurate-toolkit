#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Triton kernel for exact pairwise Euclidean distance from fp16-stored embeddings.

Why a custom kernel instead of ``torch.cdist``:

* ``torch.cdist`` on large inputs uses the ``|a|^2 + |b|^2 - 2 a.b`` matmul form. With unnormalized CLIP embeddings
  (norm ~16) that subtracts two numbers around 500 to get a d^2 below 0.01, so even in fp32 the result carries ~1e-2
  of error near the duplicate threshold. Tensor-core fp16 inputs make it far worse (~1e-1). This kernel accumulates
  ``sum((a - b)^2)`` directly, so there is no cancellation and the error is ~1e-5.
* The database stays in fp16 on the GPU (half the VRAM) and is upcast to fp32 in registers only.

Layout: both operands are column-major ``(D, cols)`` views of a ``(D, N)`` contiguous fp16 matrix, so the per-dimension
loads along ``cols`` are contiguous. Speed is on par with fp32 ``torch.cdist`` (CUDA cores, no tensor cores).
"""

import torch
import triton
import triton.language as tl


# Fixed launch configuration, picked by autotuning on an RTX 4080 Super (D=1152, N=100k): 64x64 tiles, 2 warps,
# 4x unrolled D loop. Not using @triton.autotune on purpose: the CLI is a fresh process every run and autotuning
# ~40 configs on a 100k-row problem costs minutes, far more than the whole search.
_BM, _BN, _UNROLL, _NUM_WARPS, _NUM_STAGES = 64, 64, 4, 2, 2


@triton.jit
def _l2_direct_kernel(QT, XT, OUT, M, N, D, s_qd, s_xd, s_om, BM: tl.constexpr, BN: tl.constexpr, UNROLL: tl.constexpr):
    """OUT[m, n] = || QT[:, m] - XT[:, n] ||_2 with fp16 loads and fp32 arithmetic."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mask_m = offs_m < M
    mask_n = offs_n < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for d0 in range(0, D, UNROLL):
        for u in tl.static_range(UNROLL):
            d = d0 + u
            q = tl.load(QT + d * s_qd + offs_m, mask=mask_m & (d < D), other=0.0).to(tl.float32)
            x = tl.load(XT + d * s_xd + offs_n, mask=mask_n & (d < D), other=0.0).to(tl.float32)
            diff = q[:, None] - x[None, :]
            acc += diff * diff
    tl.store(OUT + offs_m[:, None] * s_om + offs_n[None, :], tl.sqrt(acc), mask=mask_m[:, None] & mask_n[None, :])


def l2_distance_T(qT: torch.Tensor, xT: torch.Tensor) -> torch.Tensor:
    """Pairwise L2 distances between the columns of ``qT`` (D, M) and ``xT`` (D, N). Returns (M, N) float32.

    Both inputs must be fp16 CUDA tensors with unit stride along the column axis (e.g. column slices of a contiguous
    ``(D, N)`` matrix).
    """
    assert qT.dtype == torch.float16 and xT.dtype == torch.float16, "l2_distance_T expects fp16 inputs"
    assert qT.stride(1) == 1 and xT.stride(1) == 1, "l2_distance_T expects unit stride along columns"
    D, M = qT.shape
    N = xT.shape[1]
    assert xT.shape[0] == D
    out = torch.empty((M, N), device=qT.device, dtype=torch.float32)
    if M == 0 or N == 0:
        return out
    grid = (triton.cdiv(M, _BM), triton.cdiv(N, _BN))
    _l2_direct_kernel[grid](
        qT, xT, out, M, N, D, qT.stride(0), xT.stride(0), out.stride(0),
        BM=_BM, BN=_BN, UNROLL=_UNROLL, num_warps=_NUM_WARPS, num_stages=_NUM_STAGES,
    )
    return out
