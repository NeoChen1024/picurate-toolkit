"""Exact pairwise Euclidean distances between embeddings, in blocks, on CPU or GPU."""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator

import numpy as np
import torch

default_euclidean_distance_threshold = 0.1  # same images will have distance 0.0 to 1 depending on encoding model, adjust as needed

# Upper bound on the size of one (query_block, N) fp32 distance block. Keeps peak VRAM predictable on small GPUs.
default_distance_block_bytes = 256 * 2**20


def _triton_usable(device: torch.device) -> bool:
    return device.type == "cuda" and importlib.util.find_spec("triton") is not None


class DistanceIndex:
    """An ``(N, D)`` embedding matrix resident on a device, able to produce exact L2 distance blocks.

    On CUDA with Triton available the matrix is kept in fp16 (half the VRAM) and distances are computed by the
    direct-difference kernel in :mod:`.l2_triton` (fp32 math, no cancellation). Everywhere else the matrix is kept
    in fp32 and ``torch.cdist`` is used.
    """

    def __init__(self, embeddings: np.ndarray, device: str):
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be (N, D), got shape {embeddings.shape}")
        self.device = torch.device(device)
        self.n, self.dim = embeddings.shape
        self.use_triton = _triton_usable(self.device)
        host = torch.from_numpy(np.ascontiguousarray(embeddings))
        if self.use_triton:
            from .l2_triton import l2_distance_T  # imported lazily so CPU-only installs never touch triton

            self._kernel = l2_distance_T
            # Transpose on the host so the device never holds two copies of the matrix at once.
            self.dbT: torch.Tensor | None = host.to(torch.float16).T.contiguous().to(self.device)  # (D, N)
            self.db: torch.Tensor | None = None
        else:
            self._kernel = None
            self.dbT = None
            self.db = host.to(self.device, dtype=torch.float32)  # (N, D)

    @property
    def nbytes(self) -> int:
        t = self.dbT if self.dbT is not None else self.db
        assert t is not None
        return t.numel() * t.element_size()

    def backend_name(self) -> str:
        return "triton fp16-store/fp32-math direct kernel" if self.use_triton else "torch.cdist fp32"

    def distances(self, start: int, end: int, other: DistanceIndex | None = None, other_start: int = 0) -> torch.Tensor:
        """Distances from rows ``[start, end)`` of this index to rows ``[other_start, N_other)`` of ``other``.

        Returns a ``(end - start, N_other - other_start)`` fp32 tensor on the device. ``other`` defaults to ``self``.
        """
        other = self if other is None else other
        if other.use_triton != self.use_triton or other.device != self.device:
            raise ValueError("Both indices must live on the same device and backend")
        if self._kernel is not None and self.dbT is not None and other.dbT is not None:
            return self._kernel(self.dbT[:, start:end], other.dbT[:, other_start:])
        assert self.db is not None and other.db is not None
        return torch.cdist(self.db[start:end], other.db[other_start:], p=2)

    def query_block_size(self, other: DistanceIndex | None = None, budget_bytes: int = default_distance_block_bytes) -> int:
        """Number of query rows per block so that one distance block stays under ``budget_bytes``."""
        n_cols = (self if other is None else other).n
        return max(1, min(1024, budget_bytes // (4 * max(n_cols, 1))))

    def iter_blocks(self, other: DistanceIndex | None = None, budget_bytes: int = default_distance_block_bytes) -> Iterator[tuple[int, int]]:
        block = self.query_block_size(other, budget_bytes)
        for start in range(0, self.n, block):
            yield start, min(start + block, self.n)

    def release(self) -> None:
        self.db = None
        self.dbT = None


def find_close_pairs_self(index: DistanceIndex, threshold: float, start: int, end: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pairs ``(i, j, d)`` with ``start <= i < end``, ``j > i`` and ``d <= threshold`` within a single index.

    Only the strictly upper triangle is searched, so every unordered pair is reported exactly once over all blocks.
    """
    d = index.distances(start, end, other_start=start)  # (end - start, N - start)
    rows = torch.arange(d.shape[0], device=d.device)[:, None]
    cols = torch.arange(d.shape[1], device=d.device)[None, :]
    ii, jj = ((d <= threshold) & (cols > rows)).nonzero(as_tuple=True)
    return (ii + start).cpu().numpy(), (jj + start).cpu().numpy(), d[ii, jj].cpu().numpy()


def find_close_pairs_cross(
    queries: DistanceIndex, base: DistanceIndex, threshold: float, start: int, end: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pairs ``(i, j, d)`` with ``start <= i < end`` from ``queries`` and any ``j`` in ``base`` with ``d <= threshold``."""
    d = queries.distances(start, end, other=base)
    ii, jj = (d <= threshold).nonzero(as_tuple=True)
    return (ii + start).cpu().numpy(), jj.cpu().numpy(), d[ii, jj].cpu().numpy()


# ---------------------------------------------------------------------------------------------------------------------
# Small / reference helpers (kept for tests and the encoding test CLI).


# slow generic numpy version, for testing and reference
def euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the Euclidean distance matrix between two sets of vectors.

    Accepts a, b as (D,), (1, D), or (N, D)/(M, D).
    Returns shape (N, M).
    """
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
    return np.linalg.norm(diff, axis=-1)


@torch.no_grad()
def euclidean_distance_torch_1_to_many(
    a: torch.Tensor,
    b: torch.Tensor,
) -> np.ndarray:
    """
    a: (1, D) torch tensor
    b: (N, D) torch tensor already on correct device
    returns: (N,) numpy distances
    """
    # torch.cdist computes pairwise distances; result shape (1, N)
    d = torch.cdist(a, b, p=2)
    return d[0].cpu().numpy()


@torch.no_grad()
def find_similar_images_euclidean(
    image_idx: int, image_embedding_1d: np.ndarray, database: torch.Tensor, threshold: float = default_euclidean_distance_threshold
) -> list[tuple[int, float]]:
    """Find similar images in the database based on Euclidean distance (one query vs. a tensor).

    image_idx: index of the query image inside ``database``. Pass -1 when the
    query vector is not literally stored inside the tensor being searched
    (for example when searching an upper-triangular slice or a foreign DB).
    Using a global index against a sliced tensor will incorrectly filter out
    legitimate matches at slice-local index 0.
    image_embedding_1d: (D,) numpy array of the query image embedding
    database: (N, D) torch tensor of the database embeddings
    threshold: distance threshold for considering images as similar
    """
    image_embedding_unsqueezed = image_embedding_1d[np.newaxis, :]  # shape (1, D)
    image_embedding = torch.from_numpy(image_embedding_unsqueezed).to(database.device).float()
    distances = euclidean_distance_torch_1_to_many(image_embedding, database.float())

    # For a single query vector vs database, distances has shape (1, N).
    # Squeeze to 1D so indexing and thresholding behave as expected.
    if distances.ndim == 2 and distances.shape[0] == 1:
        distances = distances[0]

    matches = np.where(distances <= threshold)[0]
    if image_idx >= 0:
        similar_images = [(int(idx), float(distances[idx])) for idx in matches if idx != image_idx]
    else:
        similar_images = [(int(idx), float(distances[idx])) for idx in matches]
    return similar_images
