"""Scores for the Culling tab, computed on the CPU: cosine similarity of every image's stored embedding to query
vectors, and tag probabilities. Nothing here imports Qt.

The embeddings stay in memory as the database stores them, float16 (2.3 KiB per image for SigLIP2 so400m), with one
float32 norm per row. A product converts blocks of rows to float32, small enough to stay in the CPU cache, on eight
threads with one torch thread each: about 100-200 ms for a million images and a few query vectors on an 8-core
desktop CPU. torch rather than numpy because numpy converts float16 without the CPU's conversion instructions,
ten times slower than the product itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..dedupe.db_store import load_rows

BLOCK = 1024  # rows per float32 block: 4.5 MiB at D = 1152
THREADS = 8


@dataclass
class EmbeddingIndex:
    """The stored embeddings of a list of items, row ``i`` for ``items[i]``; items without one score NaN."""

    model_id: str
    matrix: np.ndarray  # (N, D) float16
    norms: np.ndarray  # (N,) float32, NaN where there is no embedding

    @classmethod
    def load(cls, path: Path, items: Sequence[bytes]) -> EmbeddingIndex:
        model_id, matrix, found = load_rows(str(path), items)
        index = cls(model_id, matrix, np.ones(len(items), dtype=np.float32))
        norms = index._blocks(lambda block: torch.linalg.vector_norm(block, dim=1, keepdim=True), 1)[:, 0]
        norms[~found | (norms == 0)] = np.nan
        index.norms = norms
        return index

    def vector(self, row: int) -> np.ndarray | None:
        """Row ``row`` as a float32 vector, or None without an embedding."""
        return None if np.isnan(self.norms[row]) else self.matrix[row].astype(np.float32)

    def cosine(self, queries: np.ndarray) -> np.ndarray:
        """``(N, k)`` float32 cosine similarities of every row to the ``(k, D)`` query vectors."""
        q = torch.from_numpy(np.asarray(queries, dtype=np.float32))
        q = (q / torch.linalg.vector_norm(q, dim=1, keepdim=True)).T.contiguous()
        out = self._blocks(lambda block: block @ q, q.shape[1])
        out /= self.norms[:, None]
        return out

    def _blocks(self, function, width: int) -> np.ndarray:
        """``function`` of every float32 block of rows, gathered into an ``(N, width)`` float32 array. Sets torch to one
        thread while it runs (the threads here are the parallelism); callers run torch from one thread at a time."""
        matrix = torch.from_numpy(self.matrix)
        out = np.empty((len(self.matrix), width), dtype=np.float32)
        target = torch.from_numpy(out)

        def run(start: int) -> None:
            target[start : start + BLOCK] = function(matrix[start : start + BLOCK].float())

        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with ThreadPoolExecutor(THREADS) as pool:
                list(pool.map(run, range(0, len(self.matrix), BLOCK)))
        finally:
            torch.set_num_threads(threads)
        return out
