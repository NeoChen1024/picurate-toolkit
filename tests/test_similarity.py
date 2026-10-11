import unittest

import numpy as np
import torch

from picurate.dedupe.similarity import DistanceIndex, euclidean_distance, find_close_pairs_cross, find_close_pairs_self


def _embeddings(n, d, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, d)) * 5).astype(np.float16)


class DistanceIndexCpuTests(unittest.TestCase):
    def test_cpu_backend_matches_reference(self):
        emb = _embeddings(50, 16)
        index = DistanceIndex(emb, "cpu")
        self.assertFalse(index.use_triton)
        got = index.distances(10, 20, other_start=5).numpy()
        ref = euclidean_distance(emb[10:20].astype(np.float32), emb[5:].astype(np.float32))
        # torch.cdist uses the |a|^2 + |b|^2 - 2ab form for inputs this size, which loses ~1e-2 absolute precision
        # near zero distance for vectors of norm ~20. That is the known limitation of the fallback backend.
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=2e-2)

    def test_self_pairs_are_upper_triangular_and_cross_pairs_complete(self):
        emb = _embeddings(30, 4)
        emb[7] = emb[2]
        emb[25] = emb[2]
        index = DistanceIndex(emb, "cpu")
        ii, jj, dd = find_close_pairs_self(index, 1e-6, 0, 30)
        self.assertEqual(sorted(zip(ii.tolist(), jj.tolist())), [(2, 7), (2, 25), (7, 25)])
        self.assertTrue((dd <= 1e-6).all())

        other = DistanceIndex(emb[[2, 9]], "cpu")
        ii, jj, _ = find_close_pairs_cross(index, other, 1e-6, 0, 30)
        self.assertEqual(sorted(zip(ii.tolist(), jj.tolist())), [(2, 0), (7, 0), (9, 1), (25, 0)])

    def test_block_size_respects_budget(self):
        index = DistanceIndex(_embeddings(1000, 2), "cpu")
        self.assertEqual(index.query_block_size(budget_bytes=4 * 1000 * 10), 10)
        self.assertEqual(index.query_block_size(budget_bytes=1), 1)
        self.assertEqual(index.query_block_size(budget_bytes=2**40), 1024)
        blocks = list(index.iter_blocks(budget_bytes=4 * 1000 * 300))
        self.assertEqual(blocks, [(0, 300), (300, 600), (600, 900), (900, 1000)])


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class DistanceIndexTritonTests(unittest.TestCase):
    def test_triton_backend_is_exact(self):
        emb = _embeddings(777, 1152, seed=1)
        index = DistanceIndex(emb, "cuda")
        self.assertTrue(index.use_triton)
        self.assertEqual(index.dbT.dtype, torch.float16)
        got = index.distances(100, 233, other_start=50).cpu().numpy().astype(np.float64)
        a = emb.astype(np.float64)
        ref = np.sqrt(((a[100:233, None, :] - a[None, 50:, :]) ** 2).sum(-1))
        self.assertLess(np.abs(got - ref).max(), 1e-3)
        # near-duplicates: the direct kernel must not suffer the cancellation of the matmul form
        emb2 = emb.copy()
        emb2[5] = emb2[4]
        emb2[6] = (emb2[4].astype(np.float32) + 1e-3).astype(np.float16)
        index2 = DistanceIndex(emb2, "cuda")
        d = index2.distances(4, 5).cpu().numpy()[0]
        self.assertEqual(d[5], 0.0)
        exact = np.linalg.norm(emb2[6].astype(np.float64) - emb2[4].astype(np.float64))
        self.assertAlmostEqual(float(d[6]), exact, places=4)

    def test_cross_index_and_empty_shapes(self):
        a = DistanceIndex(_embeddings(10, 32), "cuda")
        b = DistanceIndex(_embeddings(5, 32, seed=3), "cuda")
        self.assertEqual(a.distances(0, 10, other=b).shape, (10, 5))
        self.assertEqual(a.distances(10, 10, other=b).shape, (0, 5))


if __name__ == "__main__":
    unittest.main()
