import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from picurate.common.images import ImageRecord
from picurate.dedupe.dedupe import find_cross_duplicates, find_duplicate_groups, move_to_trash, trash_duplicate_groups
from picurate.dedupe.keeping import builtin_policies
from picurate.dedupe.similarity import DistanceIndex


class FindDuplicateGroupsTests(unittest.TestCase):
    def test_chain_matches_are_merged(self):
        emb = np.array([[0.0], [0.9], [1.8]], dtype=np.float32)
        groups = find_duplicate_groups(["A", "B", "C"], DistanceIndex(emb, "cpu"), threshold=1.0)
        self.assertEqual(groups, [[0, 1, 2]])

    def test_groups_span_block_boundaries(self):
        rng = np.random.default_rng(0)
        base = rng.standard_normal((40, 8)).astype(np.float32) * 10
        dup = base[[3, 17, 29, 38]] + 0.01
        emb = np.concatenate([base, dup])
        index = DistanceIndex(emb, "cpu")
        seen = []
        with mock.patch.object(DistanceIndex, "query_block_size", return_value=7):
            groups = find_duplicate_groups([f"img{i}" for i in range(44)], index, threshold=0.5, progress=seen.append)
        self.assertEqual(sorted(groups), [[3, 40], [17, 41], [29, 42], [38, 43]])
        self.assertEqual(sum(seen), 44)

    def test_cross_duplicates(self):
        base = DistanceIndex(np.array([[0.0], [10.0], [20.0]], dtype=np.float32), "cpu")
        queries = DistanceIndex(np.array([[0.05], [10.0], [55.0], [19.9]], dtype=np.float32), "cpu")
        hits = find_cross_duplicates(["q0", "q1", "q2", "q3"], queries, ["b0", "b1", "b2"], base, threshold=0.2)
        self.assertEqual(hits, {0: 1, 1: 1, 3: 1})


class TrashTests(unittest.TestCase):
    def test_move_and_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, trash = os.path.join(tmp, "root"), os.path.join(tmp, "trash")
            os.makedirs(os.path.join(root, "sub"))
            open(os.path.join(root, "sub", "a.jpg"), "w").close()
            self.assertTrue(move_to_trash(root, "sub/a.jpg", trash, dry_run=True))
            self.assertTrue(os.path.exists(os.path.join(root, "sub", "a.jpg")))
            self.assertTrue(move_to_trash(root, "sub/a.jpg", trash, dry_run=False))
            self.assertFalse(os.path.exists(os.path.join(root, "sub", "a.jpg")))
            self.assertTrue(os.path.exists(os.path.join(trash, "sub", "a.jpg")))
            self.assertFalse(move_to_trash(root, "missing.jpg", trash, dry_run=False))

    def test_trash_duplicate_groups_keeps_policy_winner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, trash = os.path.join(tmp, "root"), os.path.join(tmp, "trash")
            os.makedirs(root)
            records = []
            for name, size in (("a.jpg", 1), ("b.jpg", 3), ("c.jpg", 2), ("solo.jpg", 9)):
                open(os.path.join(root, name), "w").close()
                records.append(ImageRecord(name, 0.0, size, 1, 1, "JPEG"))
            moved = trash_duplicate_groups([[0, 1, 2]], records, root, trash, builtin_policies()["largest"], dry_run=False)
            self.assertEqual(moved, 2)
            self.assertEqual(sorted(os.listdir(root)), ["b.jpg", "solo.jpg"])
            self.assertEqual(sorted(os.listdir(trash)), ["a.jpg", "c.jpg"])


if __name__ == "__main__":
    unittest.main()
