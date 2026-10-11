import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from picurate.dedupe.db_store import SCHEMA_VERSION, EmbeddingDB, ImageRecord, ModelMismatchError, load_database


def rec(path, mtime=1.0, size=1, width=4, height=3, fmt="JPEG"):
    return ImageRecord(path, mtime, size, width, height, fmt)


class EmbeddingDBTests(unittest.TestCase):
    def test_schema_version_mismatch_requires_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "db.sqlite")
            with EmbeddingDB(path) as db:
                db.ensure_model("m")
                db.set_meta("schema_version", "0")
                db.conn.commit()
            with EmbeddingDB(path) as db:
                with self.assertRaises(ModelMismatchError):
                    db.ensure_model("m")
                db.ensure_model("m", reset_on_mismatch=True)
                self.assertEqual(db.get_meta("schema_version"), SCHEMA_VERSION)

    def test_roundtrip_upsert_load_delete(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested", "db.sqlite")
            vecs = np.arange(12, dtype=np.float32).reshape(3, 4)
            with EmbeddingDB(path) as db:
                db.ensure_model("model-a")
                db.upsert([(rec(f"{i}.jpg", 100.0 + i, 10 * i), vecs[i]) for i in range(3)])
                self.assertEqual(db.dim, 4)
                self.assertEqual(len(db), 3)
                self.assertEqual(db.load_index(), {"0.jpg": 100.0, "1.jpg": 101.0, "2.jpg": 102.0})

                # replace one row, delete another
                db.upsert([(rec("1.jpg", 999.0, 1), vecs[1] * 2)])
                self.assertEqual(db.delete(["2.jpg"]), 1)

            with EmbeddingDB(path) as db:
                records, arr = db.load_all()
                self.assertEqual([r.path for r in records], ["0.jpg", "1.jpg"])
                self.assertEqual(records[1], rec("1.jpg", 999.0, 1))
                self.assertEqual(records[0].pixels, 4 * 3)
                self.assertEqual(arr.dtype, np.float16)
                self.assertEqual(arr.shape, (2, 4))
                np.testing.assert_array_equal(arr[0], vecs[0])
                np.testing.assert_array_equal(arr[1], vecs[1] * 2)
                self.assertTrue(arr.flags.writeable)

    def test_model_mismatch_raises_unless_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "db.sqlite")
            with EmbeddingDB(path) as db:
                db.ensure_model("model-a")
                db.upsert([(rec("a.jpg"), np.zeros(2, dtype=np.float32))])
            with EmbeddingDB(path) as db:
                with self.assertRaises(ModelMismatchError):
                    db.ensure_model("model-b")
                db.ensure_model("model-b", reset_on_mismatch=True)
                self.assertEqual(len(db), 0)
                self.assertEqual(db.model_id, "model-b")
                self.assertIsNone(db.dim)

    def test_dimension_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with EmbeddingDB(os.path.join(tmp, "db.sqlite")) as db:
                db.ensure_model("m")
                db.upsert([(rec("a.jpg"), np.zeros(3, dtype=np.float32))])
                with self.assertRaises(ValueError):
                    db.upsert([(rec("b.jpg"), np.zeros(4, dtype=np.float32))])

    def test_load_database_missing_file_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            records, arr = load_database(os.path.join(tmp, "missing.sqlite"))
            self.assertEqual(records, [])
            self.assertEqual(arr.shape[0], 0)


if __name__ == "__main__":
    unittest.main()
