import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import PIL.Image
import torch

from picurate.dedupe import encoding_pipeline
from picurate.dedupe.db_store import EmbeddingDB, load_database
from picurate.dedupe.encoding_pipeline import find_candidates, is_image_path, update_database


def _fake_preprocess(img):
    return np.zeros((3, 4, 4), dtype=np.float16)


class StubEncoder:
    """Stands in for CLIPImageEncoder: picklable preprocessor, deterministic 'embedding' = mean RGB."""

    model_id = "stub"
    dim = 8

    def get_preprocessor(self):
        return _fake_preprocess

    def submit(self, arrays):
        assert all(a.dtype == np.float16 and a.shape == (3, 4, 4) for a in arrays), "workers must deliver fully preprocessed arrays"
        return len(arrays)

    def collect(self, pending):
        return np.random.default_rng(pending).random((pending, self.dim)).astype(np.float32)


def _make_images(image_dir):
    os.makedirs(os.path.join(image_dir, "sub"), exist_ok=True)
    PIL.Image.new("RGB", (8, 6), (255, 0, 0)).save(os.path.join(image_dir, "a.png"))
    PIL.Image.new("RGB", (4, 4), (0, 255, 0)).save(os.path.join(image_dir, "sub", "b.jpg"), quality=90)
    PIL.Image.new("P", (3, 2)).save(os.path.join(image_dir, "c.gif"))
    PIL.Image.new("RGB", (5, 5)).save(os.path.join(image_dir, "d.webp"))
    with open(os.path.join(image_dir, "notes.txt"), "w") as f:
        f.write("not an image")
    with open(os.path.join(image_dir, "broken.jpg"), "wb") as f:
        f.write(b"\xff\xd8garbage")


class PipelineTests(unittest.TestCase):
    def test_is_image_path(self):
        self.assertTrue(is_image_path("x/y.JPG"))
        self.assertTrue(is_image_path("x.gif"))
        self.assertFalse(is_image_path("x.txt"))
        self.assertFalse(is_image_path("noext"))

    def test_full_update_with_real_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir, db_path = os.path.join(tmp, "img"), os.path.join(tmp, "db.sqlite")
            _make_images(image_dir)
            with self.assertLogs("picurate.common.pipeline", level="WARNING") as logs:
                update_database(StubEncoder(), image_dir, db_path, batch_size=2, workers=2)
            self.assertTrue(any("broken.jpg" in m for m in logs.output))
            records, emb = load_database(db_path)
            self.assertEqual([r.path for r in records], ["a.png", "c.gif", "d.webp", "sub/b.jpg"])
            self.assertEqual(emb.shape, (4, 8))
            self.assertEqual(emb.dtype, np.float16)
            by_path = {r.path: r for r in records}
            self.assertEqual((by_path["a.png"].width, by_path["a.png"].height, by_path["a.png"].format), (8, 6, "PNG"))
            self.assertEqual(by_path["c.gif"].format, "GIF")
            self.assertEqual(by_path["d.webp"].format, "WEBP")
            self.assertEqual(by_path["sub/b.jpg"].format, "JPEG")
            self.assertEqual(by_path["sub/b.jpg"].size, os.path.getsize(os.path.join(image_dir, "sub", "b.jpg")))

            # second run: only the unreadable file is retried (it has no row), nothing else is re-encoded
            with mock.patch.object(encoding_pipeline, "encode_images", return_value=0) as enc:
                update_database(StubEncoder(), image_dir, db_path)
                self.assertEqual(list(enc.call_args.args[3]), ["broken.jpg"])
            os.remove(os.path.join(image_dir, "broken.jpg"))

            # modify one, delete one -> one re-encode, one orphan removed
            os.utime(os.path.join(image_dir, "a.png"), (1_000, 1_000))
            os.remove(os.path.join(image_dir, "c.gif"))
            with mock.patch.object(encoding_pipeline, "encode_images", return_value=1) as enc:
                update_database(StubEncoder(), image_dir, db_path)
                self.assertEqual(list(enc.call_args.args[3]), ["a.png"])
            self.assertEqual([r.path for r in load_database(db_path)[0]], ["a.png", "d.webp", "sub/b.jpg"])

            # force update re-encodes everything
            with mock.patch.object(encoding_pipeline, "encode_images", return_value=3) as enc:
                update_database(StubEncoder(), image_dir, db_path, force_update=True)
                self.assertEqual(sorted(enc.call_args.args[3]), ["a.png", "d.webp", "sub/b.jpg"])

    def test_find_candidates_ignores_non_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_images(tmp)
            candidates, seen = find_candidates(tmp, {})
            self.assertEqual(sorted(seen), ["a.png", "broken.jpg", "c.gif", "d.webp", "sub/b.jpg"])
            self.assertEqual(sorted(candidates), sorted(seen))
            candidates, _ = find_candidates(tmp, {"a.png": os.path.getmtime(os.path.join(tmp, "a.png"))})
            self.assertNotIn("a.png", candidates)

    def test_model_mismatch_is_runtime_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "db.sqlite")
            with EmbeddingDB(db_path) as db:
                db.ensure_model("other-model")
            with self.assertRaises(RuntimeError):
                update_database(StubEncoder(), tmp, db_path)
            update_database(StubEncoder(), tmp, db_path, force_update=True)  # rebinding is allowed


if __name__ == "__main__":
    unittest.main()
