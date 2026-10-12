import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import PIL.Image

from picurate.common.images import standalone_items, standalone_roots
from picurate.dedupe import encoding_pipeline
from picurate.dedupe.db_store import load_database
from picurate.dedupe.encoding_pipeline import update_database as _update_database


def update_database(encoder, image_dir, db_path, **kw):
    _update_database(encoder, standalone_roots(image_dir), standalone_items(image_dir), db_path, **kw)


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
                self.assertEqual([i.path for i in enc.call_args.args[3]], ["broken.jpg"])
            os.remove(os.path.join(image_dir, "broken.jpg"))

            # modify one, delete one -> one re-encode, one orphan removed
            os.utime(os.path.join(image_dir, "a.png"), (1_000, 1_000))
            os.remove(os.path.join(image_dir, "c.gif"))
            with mock.patch.object(encoding_pipeline, "encode_images", return_value=1) as enc:
                update_database(StubEncoder(), image_dir, db_path)
                self.assertEqual([i.path for i in enc.call_args.args[3]], ["a.png"])
            self.assertEqual([r.path for r in load_database(db_path)[0]], ["a.png", "d.webp", "sub/b.jpg"])

            # force update re-encodes everything
            with mock.patch.object(encoding_pipeline, "encode_images", return_value=3) as enc:
                update_database(StubEncoder(), image_dir, db_path, force_update=True)
                self.assertEqual(sorted(i.path for i in enc.call_args.args[3]), ["a.png", "d.webp", "sub/b.jpg"])


if __name__ == "__main__":
    unittest.main()
