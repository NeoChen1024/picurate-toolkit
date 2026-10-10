import io
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import PIL.Image
import PIL.ImageDraw

from clip_image_deduper import pixel_check as pc


def picture(w=640, h=480, seed=0) -> PIL.Image.Image:
    """Smooth shading with dark line art on top: enough texture for JPEG ringing, like an illustration."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.stack([128 + 80 * np.sin(x / (40 + 10 * c) + c) * np.cos(y / (55 + 7 * c)) for c in range(3)], axis=2)
    img = PIL.Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")
    draw = PIL.ImageDraw.Draw(img)
    for _ in range(25):
        x0, y0, x1, y1 = rng.integers(0, w), rng.integers(0, h), rng.integers(0, w), rng.integers(0, h)
        draw.line((int(x0), int(y0), int(x1), int(y1)), fill=(20, 20, 30), width=int(rng.integers(1, 4)))
    return img


def jpeg(img: PIL.Image.Image, q: int) -> PIL.Image.Image:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=q)
    buf.seek(0)
    return PIL.Image.open(buf).convert("RGB")


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.img = picture()

    def test_compression_only_passes(self):
        for q in (90, 75, 60):
            v = pc.compare_images(self.img, jpeg(self.img, q))
            self.assertTrue(v.same, (q, v))
        half = jpeg(self.img.resize((320, 240), PIL.Image.Resampling.LANCZOS), 85)
        self.assertTrue(pc.compare_images(self.img, half).same)
        identical = pc.compare_images(self.img, self.img.copy())
        self.assertEqual((identical.status, identical.peak, identical.mean), (pc.SAME, 0.0, 0.0))

    def test_local_edit_fails(self):
        edited = self.img.copy()
        PIL.ImageDraw.Draw(edited).text((500, 440), "@artist", fill=(255, 255, 255), font_size=14)
        v = pc.compare_images(self.img, jpeg(edited, 90))
        self.assertEqual(v.status, pc.DIFFERENT, v)

    def test_local_edit_on_a_lossless_copy_fails_by_the_peak_ratio(self):
        # The case a peak threshold alone misses: everything identical except one small mark.
        edited = self.img.copy()
        PIL.ImageDraw.Draw(edited).rectangle((300, 200, 302, 202), fill=(250, 250, 250))  # a 3x3 px mark
        v = pc.compare_images(self.img, edited)
        self.assertLessEqual(v.peak, pc.MAX_PEAK)
        self.assertEqual(v.status, pc.DIFFERENT, v)

    def test_global_colour_change_fails(self):
        arr = np.asarray(self.img, dtype=np.int16) + np.array([8, 0, -8], dtype=np.int16)
        tinted = PIL.Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
        self.assertEqual(pc.compare_images(self.img, jpeg(tinted, 90)).status, pc.DIFFERENT)

    def test_other_aspect_ratio_is_not_compared(self):
        crop = self.img.crop((0, 0, 600, 480))
        self.assertEqual(pc.compare_images(self.img, crop).status, pc.ASPECT)

    def test_transparency_is_composited(self):
        rgba = self.img.convert("RGBA")
        rgba.putalpha(PIL.Image.new("L", rgba.size, 255))
        arr = np.asarray(rgba).copy()
        arr[:50, :50, 3] = 0  # transparent corner
        hidden_a, hidden_b = arr.copy(), arr.copy()
        hidden_a[:50, :50, :3] = 0
        hidden_b[:50, :50, :3] = 255  # different colours under full transparency
        with tempfile.TemporaryDirectory() as tmp:
            pa, pb = os.path.join(tmp, "a.png"), os.path.join(tmp, "b.png")
            PIL.Image.fromarray(hidden_a, "RGBA").save(pa)
            PIL.Image.fromarray(hidden_b, "RGBA").save(pb)
            self.assertEqual(pc.compare_files(pa, pb).peak, 0.0)

    def test_unreadable_file_is_an_error_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            pa = os.path.join(tmp, "a.png")
            self.img.save(pa)
            v = pc.compare_files(pa, os.path.join(tmp, "missing.png"))
        self.assertEqual(v.status, pc.ERROR)
        self.assertFalse(v.same)


class CheckPairsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = picture()
        other = picture(seed=1)
        self.paths = {}
        for name, img in [("a", base), ("b", jpeg(base, 80)), ("c", jpeg(base, 70)), ("x", other), ("y", jpeg(other, 85))]:
            path = os.path.join(self.tmp.name, f"{name}.png")
            img.save(path)
            self.paths[name] = path

    def tearDown(self):
        self.tmp.cleanup()

    def pairs(self, *names):
        return [(self.paths[a], self.paths[b]) for a, b in names]

    def test_results_in_order_and_each_image_decoded_once(self):
        calls = []
        real = pc.load_rgb

        def counting(path):
            calls.append(os.path.basename(path))
            return real(path)

        with mock.patch.object(pc, "load_rgb", counting):
            verdicts = pc.check_pairs(self.pairs(("a", "b"), ("x", "y"), ("b", "c"), ("a", "c"), ("a", "x")), workers=1)
        self.assertEqual([v.status for v in verdicts], [pc.SAME, pc.SAME, pc.SAME, pc.SAME, pc.DIFFERENT])
        self.assertEqual(sorted(calls), ["a.png", "b.png", "c.png", "x.png", "y.png"])

    def test_tiny_buffer_still_correct(self):
        calls = []
        real = pc.load_rgb
        with mock.patch.object(pc, "load_rgb", lambda p: calls.append(p) or real(p)):
            verdicts = pc.check_pairs(self.pairs(("a", "b"), ("b", "c"), ("a", "c")), workers=1, buffer_bytes=1)
        self.assertEqual([v.status for v in verdicts], [pc.SAME] * 3)
        self.assertGreater(len(calls), 3)  # nothing beyond the current pair was kept

    def test_parallel_matches_serial(self):
        pairs = self.pairs(("a", "b"), ("x", "y"), ("a", "x"))
        self.assertEqual(pc.check_pairs(pairs, workers=2), pc.check_pairs(pairs, workers=1))
