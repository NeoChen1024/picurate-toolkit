import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import PIL.Image
from click.testing import CliRunner

from picurate.dedupe import calibrate as cal
from picurate.dedupe import cli as cli_mod
from picurate.dedupe.cli import cli
from picurate.common.images import ImageRecord
from picurate.dedupe.similarity import DistanceIndex


def _records(*names):
    return [ImageRecord(n, 0.0, 1, 1, 1, "JPEG") for n in names]


class StubEncoder:
    """Embedding = mean RGB of the (resized) image, so variants land close to the original."""

    model_id = "stub"

    def preprocess(self, img):
        return np.asarray(img.resize((4, 4)), dtype=np.float32).reshape(-1, 3)

    def encode_images(self, preprocessed):
        return np.stack([p.mean(axis=0) for p in preprocessed]).astype(np.float32)


class SamplingTests(unittest.TestCase):
    def test_same_seed_same_paths_regardless_of_row_order(self):
        a = _records("c", "a", "b", "d", "e")
        b = _records("e", "d", "c", "b", "a")
        pa = {a[i].path for i in cal.sample_rows(a, 3, seed=7)}
        pb = {b[i].path for i in cal.sample_rows(b, 3, seed=7)}
        self.assertEqual(pa, pb)
        self.assertEqual(len(pa), 3)
        self.assertNotEqual(pa, {a[i].path for i in cal.sample_rows(a, 3, seed=8)})

    def test_sample_larger_than_db_takes_everything(self):
        self.assertEqual(cal.sample_rows(_records("a", "b"), 10, seed=0), [0, 1])


class DistanceTests(unittest.TestCase):
    def test_nearest_other_excludes_self(self):
        emb = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 1.0]], dtype=np.float32)
        index = DistanceIndex(emb, "cpu")
        d = cal.nearest_other_distances(index, [0, 1])
        np.testing.assert_allclose(d, [1.0, np.hypot(3, 3)], rtol=1e-5)

    def test_calibrate_measures_variants_against_stored_embeddings(self):
        with tempfile.TemporaryDirectory() as tmp:
            names = []
            for i, colour in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
                name = f"{i}.png"
                PIL.Image.new("RGB", (32, 32), colour).save(os.path.join(tmp, name))
                names.append(name)
            with open(os.path.join(tmp, "broken.png"), "wb") as f:
                f.write(b"nope")
            names.append("broken.png")
            records = _records(*names)
            enc = StubEncoder()
            emb = np.stack([enc.encode_images([enc.preprocess(PIL.Image.open(os.path.join(tmp, n)).convert("RGB"))])[0] for n in names[:3]] + [np.zeros(3, np.float32)])
            emb = emb.astype(np.float16)
            result = cal.calibrate(enc, tmp, records, emb, DistanceIndex(emb, "cpu"), samples=4, seed=0, variants=("jpeg90", "half"), batch_size=3)
            self.assertEqual(result.skipped, ["broken.png"])
            self.assertEqual(set(result.series), {"nearest-other", "jpeg90", "half"})
            self.assertEqual(len(result.series["jpeg90"]), 3)
            self.assertTrue((result.series["half"] < 1).all())  # a flat colour survives downscaling exactly
            self.assertEqual(len(result.series["nearest-other"]), 4)

    def test_unknown_variant_rejected(self):
        with self.assertRaises(ValueError):
            cal.calibrate(StubEncoder(), ".", [], np.empty((0, 3)), None, samples=1, seed=0, variants=("nope",))


class ReportTests(unittest.TestCase):
    def test_histogram_lines_and_suggestion(self):
        result = cal.Calibration(series={
            "nearest-other": np.array([5.6, 7.0, 9.0], dtype=np.float32),
            "jpeg90": np.array([0.3, 0.45, 0.52], dtype=np.float32),
            "half": np.array([0.0, 0.2, 0.4], dtype=np.float32),
        })
        lines = cal.report(result, thresholds=(0.1, 1.0))
        text = "\n".join(lines)
        self.assertIn("jpeg90 (3)", text)
        self.assertIn("#", text)
        self.assertIn("      0", text)  # the zero-distance row
        self.assertIn("At threshold 0.1: 16.7% of variants caught, 0 of 3", text)
        self.assertIn("At threshold 1: 100.0% of variants caught, 0 of 3", text)
        self.assertIn("Suggested threshold for jpeg90, half: 0.52", text)

    def test_overlap_reports_no_clean_threshold(self):
        result = cal.Calibration(series={"nearest-other": np.array([0.3], dtype=np.float32), "half": np.array([0.5], dtype=np.float32)})
        self.assertIn("No clean threshold", "\n".join(cal.report(result)))

    def test_edges_cover_all_values_in_whole_decades(self):
        edges = cal.log_edges([np.array([0.03, 7.0])])
        self.assertAlmostEqual(edges[0], 0.01)
        self.assertAlmostEqual(edges[-1], 10.0)
        self.assertEqual(len(edges), 3 * 8 + 1)


class CliTests(unittest.TestCase):
    def test_calibrate_passes_seed_and_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            emb = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
            with mock.patch.object(cli_mod, "CLIPImageEncoder") as enc_cls, mock.patch.object(cli_mod, "EmbeddingDB") as db_cls, mock.patch.object(
                cli_mod, "load_database", return_value=(_records("a.jpg", "b.jpg"), emb)
            ), mock.patch.object(cli_mod, "calibrate", return_value=cal.Calibration()) as run, mock.patch.object(cli_mod, "report", return_value=["ok"]):
                enc_cls.return_value.model_id = "m"
                db_cls.return_value.__enter__.return_value.model_id = "m"
                res = CliRunner().invoke(cli, ["calibrate", "-i", tmp, "-d", "x.sqlite", "--skip-update", "-c", "cpu", "--seed", "42", "--variants", "half,jpeg90", "-s", "5", "-t", "0.3"])
            self.assertEqual(res.exit_code, 0, res.output)
            self.assertIn("ok", res.output)
            kw = run.call_args.kwargs
            self.assertEqual((kw["seed"], kw["samples"], kw["variants"]), (42, 5, ["half", "jpeg90"]))

    def test_calibrate_rejects_unknown_variant(self):
        res = CliRunner().invoke(cli, ["calibrate", "-i", ".", "-d", "x.sqlite", "--variants", "nope"])
        self.assertNotEqual(res.exit_code, 0)
        self.assertIn("Unknown variants nope", res.output)

    def test_calibrate_refuses_model_mismatch(self):
        with mock.patch.object(cli_mod, "CLIPImageEncoder") as enc_cls, mock.patch.object(cli_mod, "EmbeddingDB") as db_cls:
            enc_cls.return_value.model_id = "new"
            db_cls.return_value.__enter__.return_value.model_id = "old"
            res = CliRunner().invoke(cli, ["calibrate", "-i", ".", "-d", "x.sqlite", "--skip-update", "-c", "cpu"])
        self.assertNotEqual(res.exit_code, 0)
        self.assertIn("encoded with model 'old'", res.output)
