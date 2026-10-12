import os
import tempfile
import unittest
from unittest import mock

import numpy as np
from click.testing import CliRunner

from picurate.common.images import ImageRecord
from picurate.dedupe import cli as cli_mod
from picurate.dedupe.cli import cli


def _records(*names):
    return [ImageRecord(n, 0.0, 1, 1, 1, "JPEG") for n in names]


class CliTests(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.tmp = tempfile.TemporaryDirectory()
        self.img = os.path.join(self.tmp.name, "img")
        os.makedirs(self.img)

    def tearDown(self):
        self.tmp.cleanup()

    def test_dedupe_dry_run_still_refreshes_database(self):
        trash = os.path.join(self.tmp.name, "trash")
        with (
            mock.patch.object(cli_mod, "CLIPImageEncoder"),
            mock.patch.object(cli_mod, "update_database") as upd,
            mock.patch.object(
                cli_mod, "load_database", return_value=(_records("a.jpg", "b.jpg"), np.array([[0.0], [0.05]], dtype=np.float32))
            ),
        ):
            result = self.runner.invoke(cli, ["dedupe", "-i", self.img, "-d", "db.sqlite", "-n", "--trash-dir", trash, "-c", "cpu"])
        self.assertEqual(result.exit_code, 0, result.output)
        upd.assert_called_once()
        self.assertTrue(upd.call_args.kwargs["clean_orphans"])
        self.assertFalse(os.path.exists(trash))  # dry run moved nothing


class PixelCheckCliTests(unittest.TestCase):
    """Real files: a.png, its JPEG re-encode b.jpg (same picture), and c.png with a signature (an edit)."""

    def setUp(self):
        import PIL.Image
        import PIL.ImageDraw

        from tests.test_pixel_check import picture

        self.tmp = tempfile.TemporaryDirectory()
        self.img = os.path.join(self.tmp.name, "img")
        os.makedirs(self.img)
        base = picture()
        base.save(os.path.join(self.img, "a.png"))
        base.save(os.path.join(self.img, "b.jpg"), quality=80)
        edited = base.copy()
        PIL.ImageDraw.Draw(edited).text((500, 440), "@artist", fill=(255, 255, 255), font_size=14)
        edited.save(os.path.join(self.img, "c.png"))
        self.records = []
        for name, size in [("a.png", 3), ("b.jpg", 1), ("c.png", 2)]:
            st = os.stat(os.path.join(self.img, name))
            self.records.append(ImageRecord(name, st.st_mtime, size, 640, 480, "PNG"))
        self.emb = np.array([[0.0], [0.03], [0.06]], dtype=np.float32)
        self.db = os.path.join(self.tmp.name, "pictures.sqlite")
        self.trash = os.path.join(self.tmp.name, "trash")

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, *args, records=None, emb=None):
        with (
            mock.patch.object(cli_mod, "load_database", return_value=(records or self.records, self.emb if emb is None else emb)),
            mock.patch.object(cli_mod, "EmbeddingDB") as store,
        ):
            store.return_value.__enter__.return_value.model_id = "m"
            return CliRunner().invoke(cli, [*args, "--skip-update", "-c", "cpu", "-j", "1"])

    def test_dedupe_merges_compression_only_and_keeps_the_edit(self):
        res = self.run_cli("dedupe", "-i", self.img, "-d", self.db, "--trash-dir", self.trash)
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertEqual(sorted(os.listdir(self.trash)), ["b.jpg"])  # largest (a.png) kept, the edit untouched
        self.assertTrue(os.path.exists(os.path.join(self.img, "c.png")))

    def test_review_mode_stores_pixel_results_and_reuses_them(self):
        from picurate.dedupe import review as rv

        res = self.run_cli("dedupe", "-i", self.img, "-d", self.db, "--review-threshold", "0.5")
        self.assertEqual(res.exit_code, 0, res.output)
        with rv.ReviewDB(rv.default_review_path(self.db)) as review:
            g = review.groups()[0]
            self.assertEqual((g.status, g.pixel), ("pending", "different"))  # the edit keeps the group in review
            self.assertEqual(review.edge_pixels(g.id)[("a.png", "b.jpg")].status, "same")
            self.assertEqual(review.get_meta("pixel_check"), "on")
        with mock.patch.object(cli_mod, "check_pairs", side_effect=AssertionError("nothing should be re-read")):
            res = self.run_cli("dedupe", "-i", self.img, "-d", self.db, "--review-threshold", "0.5")
        self.assertEqual(res.exit_code, 0, res.output)
        res = CliRunner().invoke(cli, ["review-status", "-d", self.db])
        self.assertIn("with pixel differences", res.output)

    def test_dedupe_import_ignores_an_edited_copy(self):
        base = os.path.join(self.tmp.name, "base")
        os.makedirs(base)
        os.rename(os.path.join(self.img, "a.png"), os.path.join(base, "a.png"))
        base_rec = [ImageRecord("a.png", 0.0, 3, 640, 480, "PNG")]
        imports = [r for r in self.records if r.path != "a.png"]
        with mock.patch.object(
            cli_mod,
            "load_database",
            side_effect=[(base_rec, np.array([[0.0]], dtype=np.float32)), (imports, np.array([[0.03], [0.06]], dtype=np.float32))],
        ):
            res = CliRunner().invoke(
                cli,
                [
                    "dedupe-import",
                    "--base-image-dir",
                    base,
                    "--base-db",
                    "b.sqlite",
                    "--import-image-dir",
                    self.img,
                    "--import-db",
                    "i.sqlite",
                    "--trash-dir",
                    self.trash,
                    "--skip-update",
                    "-c",
                    "cpu",
                    "-j",
                    "1",
                ],
            )
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertEqual(sorted(os.listdir(self.trash)), ["b.jpg"])


if __name__ == "__main__":
    unittest.main()
