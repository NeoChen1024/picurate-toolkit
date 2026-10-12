import os
import tempfile
import unittest
from unittest import mock

import numpy as np
from click.testing import CliRunner

from picurate.common.images import ImageRecord
from picurate.dedupe import cli as cli_mod
from picurate.dedupe.cli import cli
from picurate.project.project import REVIEW, Project


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        img = os.path.join(self.tmp.name, "img")
        os.makedirs(img)
        for name in ("a.png", "b.png", "c.png"):
            open(os.path.join(img, name), "wb").close()
        self.project = Project(os.path.join(self.tmp.name, "project"), create=True)
        self.project.import_images([img])
        self.ids = {row["path"]: item for item, row in self.project.items.items()}

    def tearDown(self):
        self.project.close()
        self.tmp.cleanup()

    def test_update_keeps_rows_of_every_project_item(self):
        """Excluded and missing images are not encoded, but their rows stay; only removed items lose theirs."""
        state = self.project.state(self.ids["b.png"])
        state["exclusion"] = {"decision": "exclude", "reason": "test"}
        self.project.save(state, self.project.begin_op("edit"))
        self.project.db.execute("UPDATE items SET missing = 1 WHERE id = ?", (self.ids["c.png"],))
        with mock.patch.object(cli_mod, "CLIPImageEncoder"), mock.patch.object(cli_mod, "update_database") as upd:
            with mock.patch.object(cli_mod, "load_database", return_value=([], np.empty((0, 0), np.float16))):
                result = CliRunner().invoke(cli, ["update-db", str(self.project.directory), "-c", "cpu"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual([item.path for item in upd.call_args.args[2]], ["a.png"])
        self.assertEqual(set(upd.call_args.kwargs["keep_rows"]), set(self.ids.values()))


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
        self.emb = np.array([[0.0], [0.03], [0.06]], dtype=np.float32)
        self.trash = os.path.join(self.tmp.name, "trash")
        self.projects = []

    def tearDown(self):
        for project in self.projects:
            project.close()
        self.tmp.cleanup()

    def project(self, directory, sizes):
        """A project over ``directory`` and embedding records of its images."""
        project = Project(os.path.join(self.tmp.name, "project"), create=True)
        self.projects.append(project)
        project.import_images([directory])
        records = []
        for item, row in project.items.items():
            st = os.stat(project.source_path(item))
            records.append(ImageRecord(row["path"], st.st_mtime, sizes[row["path"]], 640, 480, "PNG", row["source_id"], item))
        return project, sorted(records, key=lambda r: r.path)

    def run_cli(self, *args, records, emb=None):
        with (
            mock.patch.object(cli_mod, "load_database", return_value=(records, self.emb if emb is None else emb)),
            mock.patch.object(cli_mod, "EmbeddingDB") as store,
        ):
            store.return_value.__enter__.return_value.model_id = "m"
            return CliRunner().invoke(cli, [*args, "--skip-update", "-c", "cpu", "-j", "1"])

    def test_review_mode_stores_pixel_results_and_reuses_them(self):
        from picurate.dedupe import review as rv

        project, records = self.project(self.img, {"a.png": 3, "b.jpg": 1, "c.png": 2})
        res = self.run_cli("dedupe", str(project.directory), "--review-threshold", "0.5", records=records)
        self.assertEqual(res.exit_code, 0, res.output)
        a, b = records[0].id, records[1].id
        with rv.ReviewDB(str(project.directory / REVIEW)) as review:
            g = review.groups()[0]
            self.assertEqual((g.status, g.pixel), ("pending", "different"))  # the edit keeps the group in review
            self.assertEqual(review.edge_pixels(g.id)[(a, b)].status, "same")
            self.assertEqual(review.get_meta("pixel_check"), "on")
        self.assertEqual(sorted(os.listdir(self.img)), ["a.png", "b.jpg", "c.png"])  # nothing moved
        with mock.patch.object(cli_mod, "check_pairs", side_effect=AssertionError("nothing should be re-read")):
            res = self.run_cli("dedupe", str(project.directory), "--review-threshold", "0.5", records=records)
        self.assertEqual(res.exit_code, 0, res.output)
        res = CliRunner().invoke(cli, ["review-status", str(project.directory)])
        self.assertIn("with pixel differences", res.output)

    def test_dedupe_import_ignores_an_edited_copy(self):
        base = os.path.join(self.tmp.name, "base")
        os.makedirs(base)
        os.rename(os.path.join(self.img, "a.png"), os.path.join(base, "a.png"))
        project, base_records = self.project(base, {"a.png": 3})
        imports = [ImageRecord(p, 0.0, 1, 640, 480, "PNG") for p in ("b.jpg", "c.png")]
        with (
            mock.patch.object(cli_mod, "CLIPImageEncoder"),
            mock.patch.object(cli_mod, "update_database"),
            mock.patch.object(
                cli_mod,
                "load_database",
                side_effect=[
                    (base_records, np.array([[0.0]], dtype=np.float32)),
                    (imports, np.array([[0.03], [0.06]], dtype=np.float32)),
                ],
            ),
        ):
            res = CliRunner().invoke(
                cli,
                [
                    "dedupe-import",
                    "-i",
                    self.img,
                    str(project.directory),
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
        self.assertEqual(sorted(os.listdir(base)), ["a.png"])


if __name__ == "__main__":
    unittest.main()
