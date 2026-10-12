"""Dataset projects and the editor's rendering: item identity, import, caption ownership, decisions and their
history, relinking, export transactions and rendering. These guard the user's captions and exported datasets; nothing here needs Qt.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from picurate.dataset_editor.render import export_one, export_project, load_image, render, save_repair
from picurate.project.project import STANDALONE_SOURCE, Project, item_id, validate_source


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / "input"
        self.root.mkdir()
        self.im = Image.new("RGB", (120, 100), (80, 100, 120))
        self.im.paste((255, 0, 0), (20, 20, 40, 40))
        self.path = self.root / "image.png"
        self.im.save(self.path)
        self.project = Project(self.base / "project", create=True)
        self.project.import_images([self.root])
        self.item_id = next(iter(self.project.items))
        self.state = self.project.state(self.item_id)

    def tearDown(self):
        self.project.close()
        self.temp.cleanup()

    def test_export_pixels(self):
        self.state.update(crop=[10, 10, 70, 60], fills=[{"rect": [20, 20, 20, 20], "ring_px": 3}], caption="a caption")
        for fmt in ("PNG", "WEBP"):
            export_one(self.path, self.base / fmt, self.state, fmt, "image")
            image = load_image(self.base / fmt / ("image." + fmt.lower()))
            self.assertEqual(image.tobytes(), render(self.im, self.state).tobytes())
            self.assertEqual(image.getpixel((15, 15)), (80, 100, 120))
            self.assertEqual((self.base / fmt / "image.txt").read_text(), "a caption")

    def test_no_sampling_pixels(self):
        self.state["fills"] = [{"rect": [0, 0, 120, 100], "ring_px": 4}]
        with self.assertRaises(ValueError):
            render(self.im, self.state)

    def test_item_id_derivation(self):
        """Every database is keyed by this; a change would silently orphan their rows."""
        self.assertEqual(
            item_id(STANDALONE_SOURCE, "a/b.png").hex(), "7df4921e2c1d3afbf9a3bb76701be3bbfe7afe7e081d5390b163f2b87f7492dc"
        )
        self.assertEqual(
            item_id(bytes(range(32)), "dir/\udcff.png").hex(), "2dcc9f6f5c8b4f60ad87279e7f26fe0d6d1aa5e23b423eddc9a2fa7196ef3247"
        )

    def test_decisions_and_history(self):
        db = self.project.db
        op = self.project.begin_op("edit")
        self.assertEqual(self.project.save(self.state, op), [])  # nothing differs from the defaults
        self.state["caption"] = "first"
        self.state["review_status"] = "approved"
        self.assertEqual(self.project.save(self.state, op), ["caption", "review_status"])
        self.state["caption"] = "second"
        self.project.save(self.state, op)
        self.assertEqual(db.execute("SELECT value FROM history WHERE field = 'caption'").fetchall(), [('"second"',)])
        later = self.project.begin_op("edit")
        self.state["caption"] = ""  # back to the derived (imported) caption: the row goes, the history keeps the step
        self.project.save(self.state, later)
        self.assertIsNone(db.execute("SELECT 1 FROM state WHERE field = 'caption'").fetchone())
        self.assertEqual(
            db.execute("SELECT op, value FROM history WHERE field = 'caption' ORDER BY seq").fetchall(),
            [(op, '"second"'), (later, None)],
        )
        reopened = Project(self.project.directory)
        self.assertEqual(reopened.state(self.item_id)["review_status"], "approved")
        self.assertEqual(reopened.state(self.item_id)["caption"], "")

    def test_one_editor_at_a_time(self):
        self.project.lock()
        other = Project(self.project.directory)
        with self.assertRaises(ValueError):
            other.lock()
        self.project.close()
        other.lock()

    def test_source_change(self):
        """A changed file only warns; edits made on an image of another size are refused."""
        self.assertIsNone(validate_source(self.path, self.state))
        self.state.update(crop=[0, 0, 50, 50], image_size=[120, 100])
        Image.new("RGB", (100, 100)).save(self.path)
        self.assertIn("changed", validate_source(self.path, self.state))
        with self.assertRaises(ValueError):
            render(load_image(self.path), self.state)

    def test_import_snapshot_dedup_and_collision(self):
        other = self.base / "other"
        other.mkdir()
        self.im.save(other / "image.png")
        (other / "image.txt").write_text("original caption", encoding="utf-8")
        imported, skipped = self.project.import_images([other])
        self.assertEqual((imported, skipped), (1, 0))
        second = list(self.project.items)[1]
        self.assertNotEqual(self.project.items[second]["output_stem"], self.project.items[self.item_id]["output_stem"])
        state = self.project.state(second)
        self.assertEqual(state["caption"], "original caption")
        state["caption"] = "edited caption"
        self.project.save(state, self.project.begin_op("edit"))
        (other / "image.txt").write_text("externally changed", encoding="utf-8")
        self.assertEqual(self.project.import_images([other]), (0, 1))
        reopened = Project(self.project.directory)
        self.assertEqual(reopened.state(second)["caption"], "edited caption")
        self.assertEqual((other / "image.txt").read_text(), "externally changed")
        self.im.save(other / "new.png")
        self.assertEqual(len(Project(self.project.directory).items), 2)  # No implicit scan.
        self.assertEqual(self.project.import_images([other], captions=False), (1, 1))
        self.assertEqual(self.project.state(list(self.project.items)[2])["caption"], "")

    def test_relink_and_missing_source(self):
        source_id = self.project.items[self.item_id]["source_id"]
        moved = self.base / "moved"
        shutil.move(self.root, moved)
        self.assertEqual(Project(self.project.directory).state(self.item_id)["caption"], "")
        self.project.relink(source_id, moved)
        self.assertEqual(self.project.source_path(self.item_id), moved / "image.png")
        validate_source(self.project.source_path(self.item_id), self.state)
        bad = self.base / "bad"
        bad.mkdir()
        Image.new("RGB", (120, 100), "white").save(bad / "image.png")
        with self.assertRaises(ValueError):
            self.project.relink(source_id, bad)
        self.assertEqual(self.project.source_path(self.item_id), moved / "image.png")

    def test_export_transaction(self):
        self.state["caption"] = "saved caption"
        snapshot = self.project.export_snapshot([self.state], "PNG")
        self.state["caption"] = "later edit"
        directory = self.project.directory
        output = directory / "output"
        self.assertEqual(export_project(directory, snapshot), (1, False))
        self.assertEqual((output / "image.txt").read_text(), "saved caption")
        (output / "notes.md").write_text("keep me")
        webp = self.project.export_snapshot([self.state], "WEBP")
        self.assertEqual(export_project(directory, webp, cancelled=lambda: True), (0, True))
        self.assertTrue((output / "image.png").exists())
        self.path.write_bytes(b"changed")
        with self.assertRaises(OSError):  # unreadable now
            export_project(directory, webp)
        self.assertTrue((output / "image.png").exists())
        self.im.save(self.path)
        export_project(directory, webp)
        self.assertFalse((output / "image.png").exists())
        self.assertTrue((output / "image.webp").exists())
        self.assertEqual((output / "notes.md").read_text(), "keep me")
        self.assertEqual((output / "image.txt").read_text(), "later edit")
        self.assertFalse(list(directory.glob(".export-*")))

    def test_export_unmanaged_collision(self):
        output = self.project.directory / "output"
        (output / "image.txt").write_text("unmanaged")
        snapshot = self.project.export_snapshot([self.state], "PNG")
        with self.assertRaises(ValueError):
            export_project(self.project.directory, snapshot)
        self.assertEqual((output / "image.txt").read_text(), "unmanaged")

    def test_output_swap_recovery(self):
        directory = self.project.directory
        (directory / "output" / "previous.txt").write_text("previous")
        (directory / "output").rename(directory / ".output-previous")
        Project(directory)
        self.assertEqual((directory / "output" / "previous.txt").read_text(), "previous")

    def test_repair_persistence_export_and_staleness(self):
        mask = Image.new("L", (20, 20), 0)
        ImageDraw.Draw(mask).rectangle((4, 4, 15, 15), fill=255)
        patch = Image.new("RGB", (20, 20), (10, 20, 30))
        op = save_repair(self.project.directory, self.item_id, self.im, [20, 20, 20, 20], mask, patch, {"test": True})
        self.state["fills"] = [op]
        self.project.save(self.state, self.project.begin_op("edit"))
        reopened = Project(self.project.directory)
        result = render(self.im, reopened.state(self.item_id), project_dir=self.project.directory)
        self.assertEqual(result.getpixel((25, 25)), (10, 20, 30))
        self.assertEqual(result.getpixel((20, 20)), self.im.getpixel((20, 20)))
        self.state["crop"] = [10, 10, 60, 60]
        snapshot = self.project.export_snapshot([self.state], "PNG")
        export_project(self.project.directory, snapshot)
        self.assertEqual(load_image(self.project.directory / "output/image.png").size, (60, 60))
        self.state["fills"].insert(0, {"rect": [10, 10, 40, 40], "ring_px": 4})
        with self.assertRaisesRegex(ValueError, "Stale"):
            render(self.im, self.state, project_dir=self.project.directory)
        self.state["fills"] = [op]
        (self.project.directory / op["mask"]).write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            render(self.im, self.state, project_dir=self.project.directory)


if __name__ == "__main__":
    unittest.main()
