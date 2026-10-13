"""Offscreen checks of the curation GUI's Duplicates tab. Skipped when PySide6 is not installed."""

import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import PIL.Image

try:
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from picurate.curate.canvas import CompareCanvas
    from picurate.curate.window import CurateWindow
except ImportError:  # pragma: no cover
    QApplication = None  # type: ignore[assignment]

from picurate.common.images import ImageRecord
from picurate.dedupe.keeping import builtin_policies
from picurate.dedupe.review import ReviewDB
from picurate.project.project import REVIEW, Project


def wait_idle(tab) -> None:
    """Block until the tab's background jobs have finished and their results were delivered."""
    for _ in range(3):  # a delivered image can start a diff; a delivered diff is the end of the chain
        tab.pool.waitForDone()
        tab.prefetch_pool.waitForDone()
        QApplication.processEvents()
        QApplication.processEvents()


@unittest.skipIf(QApplication is None, "PySide6 not installed")
class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.img = os.path.join(self.tmp.name, "img")
        os.makedirs(os.path.join(self.img, "sub"))
        recs = {}
        for path, colour, size in [
            ("a.png", (255, 0, 0), (40, 30)),
            ("sub/b.png", (250, 0, 0), (20, 15)),
            ("c.png", (0, 255, 0), (40, 40)),
            ("d.png", (0, 250, 0), (40, 40)),
        ]:
            PIL.Image.new("RGB", size, colour).save(os.path.join(self.img, path))
        self.project = Project(os.path.join(self.tmp.name, "project"), create=True)
        self.project.import_images([self.img], recursive=True)
        self.ids = {}
        for item, row in self.project.items.items():
            st = os.stat(self.project.source_path(item))
            with PIL.Image.open(self.project.source_path(item)) as im:
                size = im.size
            recs[row["path"]] = ImageRecord(row["path"], st.st_mtime, st.st_size, *size, "PNG", row["source_id"], item)
            self.ids[row["path"]] = item
        self.review_path = os.path.join(self.project.directory, REVIEW)
        with ReviewDB(self.review_path) as db:
            db.begin_session(project_id="p", model_id="m", review_threshold=1.0, auto_threshold=0.1, policy="largest")
            db.upsert_groups(
                [
                    ([recs["a.png"], recs["sub/b.png"]], [(self.ids["a.png"], self.ids["sub/b.png"], 0.3)]),
                    ([recs["c.png"], recs["d.png"]], [(self.ids["c.png"], self.ids["d.png"], 0.05)]),
                ],
                builtin_policies()["largest"],
                0.1,
            )
        self.window = CurateWindow(self.project.directory)
        self.win = self.window.duplicates
        self.window.show()
        self.app.setActiveWindow(self.window)
        self.win.canvas.left.setFocus()
        wait_idle(self.win)

    def tearDown(self):
        self.window.close()
        self.project.close()
        self.tmp.cleanup()

    def key(self, key, modifier=Qt.KeyboardModifier.NoModifier):
        QTest.keyClick(self.win.canvas.left, key, modifier)
        wait_idle(self.win)

    def test_groups_listed_surest_first_and_images_loaded(self):
        self.assertEqual([g.min_distance for g in self.win.groups_model.groups], [0.05, 0.3])
        self.assertEqual(self.win.current.status, "decided")  # pre-decided within 0.1
        self.assertEqual(self.win.position_label.text(), "Group 1 / 2")
        self.assertEqual(self.win.canvas.left.image_size, (40, 40))
        self.assertEqual(self.win.member_list.count(), 2)

    def test_a_b_selection_and_keep_toggle_persist(self):
        self.key(Qt.Key.Key_X)
        members = self.win.members
        self.assertEqual([m.path for m in members], ["a.png", "sub/b.png"])
        self.assertEqual((self.win.a, self.win.b), (0, 1))  # a.png is largest, so A
        self.assertEqual(self.win.canvas.left.image_size, (40, 30))
        self.assertEqual(self.win.canvas.right.image_size, (20, 15))
        self.key(Qt.Key.Key_2)
        self.assertEqual((self.win.a, self.win.b), (1, 0))
        self.key(Qt.Key.Key_K)
        with ReviewDB(self.review_path) as db:
            self.assertEqual({m.path: m.keep for m in db.members(self.win.current.id)}, {"a.png": True, "sub/b.png": True})
        self.key(Qt.Key.Key_K, Qt.KeyboardModifier.ShiftModifier)
        with ReviewDB(self.review_path) as db:
            self.assertEqual({m.path: m.keep for m in db.members(self.win.current.id)}, {"a.png": False, "sub/b.png": True})
        self.key(Qt.Key.Key_P, Qt.KeyboardModifier.ShiftModifier)  # this group only
        with ReviewDB(self.review_path) as db:
            self.assertEqual({m.path: m.keep for m in db.members(self.win.current.id)}, {"a.png": True, "sub/b.png": False})
        self.key(Qt.Key.Key_U)
        with ReviewDB(self.review_path) as db:
            self.assertEqual({m.path: m.keep for m in db.members(self.win.current.id)}, {"a.png": False, "sub/b.png": True})

    def test_enter_decides_and_skip_cycle(self):
        self.key(Qt.Key.Key_X)
        gid = self.win.current.id
        self.key(Qt.Key.Key_Return)
        with ReviewDB(self.review_path) as db:
            self.assertEqual(db.group(gid).status, "decided")
        self.key(Qt.Key.Key_C)  # decided -> skipped
        with ReviewDB(self.review_path) as db:
            self.assertEqual(db.group(gid).status, "skipped")
        self.assertIn("Skipped", self.win.status_label.text())

    def test_letters_ignored_in_note_field_and_note_saved(self):
        self.key(Qt.Key.Key_X)
        self.key(Qt.Key.Key_F)
        self.assertTrue(self.win.note.hasFocus())
        QTest.keyClicks(self.win.note, "xz keep both")
        QTest.keyClick(self.win.note, Qt.Key.Key_Return)
        wait_idle(self.win)
        self.assertEqual(self.win.position_label.text(), "Group 2 / 2")  # x/z did not navigate
        with ReviewDB(self.review_path) as db:
            self.assertEqual(db.group(self.win.current.id).note, "xz keep both")
        QTest.keyClick(self.win.note, Qt.Key.Key_Escape)
        self.assertFalse(self.win.note.hasFocus())

    def test_decided_losers_are_excluded_unless_kept_by_hand(self):
        """What the project excludes (and so does not export) comes from the decided groups and the human decision."""

        def excluded():
            return sorted(state["image"] for state in self.project.states() if state["excluded"])

        self.assertEqual(excluded(), ["c.png"])  # c/d decided automatically, equal sizes: the policy kept d.png
        state = self.project.state(self.ids["c.png"])
        self.assertEqual(state["excluded_by"], {"by": "dedupe", "group": self.win.current.id, "kept": [self.ids["d.png"].hex()]})
        state["exclusion"] = {"decision": "keep", "reason": ""}
        self.project.save(state, self.project.begin_op("edit"))
        self.assertEqual(excluded(), [])
        self.key(Qt.Key.Key_X)
        self.key(Qt.Key.Key_Return)  # decide a/b: a.png is the larger one
        self.assertEqual(excluded(), ["sub/b.png"])
        other = Project(self.project.directory)
        with self.assertRaises(ValueError):  # the window holds the review database
            other.lock("review")
        other.close()

    def test_policy_pick_resets_all_pending_groups(self):
        pending = next(g for g in self.win.groups_model.groups if g.status == "pending")  # a.png (40x30) / sub/b.png (20x15)
        with ReviewDB(self.review_path) as db:
            db.decide(pending.id, keeps={self.ids["a.png"]: False, self.ids["sub/b.png"]: True})
        self.win.reload_groups(select_id=pending.id)
        self.win.policy_box.setCurrentText("largest")
        self.win.reset_pending_to_policy(confirm=False)
        wait_idle(self.win)
        with ReviewDB(self.review_path) as db:
            self.assertEqual({m.path: m.keep for m in db.members(pending.id)}, {"a.png": True, "sub/b.png": False})
            self.assertEqual(db.group(pending.id).status, "pending")  # keeps only; still to be reviewed
            self.assertEqual(db.get_meta("policy"), "largest")
        self.key(Qt.Key.Key_U)
        with ReviewDB(self.review_path) as db:
            self.assertEqual({m.path: m.keep for m in db.members(pending.id)}, {"a.png": False, "sub/b.png": True})

    def test_keeping_nothing_is_refused_and_r_discards_the_whole_group(self):
        self.key(Qt.Key.Key_X)
        gid = self.win.current.id
        self.key(Qt.Key.Key_K)  # untick a.png, the only keep
        self.assertIn("keeps nothing", self.win.status_label.text())
        self.assertTrue(self.win.groups_model.data(self.win.groups_model.index(1)).startswith("∅ "))
        self.key(Qt.Key.Key_Return)
        self.key(Qt.Key.Key_C)
        with ReviewDB(self.review_path) as db:
            self.assertEqual(db.group(gid).status, "pending")  # neither Enter nor C decides it
        self.win.filter.setCurrentText("keeps nothing")
        wait_idle(self.win)
        self.assertEqual([g.id for g in self.win.groups_model.groups], [gid])
        self.key(Qt.Key.Key_R)
        with ReviewDB(self.review_path) as db:
            g = db.group(gid)
            self.assertEqual((g.status, g.discard), ("decided", True))
        self.assertTrue(self.win.groups_model.data(self.win.groups_model.index(0)).startswith("✕ "))
        self.assertEqual(
            sorted(state["image"] for state in self.project.states() if state["excluded"]), ["a.png", "c.png", "sub/b.png"]
        )
        self.key(Qt.Key.Key_R)  # takes it back
        with ReviewDB(self.review_path) as db:
            g = db.group(gid)
            self.assertEqual((g.status, g.discard), ("pending", False))


@unittest.skipIf(QApplication is None, "PySide6 not installed")
class CanvasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_fit_follows_the_final_viewport(self):
        """A fit asked for before the layout has settled (mode switch, resize, 100% with scrollbars) must end up
        matching the viewport the view actually gets, not the stale one it had at the time."""
        from picurate.common.qt import qimage

        def expected(view):
            size = view.maximumViewportSize()
            return min((size.width() - 2) / view.image_size[0], (size.height() - 2) / view.image_size[1])

        canvas = CompareCanvas()
        canvas.resize(400, 200)
        canvas.show()
        QApplication.processEvents()
        canvas.set_images(qimage(PIL.Image.new("RGB", (1000, 500))), qimage(PIL.Image.new("RGB", (300, 300))))
        canvas.set_mode("flip")  # fit happens while the left view is still half width
        QApplication.processEvents()
        self.assertAlmostEqual(canvas.left.transform().m11(), expected(canvas.left), places=4)
        canvas.set_mode("side")
        QApplication.processEvents()
        canvas.actual_size()  # scrollbars appear
        QApplication.processEvents()
        canvas.fit()  # a one-shot fit against the scrollbar-shrunk viewport would be too small
        QApplication.processEvents()
        for view in (canvas.left, canvas.right):
            self.assertAlmostEqual(view.transform().m11(), expected(view), places=4)
        canvas.resize(600, 350)
        QApplication.processEvents()
        for view in (canvas.left, canvas.right):
            self.assertAlmostEqual(view.transform().m11(), expected(view), places=4)
        self.assertEqual(canvas.zoom_state, "fit")
        canvas.close()
