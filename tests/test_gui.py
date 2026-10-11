"""Offscreen checks of the review GUI. Skipped when PySide6 is not installed."""

import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import PIL.Image

try:
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from picurate.dedupe.gui.canvas import CompareCanvas
    from picurate.dedupe.gui.window import ReviewWindow
except ImportError:  # pragma: no cover
    QApplication = None  # type: ignore[assignment]

from picurate.common.images import ImageRecord
from picurate.dedupe.keeping import builtin_policies
from picurate.dedupe.review import ReviewDB


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
        for path, colour, size in [("a.png", (255, 0, 0), (40, 30)), ("sub/b.png", (250, 0, 0), (20, 15)), ("c.png", (0, 255, 0), (40, 40)), ("d.png", (0, 250, 0), (40, 40))]:
            full = os.path.join(self.img, path)
            PIL.Image.new("RGB", size, colour).save(full)
            st = os.stat(full)
            recs[path] = ImageRecord(path, st.st_mtime, st.st_size, size[0], size[1], "PNG")
        self.review_path = os.path.join(self.tmp.name, "x.review.sqlite")
        with ReviewDB(self.review_path) as db:
            db.begin_session(image_dir=self.img, model_id="m", review_threshold=1.0, auto_threshold=0.1, policy="largest")
            db.upsert_groups(
                [([recs["a.png"], recs["sub/b.png"]], [("a.png", "sub/b.png", 0.3)]), ([recs["c.png"], recs["d.png"]], [("c.png", "d.png", 0.05)])],
                builtin_policies()["largest"],
                0.1,
            )
        self.trash = os.path.join(self.tmp.name, "trash")
        self.win = ReviewWindow(self.review_path, trash_dir=self.trash)
        self.win.show()
        self.app.setActiveWindow(self.win)
        self.win.canvas.left.setFocus()
        self.win.wait_idle()

    def tearDown(self):
        self.win.close()
        self.tmp.cleanup()

    def key(self, key, modifier=Qt.KeyboardModifier.NoModifier):
        QTest.keyClick(self.win.canvas.left, key, modifier)
        self.win.wait_idle()

    def test_groups_listed_surest_first_and_images_loaded(self):
        self.assertEqual([g.min_distance for g in self.win.groups_model.groups], [0.05, 0.3])
        self.assertEqual(self.win.current.status, "decided")  # pre-decided within 0.1
        self.assertEqual(self.win.position_label.text(), "Group 1 / 2")
        self.assertEqual(self.win.canvas.left.image_size, (40, 40))
        self.assertEqual(self.win.member_list.count(), 2)

    def test_navigation_keys(self):
        self.key(Qt.Key.Key_X)
        self.assertEqual(self.win.current.min_distance, 0.3)
        self.assertEqual(self.win.position_label.text(), "Group 2 / 2")
        self.key(Qt.Key.Key_Z)
        self.assertEqual(self.win.current.min_distance, 0.05)
        self.key(Qt.Key.Key_X, Qt.KeyboardModifier.ShiftModifier)  # next pending
        self.assertEqual(self.win.current.status, "pending")

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

    def test_modes_flip_and_diff(self):
        self.key(Qt.Key.Key_X)
        self.key(Qt.Key.Key_D)
        self.assertEqual(self.win.canvas.mode, "flip")
        self.assertEqual(self.win.canvas.left.image_size, (40, 30))
        QTest.keyPress(self.win.canvas.left, Qt.Key.Key_Space)
        self.assertEqual(self.win.canvas.left.image_size, (20, 15))
        QTest.keyRelease(self.win.canvas.left, Qt.Key.Key_Space)
        self.assertEqual(self.win.canvas.left.image_size, (40, 30))
        self.key(Qt.Key.Key_D)
        self.assertEqual(self.win.canvas.mode, "diff")
        self.win.wait_idle()
        self.assertEqual(self.win.canvas.left.image_size, (40, 30))
        self.assertFalse(self.win.canvas.right.isVisible())
        overlay = self.win.canvas.left.pix.pixmap().toImage()
        self.assertGreater(overlay.pixelColor(0, 0).red(), 100)  # the dimmed red original shows through
        QTest.keyPress(self.win.canvas.left, Qt.Key.Key_Space)
        plain = self.win.canvas.left.pix.pixmap().toImage()
        self.assertLess(plain.pixelColor(0, 0).red(), 60)  # on black only the amplified difference (5 * 4) remains
        QTest.keyRelease(self.win.canvas.left, Qt.Key.Key_Space)
        self.assertGreater(self.win.canvas.left.pix.pixmap().toImage().pixelColor(0, 0).red(), 100)
        self.key(Qt.Key.Key_D)
        self.assertEqual(self.win.canvas.mode, "side")

    def test_letters_ignored_in_note_field_and_note_saved(self):
        self.key(Qt.Key.Key_X)
        self.key(Qt.Key.Key_F)
        self.assertTrue(self.win.note.hasFocus())
        QTest.keyClicks(self.win.note, "xz keep both")
        QTest.keyClick(self.win.note, Qt.Key.Key_Return)
        self.win.wait_idle()
        self.assertEqual(self.win.position_label.text(), "Group 2 / 2")  # x/z did not navigate
        with ReviewDB(self.review_path) as db:
            self.assertEqual(db.group(self.win.current.id).note, "xz keep both")
        QTest.keyClick(self.win.note, Qt.Key.Key_Escape)
        self.assertFalse(self.win.note.hasFocus())

    def test_apply_moves_losers(self):
        text, n = self.win.plan_text()
        self.assertEqual(n, 1)
        self.assertIn("c.png", text)  # d.png is the same size; the stable sort keeps the first, c... no: largest wins, tie keeps input order
        moved, refused = self.win.apply_moves()
        self.assertEqual((moved, refused), (1, []))
        self.assertTrue(os.path.exists(os.path.join(self.trash, "c.png")) or os.path.exists(os.path.join(self.trash, "d.png")))
        self.assertEqual(self.win.current.status, "applied")
        self.win.undo_apply()
        self.assertEqual(self.win.current.status, "decided")
        self.assertTrue(os.path.exists(os.path.join(self.img, "c.png")) and os.path.exists(os.path.join(self.img, "d.png")))

    def test_b_defaults_to_nearest_of_a(self):
        with ReviewDB(self.review_path) as db:
            recs = {m.path: m.record for g in db.groups() for m in db.members(g.id)}
            db.upsert_groups(
                [([recs["a.png"], recs["sub/b.png"], recs["c.png"]], [("a.png", "sub/b.png", 0.9), ("sub/b.png", "c.png", 0.2)])],
                builtin_policies()["largest"], 0.1,
            )
        self.win.reload_groups()
        self.win.wait_idle()
        self.assertEqual(self.win.current.n, 3)
        self.assertEqual([m.path for m in self.win.members], ["a.png", "c.png", "sub/b.png"])
        self.assertEqual(self.win.members[self.win.a].path, "c.png")  # largest pixel count... size: a.png 40x30 png vs c 40x40
        self.assertEqual(self.win.members[self.win.b].path, "sub/b.png")  # nearest to c.png (0.2), not member 1

    def test_checkbox_click_does_not_rebuild_list(self):
        self.key(Qt.Key.Key_X)
        item = self.win.member_list.item(1)
        item.setCheckState(Qt.CheckState.Checked)  # emits itemChanged like a click on the box
        self.win._member_clicked(item)  # the click that follows must still find the same item
        self.win._member_clicked(None)  # and a stray click on no item must not raise
        self.assertIs(self.win.member_list.item(1), item)
        with ReviewDB(self.review_path) as db:
            self.assertEqual({m.path: m.keep for m in db.members(self.win.current.id)}, {"a.png": True, "sub/b.png": True})

    def test_zoom_buttons_reflect_state(self):
        self.assertTrue(self.win.zoom_actions["fit"].isChecked())
        self.key(Qt.Key.Key_0)
        self.assertTrue(self.win.zoom_actions["actual"].isChecked())
        self.assertFalse(self.win.zoom_actions["fit"].isChecked())
        self.win.canvas.zoom(1.5)
        self.assertFalse(self.win.zoom_actions["actual"].isChecked() or self.win.zoom_actions["fit"].isChecked())
        self.key(Qt.Key.Key_Home)
        self.assertTrue(self.win.zoom_actions["fit"].isChecked())
        self.win.zoom_actions["fit"].trigger()  # clicking the depressed button must not release it
        self.assertTrue(self.win.zoom_actions["fit"].isChecked())
        self.win.zoom_actions["actual"].trigger()
        self.win.zoom_actions["actual"].trigger()
        self.assertTrue(self.win.zoom_actions["actual"].isChecked())
        self.assertEqual(self.win.canvas.zoom_state, "actual")

    def test_prefetches_in_the_direction_of_travel(self):
        # Startup shows group 1 (c/d) and the default direction is forward: group 2 (a/b) is loaded ahead of time.
        self.assertIn("c.png", self.win.images)
        self.assertIn("a.png", self.win.images)
        self.assertIn("sub/b.png", self.win.images)
        self.assertEqual(self.win.loading, {})
        self.win.images.clear()
        self.key(Qt.Key.Key_X)  # now on the last group, moving forward: nothing ahead to prefetch
        self.assertEqual(sorted(self.win.images._d), ["a.png", "sub/b.png"])
        self.key(Qt.Key.Key_Z)  # turned around: the group behind (none, we are first) and the current one only
        self.assertEqual(self.win.direction, -1)
        self.assertIn("c.png", self.win.images)
        self.assertEqual(self.win.canvas.left.image_size, (40, 40))

    def test_viewing_a_group_writes_no_history(self):
        self.key(Qt.Key.Key_X)
        self.key(Qt.Key.Key_Z)
        with ReviewDB(self.review_path) as db:
            self.assertEqual(db.conn.execute("SELECT COUNT(*) FROM history").fetchone()[0], 2)  # one creation snapshot per group

    def test_policy_pick_resets_all_pending_groups(self):
        pending = next(g for g in self.win.groups_model.groups if g.status == "pending")  # a.png (40x30) / sub/b.png (20x15)
        with ReviewDB(self.review_path) as db:
            db.decide(pending.id, keeps={"a.png": False, "sub/b.png": True})
        self.win.reload_groups(select_id=pending.id)
        self.win.policy_box.setCurrentText("largest")
        self.win.reset_pending_to_policy(confirm=False)
        self.win.wait_idle()
        with ReviewDB(self.review_path) as db:
            self.assertEqual({m.path: m.keep for m in db.members(pending.id)}, {"a.png": True, "sub/b.png": False})
            self.assertEqual(db.group(pending.id).status, "pending")  # keeps only; still to be reviewed
            self.assertEqual(db.get_meta("policy"), "largest")
        self.key(Qt.Key.Key_U)
        with ReviewDB(self.review_path) as db:
            self.assertEqual({m.path: m.keep for m in db.members(pending.id)}, {"a.png": False, "sub/b.png": True})

    def test_diff_is_prefetched_and_reused(self):
        self.win.canvas.set_mode("diff")
        self.win.wait_idle()
        self.assertEqual(sorted(self.win.diffs._d), [("a.png", "sub/b.png"), ("d.png", "c.png")])  # current and the group ahead
        import picurate.dedupe.gui.window as w

        calls = []
        original = w.diff_images
        w.diff_images = lambda a, b: calls.append(1) or original(a, b)
        try:
            self.key(Qt.Key.Key_X)
            self.assertIsNotNone(self.win.canvas.diff)
            self.assertEqual(self.win.canvas.left.image_size, (40, 30))
        finally:
            w.diff_images = original
        self.assertEqual(calls, [])  # served from the cache

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
        self.win.wait_idle()
        self.assertEqual([g.id for g in self.win.groups_model.groups], [gid])
        self.key(Qt.Key.Key_R)
        with ReviewDB(self.review_path) as db:
            g = db.group(gid)
            self.assertEqual((g.status, g.discard), ("decided", True))
        self.assertTrue(self.win.groups_model.data(self.win.groups_model.index(0)).startswith("✕ "))
        text, n = self.win.plan_text()
        self.assertIn("sub/b.png  →  trash   (whole group discarded)", text)
        self.assertEqual(n, 3)
        self.key(Qt.Key.Key_R)  # takes it back
        with ReviewDB(self.review_path) as db:
            g = db.group(gid)
            self.assertEqual((g.status, g.discard), ("pending", False))

    def test_filter(self):
        self.win.filter.setCurrentText("pending")
        self.win.wait_idle()
        self.assertEqual(self.win.groups_model.rowCount(), 1)
        self.assertEqual(self.win.current.status, "pending")


@unittest.skipIf(QApplication is None, "PySide6 not installed")
class CanvasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_each_view_fits_its_own_image_and_zoom_multiplies_both(self):
        from picurate.dedupe.gui.jobs import qimage

        canvas = CompareCanvas()
        canvas.resize(400, 200)
        canvas.show()
        QApplication.processEvents()
        canvas.set_images(qimage(PIL.Image.new("RGB", (1000, 500))), qimage(PIL.Image.new("RGB", (50, 50))))
        left, right = canvas.left.transform().m11(), canvas.right.transform().m11()
        self.assertGreater(right, left * 5)  # the small image is not shrunk to the big one's scale
        canvas.zoom(2.0)
        self.assertAlmostEqual(canvas.left.transform().m11(), left * 2)
        self.assertAlmostEqual(canvas.right.transform().m11(), right * 2)
        canvas.actual_size()
        self.assertEqual(canvas.scale_factor(), 1.0)
        canvas.close()

    def test_fit_ignores_the_hidden_view(self):
        from picurate.dedupe.gui.jobs import qimage

        canvas = CompareCanvas()
        canvas.resize(400, 200)
        canvas.show()
        QApplication.processEvents()
        canvas.set_images(qimage(PIL.Image.new("RGB", (100, 50))), qimage(PIL.Image.new("RGB", (100, 50))))
        side = canvas.scale_factor()
        canvas.set_mode("flip")
        QApplication.processEvents()
        canvas.fit()
        self.assertGreater(canvas.scale_factor(), side * 1.5)  # one view has the whole width now
        canvas.set_mode("side")
        QApplication.processEvents()
        canvas.zoom(4.0)
        canvas.left.horizontalScrollBar().setValue(canvas.left.horizontalScrollBar().maximum())
        self.assertEqual(canvas.right.horizontalScrollBar().value(), canvas.right.horizontalScrollBar().maximum())  # linked by fraction
        canvas.zoom(1.5)
        canvas.resize(500, 300)
        QApplication.processEvents()
        self.assertFalse(canvas._auto_fit)  # a manual zoom survives resizes
        canvas.close()

    def test_fit_follows_the_final_viewport(self):
        """A fit asked for before the layout has settled (mode switch, resize, 100% with scrollbars) must end up
        matching the viewport the view actually gets, not the stale one it had at the time."""
        from picurate.dedupe.gui.jobs import qimage

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

    def test_diff_arriving_later_is_fitted(self):
        """The diff is computed off-thread and swapped in after the group was shown; in fit mode that swap must
        not bring back the scale of the previous picture."""
        from picurate.dedupe.gui.jobs import qimage

        canvas = CompareCanvas()
        canvas.resize(400, 200)
        canvas.show()
        QApplication.processEvents()
        small = qimage(PIL.Image.new("RGB", (100, 50)))
        canvas.set_images(small, small)
        canvas.set_mode("diff")
        canvas.set_diff(small, small)
        QApplication.processEvents()
        first = canvas.scale_factor()
        big = qimage(PIL.Image.new("RGB", (2000, 1000)))
        canvas.set_images(big, big)  # next group: the diff is not there yet
        canvas.set_diff(big, big)
        QApplication.processEvents()
        self.assertAlmostEqual(canvas.scale_factor(), first / 20, places=4)
        canvas.zoom(2.0)  # a manual zoom is kept across the Space swap
        zoomed = canvas.scale_factor()
        canvas.set_flipped(True)
        self.assertEqual(canvas.scale_factor(), zoomed)
        canvas.close()
