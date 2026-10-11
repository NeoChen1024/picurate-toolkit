import os
import tempfile
import unittest
from unittest import mock

import numpy as np
from click.testing import CliRunner

from picurate.dedupe import cli as cli_mod
from picurate.dedupe import review as rv
from picurate.dedupe.cli import cli
from picurate.common.images import ImageRecord
from picurate.dedupe.dedupe import group_edges
from picurate.dedupe.keeping import builtin_policies
from picurate.dedupe.pixel_check import PixelVerdict


def rec(path, size=1, w=1, h=1, mtime=0.0):
    return ImageRecord(path, mtime, size, w, h, "PNG")


LARGEST = builtin_policies()["largest"]


class GroupEdgesTests(unittest.TestCase):
    def test_groups_keep_their_edges_and_are_sorted(self):
        groups = group_edges(6, [(4, 5, 0.3), (0, 1, 0.1), (1, 2, 0.2)])
        self.assertEqual([g.members for g in groups], [[0, 1, 2], [4, 5]])
        self.assertEqual(groups[0].edges, [(0, 1, 0.1), (1, 2, 0.2)])
        self.assertEqual((groups[0].min_distance, groups[0].max_distance), (0.1, 0.2))


class ReviewDBTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = rv.ReviewDB(os.path.join(self.tmp.name, "x.review.sqlite"))
        self.recs = {p: rec(p, size=s) for p, s in [("a.png", 10), ("b.png", 20), ("c.png", 5), ("d.png", 1), ("e.png", 2)]}

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def groups(self, *specs):
        return [([self.recs[p] for p in members], edges) for members, edges in specs]

    def test_default_path(self):
        self.assertEqual(rv.default_review_path("/x/pictures.sqlite"), "/x/pictures.review.sqlite")

    def test_upsert_new_predecided_kept_stale(self):
        stats = self.db.upsert_groups(self.groups((["a.png", "b.png"], [("a.png", "b.png", 0.05)]), (["c.png", "d.png"], [("c.png", "d.png", 0.4)])), LARGEST, 0.1)
        self.assertEqual((stats.new, stats.predecided, stats.kept, stats.stale), (2, 1, 0, 0))
        g1, g2 = self.db.groups()
        self.assertEqual((g1.status, g1.n, g1.min_distance), ("decided", 2, 0.05))
        self.assertEqual(g2.status, "pending")
        keeps = {m.path: m.keep for m in self.db.members(g1.id)}
        self.assertEqual(keeps, {"a.png": False, "b.png": True})  # largest wins
        self.assertEqual(self.db.edges(g2.id), [("c.png", "d.png", 0.4)])
        # the reviewer changes group 2, then a re-match keeps group 2 as is and replaces group 1 by a bigger one
        self.db.keep_only(g2.id, "d.png")
        stats = self.db.upsert_groups(
            self.groups((["a.png", "b.png", "e.png"], [("a.png", "b.png", 0.05), ("b.png", "e.png", 0.3)]), (["c.png", "d.png"], [("c.png", "d.png", 0.41)])), LARGEST, 0.1
        )
        self.assertEqual((stats.new, stats.kept, stats.stale), (1, 1, 1))
        self.assertEqual(self.db.counts()["stale"], 1)
        g2b = self.db.group(g2.id)
        self.assertEqual((g2b.status, g2b.min_distance), ("decided", 0.41))
        self.assertEqual({m.path: m.keep for m in self.db.members(g2.id)}, {"c.png": False, "d.png": True})
        self.assertNotIn(g1.id, [g.id for g in self.db.groups()])
        self.assertEqual(self.db.groups_of("b.png"), [g1.id, [g.id for g in self.db.groups("pending")][0]])

    def test_decide_undo_and_status_checks(self):
        self.db.upsert_groups(self.groups((["a.png", "b.png"], [("a.png", "b.png", 0.4)])), LARGEST, 0.1)
        gid = self.db.groups()[0].id
        self.assertFalse(self.db.undo(gid))  # only the initial snapshot
        self.db.decide(gid, keeps={"a.png": True}, note="both are fine")
        self.db.decide(gid, status="skipped")
        self.assertEqual(self.db.group(gid).status, "skipped")
        self.assertTrue(self.db.undo(gid))
        g = self.db.group(gid)
        self.assertEqual((g.status, g.note), ("pending", "both are fine"))
        self.assertEqual({m.path: m.keep for m in self.db.members(gid)}, {"a.png": True, "b.png": True})
        self.assertTrue(self.db.undo(gid))
        self.assertEqual({m.path: m.keep for m in self.db.members(gid)}, {"a.png": False, "b.png": True})
        self.assertFalse(self.db.undo(gid))
        with self.assertRaises(ValueError):
            self.db.decide(gid, status="bogus")
        with self.assertRaises(KeyError):
            self.db.decide(gid, keeps={"zzz": True})
        self.db.reset_to_policy(gid, builtin_policies()["smallest"]) if "smallest" in builtin_policies() else None

    def test_pixel_results_decide_the_starting_status(self):
        same, diff = PixelVerdict.measured(1.0, 0.5), PixelVerdict.measured(40.0, 0.5)
        pixel = {("a.png", "b.png"): same, ("c.png", "d.png"): diff}
        # a, b are far apart by embedding but identical by pixels; c, d are close by embedding but edited
        stats = self.db.upsert_groups(self.groups((["a.png", "b.png"], [("a.png", "b.png", 0.6)]), (["c.png", "d.png"], [("c.png", "d.png", 0.05)])), LARGEST, 0.1, pixel)
        self.assertEqual((stats.new, stats.predecided, stats.pixel_same), (2, 1, 1))
        by_first = {self.db.members(g.id)[0].path: g for g in self.db.groups()}
        self.assertEqual((by_first["a.png"].status, by_first["a.png"].pixel), ("decided", "same"))
        self.assertEqual((by_first["c.png"].status, by_first["c.png"].pixel), ("pending", "different"))
        self.assertEqual(self.db.edge_pixels(by_first["c.png"].id), {("c.png", "d.png"): diff})
        self.assertEqual(self.db.groups(pixel="same"), [self.db.group(by_first["a.png"].id)])

    def test_untouched_groups_are_rederived_touched_ones_are_not(self):
        specs = ((["a.png", "b.png"], [("a.png", "b.png", 0.6)]), (["c.png", "d.png"], [("c.png", "d.png", 0.6)]))
        self.db.upsert_groups(self.groups(*specs), LARGEST, 0.1)  # no pixel check: both pending
        g_ab, g_cd = self.db.groups()
        self.db.decide(g_cd.id, note="looked at it")
        same = PixelVerdict.measured(1.0, 0.5)
        stats = self.db.upsert_groups(self.groups(*specs), LARGEST, 0.1, {("a.png", "b.png"): same, ("c.png", "d.png"): same})
        self.assertEqual((stats.redecided, stats.reopened), (1, 0))
        self.assertEqual(self.db.group(g_ab.id).status, "decided")
        self.assertEqual(self.db.group(g_cd.id).status, "pending")  # the reviewer's group is left alone
        self.assertFalse(self.db.undo(g_ab.id))  # the history restarted from the new starting point
        # and an untouched decided group whose pixels now differ goes back to review
        stats = self.db.upsert_groups(self.groups(*specs), LARGEST, 0.1, {("a.png", "b.png"): PixelVerdict.measured(50.0, 0.5), ("c.png", "d.png"): same})
        self.assertEqual((stats.redecided, stats.reopened), (0, 1))
        self.assertEqual(self.db.group(g_ab.id).status, "pending")

    def test_no_op_decide_records_nothing_and_undo_untouches(self):
        self.db.upsert_groups(self.groups((["a.png", "b.png"], [("a.png", "b.png", 0.4)])), LARGEST, 0.1)
        gid = self.db.groups()[0].id
        self.db.decide(gid, keeps={"b.png": True}, status="pending", note="")
        self.assertFalse(self.db.group(gid).touched)
        self.assertEqual(self.db.conn.execute("SELECT COUNT(*) FROM history WHERE group_id = ?", (gid,)).fetchone()[0], 1)
        self.db.decide(gid, status="skipped")
        self.assertTrue(self.db.group(gid).touched)
        self.db.undo(gid)
        self.assertFalse(self.db.group(gid).touched)
        self.db.reset_pending_to_policy(builtin_policies()["newest"])
        self.assertFalse(self.db.group(gid).touched)  # a bulk Policy pick is not a review

    def test_pixel_cache_carries_file_state(self):
        self.recs["a.png"] = rec("a.png", size=10, mtime=5.0)
        v = PixelVerdict.measured(2.0, 0.5)
        self.db.upsert_groups(self.groups((["a.png", "b.png"], [("a.png", "b.png", 0.4)])), LARGEST, 0.1, {("a.png", "b.png"): v})
        self.assertEqual(self.db.pixel_cache(), {("a.png", "b.png"): (v, (5.0, 10, 0.0, 20))})

    def test_plan_apply_skips_groups_keeping_nothing(self):
        self.db.upsert_groups(self.groups((["a.png", "b.png"], [("a.png", "b.png", 0.05)]), (["c.png", "d.png"], [("c.png", "d.png", 0.05)])), LARGEST, 0.1)
        g1, g2 = self.db.groups()
        self.db.decide(g2.id, keeps={"c.png": False, "d.png": False})  # unticked after deciding: allowed, but flagged
        moves = self.db.plan_apply()
        self.assertEqual([(m.group_id, m.path, m.keep) for m in moves], [(g1.id, "a.png", ("b.png",))])
        self.assertEqual([(g.id, g.kept) for g in self.db.groups("decided", keeps_nothing=True)], [(g2.id, 0)])
        with self.assertRaises(ValueError):
            self.db.decide(g2.id, status="decided")  # deciding it again needs a keep or the discard mark

    def test_discard_moves_every_member_and_undoes(self):
        self.db.upsert_groups(self.groups((["a.png", "b.png"], [("a.png", "b.png", 0.4)])), LARGEST, 0.1)
        gid = self.db.groups()[0].id
        with self.assertRaises(ValueError):
            self.db.decide(gid, keeps={"b.png": False}, status="decided")
        self.db.discard(gid)
        g = self.db.group(gid)
        self.assertEqual((g.status, g.discard, g.kept, g.touched), ("decided", True, 0, True))
        self.assertEqual(self.db.groups(keeps_nothing=True), [])
        self.assertEqual(self.db.groups(discarded=True), [g])
        self.assertEqual(sorted((m.path, m.keep) for m in self.db.plan_apply()), [("a.png", ()), ("b.png", ())])
        # a re-match leaves the reviewer's discard alone
        self.db.upsert_groups(self.groups((["a.png", "b.png"], [("a.png", "b.png", 0.4)])), LARGEST, 0.1)
        g2 = self.db.group(gid)
        self.assertEqual((g2.status, g2.discard, g2.kept), ("decided", True, 0))
        self.assertTrue(self.db.undo(gid))
        g = self.db.group(gid)
        self.assertEqual((g.status, g.discard, g.kept), ("pending", False, 1))
        self.db.discard(gid)
        self.db.decide(gid, keeps={"a.png": True})  # ticking a keep takes the mark back
        g = self.db.group(gid)
        self.assertEqual((g.status, g.discard, g.kept), ("decided", False, 1))
        with self.assertRaises(ValueError):
            self.db.decide(gid, discard=True)


class ApplyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.img = os.path.join(self.tmp.name, "img")
        os.makedirs(os.path.join(self.img, "sub"))
        self.recs = []
        for p, data in [("a.png", b"aaaa"), ("sub/b.png", b"bb"), ("c.png", b"cccccc"), ("d.png", b"d")]:
            full = os.path.join(self.img, p)
            with open(full, "wb") as f:
                f.write(data)
            st = os.stat(full)
            self.recs.append(ImageRecord(p, st.st_mtime, st.st_size, 1, 1, "PNG"))
        self.db = rv.ReviewDB(os.path.join(self.tmp.name, "x.review.sqlite"))
        self.trash = os.path.join(self.tmp.name, "trash")

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_apply_moves_losers_refuses_changed_and_unapplies(self):
        r = {x.path: x for x in self.recs}
        self.db.upsert_groups(
            [([r["a.png"], r["sub/b.png"]], [("a.png", "sub/b.png", 0.05)]), ([r["c.png"], r["d.png"]], [("c.png", "d.png", 0.05)])], LARGEST, 0.1
        )
        g1, g2 = self.db.groups()
        with open(os.path.join(self.img, "d.png"), "ab") as f:
            f.write(b"changed")
        moved, refused = rv.apply_decisions(self.db, self.img, self.trash, dry_run=True)
        self.assertEqual((moved, refused), (1, [g2.id]))
        self.assertTrue(os.path.exists(os.path.join(self.img, "sub", "b.png")))
        moved, refused = rv.apply_decisions(self.db, self.img, self.trash, dry_run=False)
        self.assertEqual((moved, refused), (1, [g2.id]))
        self.assertFalse(os.path.exists(os.path.join(self.img, "sub", "b.png")))
        self.assertTrue(os.path.exists(os.path.join(self.trash, "sub", "b.png")))
        self.assertEqual(self.db.group(g1.id).status, "applied")
        self.assertEqual(self.db.group(g2.id).status, "decided")
        with self.assertRaises(ValueError):
            self.db.decide(g1.id, status="pending")
        self.assertEqual(rv.unapply_group(self.db, self.img, g1.id), 1)
        self.assertTrue(os.path.exists(os.path.join(self.img, "sub", "b.png")))
        self.assertEqual(self.db.group(g1.id).status, "decided")
        self.assertEqual(self.db.applied(), [])


class MigrationTests(unittest.TestCase):
    def test_schema_1_file_gains_columns_and_touched_from_history(self):
        import sqlite3

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "old.review.sqlite")
            con = sqlite3.connect(path)
            con.executescript(
                """CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                   INSERT INTO meta VALUES ('schema_version', '1');
                   CREATE TABLE groups (id INTEGER PRIMARY KEY, key TEXT NOT NULL UNIQUE, n INTEGER NOT NULL, min_distance REAL NOT NULL,
                       max_distance REAL NOT NULL, status TEXT NOT NULL, note TEXT NOT NULL DEFAULT '', updated_at REAL NOT NULL);
                   CREATE TABLE edges (group_id INTEGER NOT NULL, path_a TEXT NOT NULL, path_b TEXT NOT NULL, distance REAL NOT NULL);
                   CREATE TABLE history (id INTEGER PRIMARY KEY, group_id INTEGER NOT NULL, at REAL NOT NULL, keep_json TEXT NOT NULL,
                       status TEXT NOT NULL, note TEXT NOT NULL);
                   INSERT INTO meta VALUES ('policy', 'largest');
                   CREATE TABLE members (group_id INTEGER NOT NULL, path TEXT NOT NULL, keep INTEGER NOT NULL, width INTEGER NOT NULL,
                       height INTEGER NOT NULL, size INTEGER NOT NULL, format TEXT NOT NULL, mtime REAL NOT NULL, distance_to_winner REAL,
                       PRIMARY KEY (group_id, path));
                   INSERT INTO groups VALUES (1, 'k1', 2, 0.4, 0.4, 'pending', '', 0), (2, 'k2', 2, 0.4, 0.4, 'decided', '', 0),
                       (3, 'k3', 2, 0.4, 0.4, 'pending', '', 0), (4, 'k4', 2, 0.4, 0.4, 'pending', '', 0);
                   INSERT INTO members VALUES (3, 'a', 0, 1, 1, 1, 'PNG', 0, NULL), (3, 'b', 1, 1, 1, 9, 'PNG', 0, NULL),
                       (4, 'a', 1, 1, 1, 1, 'PNG', 0, NULL), (4, 'b', 0, 1, 1, 9, 'PNG', 0, NULL);
                   INSERT INTO history VALUES (1, 1, 0, '{"a": true}', 'pending', ''), (2, 1, 1, '{"a": true}', 'pending', ''),
                       (3, 2, 0, '{"a": true}', 'pending', ''), (4, 2, 1, '{"a": true}', 'decided', ''),
                       (5, 3, 0, '{"a": true, "b": false}', 'pending', ''), (6, 3, 1, '{"a": false, "b": true}', 'pending', ''),
                       (7, 4, 0, '{"a": false, "b": true}', 'pending', ''), (8, 4, 1, '{"a": true, "b": false}', 'pending', '');"""
            )
            con.commit()
            con.close()
            with rv.ReviewDB(path) as db:
                self.assertEqual(db.get_meta("schema_version"), rv.SCHEMA_VERSION)
                # 1: no-op snapshots; 2: decided by hand; 3: keeps moved to what "largest" picks (a bulk Policy
                # pick); 4: keeps moved away from the policy's choice by hand
                self.assertEqual([(g.id, g.touched, g.pixel) for g in db.groups()], [(1, False, None), (2, True, None), (3, False, None), (4, True, None)])
                self.assertFalse(any(g.discard for g in db.groups()))
                self.assertTrue(db.undo(4))  # history rows from before the discard column restore as not discarded
                self.assertFalse(db.group(4).discard)


class CliReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.img = os.path.join(self.tmp.name, "img")
        os.makedirs(self.img)
        self.db = os.path.join(self.tmp.name, "pictures.sqlite")
        self.records = [ImageRecord(p, 0.0, s, 1, 1, "PNG") for p, s in [("a.png", 1), ("b.png", 2), ("c.png", 3), ("d.png", 4)]]
        self.emb = np.array([[0.0], [0.05], [0.3], [5.0]], dtype=np.float32)

    def tearDown(self):
        self.tmp.cleanup()

    def run_dedupe(self, *extra):
        with mock.patch.object(cli_mod, "load_database", return_value=(self.records, self.emb)), mock.patch.object(cli_mod, "EmbeddingDB") as store:
            store.return_value.__enter__.return_value.model_id = "m"
            return CliRunner().invoke(cli, ["dedupe", "-i", self.img, "-d", self.db, "--skip-update", "-c", "cpu", *extra])

    def test_review_threshold_stores_groups_and_moves_nothing(self):
        res = self.run_dedupe("--review-threshold", "0.5")
        self.assertEqual(res.exit_code, 0, res.output)
        path = rv.default_review_path(self.db)
        with rv.ReviewDB(path) as review:
            self.assertEqual(review.counts()["pending"], 1)
            g = review.groups()[0]
            self.assertEqual(g.n, 3)
            self.assertEqual({m.path: m.keep for m in review.members(g.id)}, {"a.png": False, "b.png": False, "c.png": True})
            self.assertEqual(review.get_meta("model_id"), "m")
            self.assertEqual(review.auto_threshold, 0.1)
        res = CliRunner().invoke(cli, ["review-status", "-d", self.db])
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("pending  1", res.output)
        self.assertIn("0 files would be moved", res.output)

    def test_review_threshold_rejects_trash_dir_and_lower_value(self):
        self.assertNotEqual(self.run_dedupe("--review-threshold", "0.5", "--trash-dir", self.tmp.name).exit_code, 0)
        self.assertNotEqual(self.run_dedupe("--review-threshold", "0.05").exit_code, 0)

    def test_review_apply_requires_database(self):
        res = CliRunner().invoke(cli, ["review-apply", "-i", self.img, "-d", self.db, "--trash-dir", os.path.join(self.tmp.name, "t")])
        self.assertNotEqual(res.exit_code, 0)
        self.assertIn("No review database", res.output)
