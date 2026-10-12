import os
import tempfile
import unittest
from unittest import mock

import numpy as np
from click.testing import CliRunner

from picurate.common.images import ImageRecord, Item
from picurate.dedupe import cli as cli_mod
from picurate.dedupe import review as rv
from picurate.dedupe.cli import cli
from picurate.dedupe.keeping import builtin_policies
from picurate.dedupe.pixel_check import PixelVerdict
from picurate.project.project import REVIEW, Project


def rec(path, size=1, w=1, h=1, mtime=0.0):
    return ImageRecord(path, mtime, size, w, h, "PNG")


LARGEST = builtin_policies()["largest"]


def I(path):
    return Item.standalone(path).id


class ReviewDBTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = rv.ReviewDB(os.path.join(self.tmp.name, "x.review.sqlite"))
        self.recs = {p: rec(p, size=s) for p, s in [("a.png", 10), ("b.png", 20), ("c.png", 5), ("d.png", 1), ("e.png", 2)]}

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def groups(self, *specs):
        return [([self.recs[p] for p in members], [(I(a), I(b), d) for a, b, d in edges]) for members, edges in specs]

    def test_upsert_new_predecided_kept_stale(self):
        stats = self.db.upsert_groups(
            self.groups((["a.png", "b.png"], [("a.png", "b.png", 0.05)]), (["c.png", "d.png"], [("c.png", "d.png", 0.4)])),
            LARGEST,
            0.1,
        )
        self.assertEqual((stats.new, stats.predecided, stats.kept, stats.stale), (2, 1, 0, 0))
        g1, g2 = self.db.groups()
        self.assertEqual((g1.status, g1.n, g1.min_distance), ("decided", 2, 0.05))
        self.assertEqual(g2.status, "pending")
        keeps = {m.path: m.keep for m in self.db.members(g1.id)}
        self.assertEqual(keeps, {"a.png": False, "b.png": True})  # largest wins
        self.assertEqual(self.db.edges(g2.id), [(I("c.png"), I("d.png"), 0.4)])
        # the reviewer changes group 2, then a re-match keeps group 2 as is and replaces group 1 by a bigger one
        self.db.keep_only(g2.id, I("d.png"))
        stats = self.db.upsert_groups(
            self.groups(
                (["a.png", "b.png", "e.png"], [("a.png", "b.png", 0.05), ("b.png", "e.png", 0.3)]),
                (["c.png", "d.png"], [("c.png", "d.png", 0.41)]),
            ),
            LARGEST,
            0.1,
        )
        self.assertEqual((stats.new, stats.kept, stats.stale), (1, 1, 1))
        self.assertEqual(self.db.counts()["stale"], 1)
        g2b = self.db.group(g2.id)
        self.assertEqual((g2b.status, g2b.min_distance), ("decided", 0.41))
        self.assertEqual({m.path: m.keep for m in self.db.members(g2.id)}, {"c.png": False, "d.png": True})
        self.assertNotIn(g1.id, [g.id for g in self.db.groups()])
        self.assertEqual(self.db.groups_of(I("b.png")), [g1.id, [g.id for g in self.db.groups("pending")][0]])

    def test_decide_undo_and_status_checks(self):
        self.db.upsert_groups(self.groups((["a.png", "b.png"], [("a.png", "b.png", 0.4)])), LARGEST, 0.1)
        gid = self.db.groups()[0].id
        self.assertFalse(self.db.undo(gid))  # only the initial snapshot
        self.db.decide(gid, keeps={I("a.png"): True}, note="both are fine")
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
            self.db.decide(gid, keeps={I("zzz"): True})

    def test_pixel_results_decide_the_starting_status(self):
        same, diff = PixelVerdict.measured(1.0, 0.5), PixelVerdict.measured(40.0, 0.5)
        pixel = {(I("a.png"), I("b.png")): same, (I("c.png"), I("d.png")): diff}
        # a, b are far apart by embedding but identical by pixels; c, d are close by embedding but edited
        stats = self.db.upsert_groups(
            self.groups((["a.png", "b.png"], [("a.png", "b.png", 0.6)]), (["c.png", "d.png"], [("c.png", "d.png", 0.05)])),
            LARGEST,
            0.1,
            pixel,
        )
        self.assertEqual((stats.new, stats.predecided, stats.pixel_same), (2, 1, 1))
        by_first = {self.db.members(g.id)[0].path: g for g in self.db.groups()}
        self.assertEqual((by_first["a.png"].status, by_first["a.png"].pixel), ("decided", "same"))
        self.assertEqual((by_first["c.png"].status, by_first["c.png"].pixel), ("pending", "different"))
        self.assertEqual(self.db.edge_pixels(by_first["c.png"].id), {(I("c.png"), I("d.png")): diff})
        self.assertEqual(self.db.groups(pixel="same"), [self.db.group(by_first["a.png"].id)])

    def test_untouched_groups_are_rederived_touched_ones_are_not(self):
        specs = ((["a.png", "b.png"], [("a.png", "b.png", 0.6)]), (["c.png", "d.png"], [("c.png", "d.png", 0.6)]))
        self.db.upsert_groups(self.groups(*specs), LARGEST, 0.1)  # no pixel check: both pending
        g_ab, g_cd = self.db.groups()
        self.db.decide(g_cd.id, note="looked at it")
        same = PixelVerdict.measured(1.0, 0.5)
        stats = self.db.upsert_groups(
            self.groups(*specs), LARGEST, 0.1, {(I("a.png"), I("b.png")): same, (I("c.png"), I("d.png")): same}
        )
        self.assertEqual((stats.redecided, stats.reopened), (1, 0))
        self.assertEqual(self.db.group(g_ab.id).status, "decided")
        self.assertEqual(self.db.group(g_cd.id).status, "pending")  # the reviewer's group is left alone
        self.assertFalse(self.db.undo(g_ab.id))  # the history restarted from the new starting point
        # and an untouched decided group whose pixels now differ goes back to review
        stats = self.db.upsert_groups(
            self.groups(*specs),
            LARGEST,
            0.1,
            {(I("a.png"), I("b.png")): PixelVerdict.measured(50.0, 0.5), (I("c.png"), I("d.png")): same},
        )
        self.assertEqual((stats.redecided, stats.reopened), (0, 1))
        self.assertEqual(self.db.group(g_ab.id).status, "pending")

    def test_losers_skip_groups_keeping_nothing(self):
        self.db.upsert_groups(
            self.groups((["a.png", "b.png"], [("a.png", "b.png", 0.05)]), (["c.png", "d.png"], [("c.png", "d.png", 0.05)])),
            LARGEST,
            0.1,
        )
        g1, g2 = self.db.groups()
        self.db.decide(g2.id, keeps={I("c.png"): False, I("d.png"): False})  # unticked after deciding: allowed, but flagged
        self.assertEqual(self.db.losers(), {I("a.png"): (g1.id, [I("b.png")])})
        self.assertEqual([(g.id, g.kept) for g in self.db.groups("decided", keeps_nothing=True)], [(g2.id, 0)])
        with self.assertRaises(ValueError):
            self.db.decide(g2.id, status="decided")  # deciding it again needs a keep or the discard mark

    def test_discard_excludes_every_member_and_undoes(self):
        self.db.upsert_groups(self.groups((["a.png", "b.png"], [("a.png", "b.png", 0.4)])), LARGEST, 0.1)
        gid = self.db.groups()[0].id
        with self.assertRaises(ValueError):
            self.db.decide(gid, keeps={I("b.png"): False}, status="decided")
        self.db.discard(gid)
        g = self.db.group(gid)
        self.assertEqual((g.status, g.discard, g.kept, g.touched), ("decided", True, 0, True))
        self.assertEqual(self.db.groups(keeps_nothing=True), [])
        self.assertEqual(self.db.groups(discarded=True), [g])
        self.assertEqual(self.db.losers(), {I("a.png"): (gid, []), I("b.png"): (gid, [])})
        # a re-match leaves the reviewer's discard alone
        self.db.upsert_groups(self.groups((["a.png", "b.png"], [("a.png", "b.png", 0.4)])), LARGEST, 0.1)
        g2 = self.db.group(gid)
        self.assertEqual((g2.status, g2.discard, g2.kept), ("decided", True, 0))
        self.assertTrue(self.db.undo(gid))
        g = self.db.group(gid)
        self.assertEqual((g.status, g.discard, g.kept), ("pending", False, 1))
        self.db.discard(gid)
        self.db.decide(gid, keeps={I("a.png"): True})  # ticking a keep takes the mark back
        g = self.db.group(gid)
        self.assertEqual((g.status, g.discard, g.kept), ("decided", False, 1))
        with self.assertRaises(ValueError):
            self.db.decide(gid, discard=True)

    def test_new_group_carries_the_keeps_of_a_changed_group(self):
        """A human decision is only ever replaced by another one: when a group the reviewer changed gains a member,
        the new group starts pending with those keeps, and later runs do not re-decide it."""
        self.db.upsert_groups(self.groups((["a.png", "b.png"], [("a.png", "b.png", 0.05)])), LARGEST, 0.1)
        old = self.db.groups()[0]
        self.db.keep_only(old.id, I("a.png"))  # the policy kept b.png (larger); the reviewer prefers a.png
        bigger = self.groups((["a.png", "b.png", "c.png"], [("a.png", "b.png", 0.05), ("b.png", "c.png", 0.05)]))
        stats = self.db.upsert_groups(bigger, LARGEST, 0.1)
        self.assertEqual((stats.new, stats.carried, stats.predecided, stats.stale), (1, 1, 0, 1))
        new = next(g for g in self.db.groups() if g.id != old.id)
        self.assertEqual((new.status, new.carried_from), ("pending", old.id))
        self.assertEqual({m.path: m.keep for m in self.db.members(new.id)}, {"a.png": True, "b.png": False, "c.png": False})
        self.assertEqual(self.db.losers(), {})  # nothing excluded until the reviewer decides the new group
        self.db.upsert_groups(bigger, LARGEST, 0.1)
        self.assertEqual(self.db.group(new.id).status, "pending")


class CliReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        img = os.path.join(self.tmp.name, "img")
        os.makedirs(img)
        for p in ("a.png", "b.png", "c.png", "d.png"):
            open(os.path.join(img, p), "wb").close()
        self.project = Project(os.path.join(self.tmp.name, "project"), create=True)
        self.project.import_images([img])
        ids = {row["path"]: item for item, row in self.project.items.items()}
        source = next(iter(self.project.sources))
        self.records = [
            ImageRecord(p, 0.0, s, 1, 1, "PNG", source, ids[p]) for p, s in [("a.png", 1), ("b.png", 2), ("c.png", 3), ("d.png", 4)]
        ]
        self.emb = np.array([[0.0], [0.05], [0.3], [5.0]], dtype=np.float32)

    def tearDown(self):
        self.project.close()
        self.tmp.cleanup()

    def run_dedupe(self, *extra):
        with (
            mock.patch.object(cli_mod, "load_database", return_value=(self.records, self.emb)),
            mock.patch.object(cli_mod, "EmbeddingDB") as store,
        ):
            store.return_value.__enter__.return_value.model_id = "m"
            return CliRunner().invoke(
                cli, ["dedupe", str(self.project.directory), "--skip-update", "--no-pixel-check", "-c", "cpu", *extra]
            )

    def test_review_threshold_stores_groups_and_moves_nothing(self):
        res = self.run_dedupe("--review-threshold", "0.5")
        self.assertEqual(res.exit_code, 0, res.output)
        with rv.ReviewDB(str(self.project.directory / REVIEW)) as review:
            self.assertEqual(review.counts()["pending"], 1)
            g = review.groups()[0]
            self.assertEqual(g.n, 3)
            self.assertEqual({m.path: m.keep for m in review.members(g.id)}, {"a.png": False, "b.png": False, "c.png": True})
            self.assertEqual(review.get_meta("model_id"), "m")
            self.assertEqual(review.auto_threshold, 0.1)
        res = CliRunner().invoke(cli, ["review-status", str(self.project.directory)])
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("pending  1", res.output)
        self.assertIn("0 images excluded", res.output)
        self.assertEqual(len(os.listdir(os.path.join(self.tmp.name, "img"))), 4)


if __name__ == "__main__":
    unittest.main()
