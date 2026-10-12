"""Review sessions: duplicate groups found at a loose threshold, the reviewer's decisions, and applying them.

Everything lives in a SQLite file of its own next to the embedding database (``pictures.review.sqlite`` for
``pictures.sqlite``), so a ``--force-update`` rebuild of the embeddings never touches decisions, and a review of a
large library (tens of thousands of groups) is indexed rather than parsed. No Qt in here; the GUI and the CLI both
drive this module.

Group identity is the hash of its sorted member paths. Re-matching upserts by that key: an unchanged group keeps
its row, decision and history; a group whose membership changed is a new group, and the old row is marked
``stale``. A group nobody has changed yet (``touched`` is 0) gets its starting status re-derived on every run, so a
new rule (the pixel check, another auto threshold) reaches the groups still waiting for review.

Starting status: with the pixel check, ``decided`` when every edge differs only by compression, else ``pending``;
without it, ``decided`` when every edge is within the auto threshold.

A decided group must keep at least one member, unless the reviewer marked it ``discard``: then applying it moves every
member. Unticking every keep is not enough on its own, since that is as often a slip as an intent.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from ..common.images import ImageRecord
from .dedupe import DuplicateGroup, move_to_trash
from .keeping import Policy, load_policies
from .pixel_check import SAME, PixelVerdict

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "3"
STATUSES = ("pending", "decided", "skipped", "applied", "stale")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS groups (
    id           INTEGER PRIMARY KEY,
    key          TEXT NOT NULL UNIQUE,
    n            INTEGER NOT NULL,
    min_distance REAL NOT NULL,
    max_distance REAL NOT NULL,
    status       TEXT NOT NULL,
    note         TEXT NOT NULL DEFAULT '',
    updated_at   REAL NOT NULL,
    pixel        TEXT,              -- 'same' when every edge differs only by compression, 'different', NULL: unchecked
    touched      INTEGER NOT NULL DEFAULT 0, -- changed by the reviewer since it was created
    discard      INTEGER NOT NULL DEFAULT 0  -- the reviewer chose to move every member, keeping none
);
CREATE INDEX IF NOT EXISTS groups_status ON groups (status, min_distance);
CREATE TABLE IF NOT EXISTS members (
    group_id           INTEGER NOT NULL REFERENCES groups (id) ON DELETE CASCADE,
    path               TEXT NOT NULL,
    keep               INTEGER NOT NULL,
    width              INTEGER NOT NULL,
    height             INTEGER NOT NULL,
    size               INTEGER NOT NULL,
    format             TEXT NOT NULL,
    mtime              REAL NOT NULL,
    distance_to_winner REAL,
    PRIMARY KEY (group_id, path)
);
CREATE INDEX IF NOT EXISTS members_path ON members (path);
CREATE TABLE IF NOT EXISTS edges (
    group_id INTEGER NOT NULL REFERENCES groups (id) ON DELETE CASCADE,
    path_a   TEXT NOT NULL,
    path_b   TEXT NOT NULL,
    distance REAL NOT NULL,
    pixel    TEXT,                  -- pixel_check verdict: same, different, aspect, error; NULL: unchecked
    pixel_peak REAL,
    pixel_mean REAL
);
CREATE INDEX IF NOT EXISTS edges_group ON edges (group_id);
CREATE TABLE IF NOT EXISTS history (
    id        INTEGER PRIMARY KEY,
    group_id  INTEGER NOT NULL REFERENCES groups (id) ON DELETE CASCADE,
    at        REAL NOT NULL,
    keep_json TEXT NOT NULL,
    status    TEXT NOT NULL,
    note      TEXT NOT NULL,
    discard   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS history_group ON history (group_id, id);
CREATE TABLE IF NOT EXISTS applied (
    path       TEXT PRIMARY KEY,
    trash_path TEXT NOT NULL,
    group_id   INTEGER NOT NULL REFERENCES groups (id) ON DELETE CASCADE,
    at         REAL NOT NULL
);
"""


def default_review_path(db_path: str) -> str:
    """``pictures.sqlite`` -> ``pictures.review.sqlite``."""
    root, _ = os.path.splitext(db_path)
    return root + ".review.sqlite"


def group_key(paths: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(paths)).encode("utf-8")).hexdigest()


@dataclass(slots=True, frozen=True)
class GroupRow:
    id: int
    key: str
    n: int
    min_distance: float
    max_distance: float
    status: str
    note: str
    updated_at: float
    pixel: str | None = None
    touched: bool = False
    discard: bool = False
    kept: int | None = None  # members kept; None when not read


@dataclass(slots=True, frozen=True)
class MemberRow:
    path: str
    keep: bool
    width: int
    height: int
    size: int
    format: str
    mtime: float
    distance_to_winner: float | None

    @property
    def record(self) -> ImageRecord:
        return ImageRecord(self.path, self.mtime, self.size, self.width, self.height, self.format)


@dataclass(slots=True)
class UpsertStats:
    new: int = 0
    kept: int = 0
    stale: int = 0
    predecided: int = 0  # new groups that start decided
    redecided: int = 0  # untouched groups that went from pending to decided under the current rule
    reopened: int = 0  # untouched groups that went from decided back to pending
    pixel_same: int = 0  # groups of this run whose every edge differs only by compression


def initial_status(max_distance: float, pixel: str | None, auto_threshold: float) -> str:
    if pixel is not None:
        return "decided" if pixel == SAME else "pending"
    return "decided" if max_distance <= auto_threshold else "pending"


EdgeKey = tuple[str, str]


@dataclass(slots=True, frozen=True)
class Move:
    group_id: int
    path: str
    keep: tuple[str, ...]  # empty for a discarded group


class ReviewDB:
    """The review session file. Every decision is committed as it is made."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(_SCHEMA)
        self._migrate()
        if self.get_meta("schema_version") is None:
            self.set_meta("schema_version", SCHEMA_VERSION)
        self.conn.commit()

    def _migrate(self) -> None:
        """Add the columns of schemas 2 and 3 to an older file. ``touched`` is derived from the history: a group was
        touched when its status or note ever left the first snapshot's, or when its keeps differ from both the first
        snapshot and the recorded policy's choice (a bulk Policy pick is not a review decision)."""

        def columns(table: str) -> set[str]:
            return {r["name"] for r in self.conn.execute(f"PRAGMA table_info({table})")}

        edge_cols = columns("edges")
        for name, kind in (("pixel", "TEXT"), ("pixel_peak", "REAL"), ("pixel_mean", "REAL")):
            if name not in edge_cols:
                self.conn.execute(f"ALTER TABLE edges ADD COLUMN {name} {kind}")
        group_cols = columns("groups")
        if "pixel" not in group_cols:
            self.conn.execute("ALTER TABLE groups ADD COLUMN pixel TEXT")
        if "touched" not in group_cols:
            self.conn.execute("ALTER TABLE groups ADD COLUMN touched INTEGER NOT NULL DEFAULT 0")
            self._derive_touched()
            logger.info(
                "Review database %s upgraded: %d groups were changed by a reviewer",
                self.path,
                self.conn.execute("SELECT COUNT(*) FROM groups WHERE touched").fetchone()[0],
            )
        if "discard" not in group_cols:
            self.conn.execute("ALTER TABLE groups ADD COLUMN discard INTEGER NOT NULL DEFAULT 0")
        if "discard" not in columns("history"):
            self.conn.execute("ALTER TABLE history ADD COLUMN discard INTEGER NOT NULL DEFAULT 0")
        if self.get_meta("schema_version") not in (None, SCHEMA_VERSION):
            self.set_meta("schema_version", SCHEMA_VERSION)

    def _derive_touched(self) -> None:
        policy = load_policies().get(self.get_meta("policy") or "")
        first: dict[int, tuple[str, str, str]] = {}
        touched: set[int] = set()
        keeps_changed: set[int] = set()
        for r in self.conn.execute("SELECT group_id, keep_json, status, note FROM history ORDER BY group_id, id"):
            gid = r["group_id"]
            if gid not in first:
                first[gid] = (r["keep_json"], r["status"], r["note"])
                continue
            keep0, status0, note0 = first[gid]
            if r["status"] != status0 or r["note"] != note0:
                touched.add(gid)
            elif r["keep_json"] != keep0:
                keeps_changed.add(gid)
        for gid in keeps_changed - touched:
            members = self.members(gid)
            if policy is None or not members:
                touched.add(gid)
                continue
            winner = policy.select([m.record for m in members])
            if any(m.keep != (m.path == winner.path) for m in members):
                touched.add(gid)
        self.conn.executemany("UPDATE groups SET touched = 1 WHERE id = ?", [(gid,) for gid in touched])

    def __enter__(self) -> ReviewDB:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    # -- meta ------------------------------------------------------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else row[0]

    def set_meta(self, key: str, value: object) -> None:
        self.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, str(value)))

    def begin_session(self, *, image_dir: str, model_id: str, review_threshold: float, auto_threshold: float, policy: str) -> None:
        """Record what the groups were produced with. A later session with another model refuses to mix."""
        previous = self.get_meta("model_id")
        if previous is not None and previous != model_id:
            raise ValueError(
                f"Review database {self.path} holds groups from model '{previous}', not '{model_id}'. Use another file."
            )
        for k, v in (
            ("image_dir", os.path.abspath(image_dir)),
            ("model_id", model_id),
            ("review_threshold", review_threshold),
            ("auto_threshold", auto_threshold),
            ("policy", policy),
        ):
            self.set_meta(k, v)
        if self.get_meta("created_at") is None:
            self.set_meta("created_at", time.time())
        self.conn.commit()

    @property
    def review_threshold(self) -> float | None:
        v = self.get_meta("review_threshold")
        return None if v is None else float(v)

    @property
    def auto_threshold(self) -> float | None:
        v = self.get_meta("auto_threshold")
        return None if v is None else float(v)

    # -- matching results --------------------------------------------------------------------------------------------

    def upsert_groups(
        self,
        groups: Iterable[tuple[Sequence[ImageRecord], Sequence[tuple[str, str, float]]]],
        policy: Policy,
        auto_threshold: float,
        pixel: dict[EdgeKey, PixelVerdict] | None = None,
    ) -> UpsertStats:
        """Store the groups of one matching run.

        Each item is ``(member records, edges as (path_a, path_b, distance))``; ``pixel`` holds the pixel check of
        every edge, or is ``None`` when the check is off. New groups get the policy's winner as the only kept member
        and start as :func:`initial_status` says. Existing groups keep their decision; file metadata, edges and
        pixel results are refreshed, and an untouched group's status is re-derived. Groups not in this run that
        were still open are marked ``stale``.
        """
        stats = UpsertStats()
        now = time.time()
        seen: set[str] = set()
        for records, edges in groups:
            key = group_key(r.path for r in records)
            seen.add(key)
            if not edges:
                raise ValueError("A group needs at least one edge")
            dmin = min(d for _, _, d in edges)
            dmax = max(d for _, _, d in edges)
            verdicts = None if pixel is None else [pixel[(a, b)] for a, b, _ in edges]
            group_pixel = None if verdicts is None else ("same" if all(v.same for v in verdicts) else "different")
            if group_pixel == "same":
                stats.pixel_same += 1
            start = initial_status(dmax, group_pixel, auto_threshold)
            winner = policy.select(records)
            by_pair = {}
            for a, b, d in edges:
                by_pair[(a, b)] = d
                by_pair[(b, a)] = d
            row = self.conn.execute("SELECT id, status, touched FROM groups WHERE key = ?", (key,)).fetchone()
            if row is None:
                status = start
                cur = self.conn.execute(
                    "INSERT INTO groups (key, n, min_distance, max_distance, status, note, updated_at, pixel) VALUES (?, ?, ?, ?, ?, '', ?, ?)",
                    (key, len(records), dmin, dmax, status, now, group_pixel),
                )
                assert cur.lastrowid is not None
                gid = int(cur.lastrowid)
                self.conn.executemany(
                    "INSERT INTO members (group_id, path, keep, width, height, size, format, mtime, distance_to_winner) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            gid,
                            r.path,
                            int(r is winner),
                            r.width,
                            r.height,
                            r.size,
                            r.format,
                            r.mtime,
                            0.0 if r is winner else by_pair.get((r.path, winner.path)),
                        )
                        for r in records
                    ],
                )
                self._snapshot(gid, now)
                stats.new += 1
                if status == "decided":
                    stats.predecided += 1
            else:
                gid = int(row["id"])
                old_status = new_status = row["status"]
                if new_status == "stale":
                    new_status = "pending"
                rederive = not row["touched"] and new_status in ("pending", "decided")
                if rederive:
                    new_status = start
                    if old_status == "pending" and new_status == "decided":
                        stats.redecided += 1
                    elif old_status == "decided" and new_status == "pending":
                        stats.reopened += 1
                self.conn.execute(
                    "UPDATE groups SET n = ?, min_distance = ?, max_distance = ?, status = ?, updated_at = ?, pixel = ? WHERE id = ?",
                    (len(records), dmin, dmax, new_status, now, group_pixel, gid),
                )
                if rederive and new_status != old_status:
                    # Still the starting point, not a decision: restart the history from it so undo cannot step
                    # back into a status the old rule produced.
                    self.conn.execute("DELETE FROM history WHERE group_id = ?", (gid,))
                    self._snapshot(gid, now)
                for r in records:
                    self.conn.execute(
                        "UPDATE members SET width = ?, height = ?, size = ?, format = ?, mtime = ?, distance_to_winner = ? WHERE group_id = ? AND path = ?",
                        (
                            r.width,
                            r.height,
                            r.size,
                            r.format,
                            r.mtime,
                            0.0 if r is winner else by_pair.get((r.path, winner.path)),
                            gid,
                            r.path,
                        ),
                    )
                self.conn.execute("DELETE FROM edges WHERE group_id = ?", (gid,))
                stats.kept += 1
            self.conn.executemany(
                "INSERT INTO edges (group_id, path_a, path_b, distance, pixel, pixel_peak, pixel_mean) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        gid,
                        a,
                        b,
                        d,
                        *((None, None, None) if verdicts is None else (verdicts[k].status, verdicts[k].peak, verdicts[k].mean)),
                    )
                    for k, (a, b, d) in enumerate(edges)
                ],
            )
        open_rows = self.conn.execute("SELECT id, key FROM groups WHERE status IN ('pending', 'decided', 'skipped')").fetchall()
        for r in open_rows:
            if r["key"] not in seen:
                self.conn.execute("UPDATE groups SET status = 'stale', updated_at = ? WHERE id = ?", (now, r["id"]))
                stats.stale += 1
        self.conn.commit()
        return stats

    # -- reading -----------------------------------------------------------------------------------------------------

    _GROUP_SELECT = "SELECT g.*, (SELECT COUNT(*) FROM members m WHERE m.group_id = g.id AND m.keep) AS kept FROM groups g"

    @staticmethod
    def _group(row: sqlite3.Row) -> GroupRow:
        return GroupRow(
            row["id"],
            row["key"],
            row["n"],
            row["min_distance"],
            row["max_distance"],
            row["status"],
            row["note"],
            row["updated_at"],
            row["pixel"],
            bool(row["touched"]),
            bool(row["discard"]),
            row["kept"],
        )

    def groups(
        self,
        status: str | Sequence[str] | None = None,
        *,
        pixel: str | None = None,
        keeps_nothing: bool = False,
        discarded: bool = False,
    ) -> list[GroupRow]:
        """Groups ordered by minimum distance (the surest first). ``status`` filters; ``None`` means every status
        except ``stale``. ``pixel`` ("same" / "different") filters on the pixel check. ``keeps_nothing`` lists only
        groups with every keep unticked that are not marked discard (Apply refuses those when decided);
        ``discarded`` only groups marked discard."""
        if status is None:
            statuses: tuple[str, ...] = ("pending", "decided", "skipped", "applied")
        elif isinstance(status, str):
            statuses = (status,)
        else:
            statuses = tuple(status)
        marks = ",".join("?" * len(statuses))
        sql = f"SELECT * FROM ({self._GROUP_SELECT}) WHERE status IN ({marks})"
        args: list[object] = list(statuses)
        if pixel is not None:
            sql += " AND pixel = ?"
            args.append(pixel)
        if keeps_nothing:
            sql += " AND kept = 0 AND NOT discard"
        if discarded:
            sql += " AND discard"
        rows = self.conn.execute(sql + " ORDER BY min_distance, id", args).fetchall()
        return [self._group(r) for r in rows]

    def group(self, group_id: int) -> GroupRow:
        row = self.conn.execute(f"{self._GROUP_SELECT} WHERE g.id = ?", (group_id,)).fetchone()
        if row is None:
            raise KeyError(group_id)
        return self._group(row)

    def counts(self) -> dict[str, int]:
        out = {s: 0 for s in STATUSES}
        for r in self.conn.execute("SELECT status, COUNT(*) AS c FROM groups GROUP BY status"):
            out[r["status"]] = r["c"]
        return out

    def pixel_counts(self) -> dict[tuple[str, str | None], int]:
        """``{(status, pixel): count}`` over every group."""
        return {
            (r["status"], r["pixel"]): r["c"]
            for r in self.conn.execute("SELECT status, pixel, COUNT(*) AS c FROM groups GROUP BY status, pixel")
        }

    def members(self, group_id: int) -> list[MemberRow]:
        rows = self.conn.execute("SELECT * FROM members WHERE group_id = ? ORDER BY path", (group_id,)).fetchall()
        return [
            MemberRow(
                r["path"], bool(r["keep"]), r["width"], r["height"], r["size"], r["format"], r["mtime"], r["distance_to_winner"]
            )
            for r in rows
        ]

    def edges(self, group_id: int) -> list[tuple[str, str, float]]:
        rows = self.conn.execute(
            "SELECT path_a, path_b, distance FROM edges WHERE group_id = ? ORDER BY distance", (group_id,)
        ).fetchall()
        return [(r["path_a"], r["path_b"], r["distance"]) for r in rows]

    def edge_pixels(self, group_id: int) -> dict[EdgeKey, PixelVerdict]:
        rows = self.conn.execute(
            "SELECT path_a, path_b, pixel, pixel_peak, pixel_mean FROM edges WHERE group_id = ? AND pixel IS NOT NULL", (group_id,)
        )
        return {(r["path_a"], r["path_b"]): PixelVerdict(r["pixel"], r["pixel_peak"], r["pixel_mean"]) for r in rows}

    def pixel_cache(self) -> dict[EdgeKey, tuple[PixelVerdict, tuple[float, int, float, int]]]:
        """Stored pixel results by edge, with ``(mtime_a, size_a, mtime_b, size_b)`` as recorded when they were
        measured, so a re-run can skip pairs whose files did not change. Verdicts are re-derived from peak and mean,
        so a changed rule applies without re-reading anything; unreadable pairs are not cached."""
        rows = self.conn.execute(
            """SELECT e.path_a, e.path_b, e.pixel, e.pixel_peak, e.pixel_mean, ma.mtime AS mta, ma.size AS sa, mb.mtime AS mtb, mb.size AS sb
               FROM edges e
               JOIN members ma ON ma.group_id = e.group_id AND ma.path = e.path_a
               JOIN members mb ON mb.group_id = e.group_id AND mb.path = e.path_b
               WHERE e.pixel IS NOT NULL AND e.pixel != 'error'"""
        )
        out = {}
        for r in rows:
            if r["pixel_peak"] is not None:
                verdict = PixelVerdict.measured(r["pixel_peak"], r["pixel_mean"])
            else:
                verdict = PixelVerdict(r["pixel"])
            out[(r["path_a"], r["path_b"])] = (verdict, (r["mta"], r["sa"], r["mtb"], r["sb"]))
        return out

    def groups_of(self, path: str) -> list[int]:
        return [r["group_id"] for r in self.conn.execute("SELECT group_id FROM members WHERE path = ?", (path,))]

    # -- deciding ----------------------------------------------------------------------------------------------------

    def _snapshot(self, group_id: int, now: float) -> None:
        keeps = {
            r["path"]: bool(r["keep"]) for r in self.conn.execute("SELECT path, keep FROM members WHERE group_id = ?", (group_id,))
        }
        g = self.conn.execute("SELECT status, note, discard FROM groups WHERE id = ?", (group_id,)).fetchone()
        self.conn.execute(
            "INSERT INTO history (group_id, at, keep_json, status, note, discard) VALUES (?, ?, ?, ?, ?, ?)",
            (group_id, now, json.dumps(keeps, sort_keys=True), g["status"], g["note"], g["discard"]),
        )

    def decide(
        self,
        group_id: int,
        *,
        keeps: dict[str, bool] | None = None,
        status: str | None = None,
        note: str | None = None,
        discard: bool | None = None,
        touch: bool = True,
    ) -> None:
        """Change keep flags, status, note and/or the discard mark of a group in one committed step that ``undo`` can
        revert. ``touch=False`` records the step without marking the group as reviewed (a bulk Policy pick).

        Ticking a keep clears the discard mark. Setting ``status="decided"`` on a group that would keep nothing
        without being marked discard raises ``ValueError``."""
        if status is not None and status not in STATUSES:
            raise ValueError(f"Unknown status {status!r}")
        g = self.group(group_id)
        if g.status == "applied":
            raise ValueError("An applied group cannot be changed; undo the apply first")
        now = time.time()
        current = {m.path: m.keep for m in self.members(group_id)}
        if keeps:
            unknown = set(keeps) - set(current)
            if unknown:
                raise KeyError(f"Not members of group {group_id}: {sorted(unknown)}")
        after = {**current, **{p: bool(v) for p, v in (keeps or {}).items()}}
        if discard is None:
            discard = g.discard and not any(after.values())
        if discard and any(after.values()):
            raise ValueError("A discarded group keeps nothing; untick every member")
        if status == "decided" and not any(after.values()) and not discard:
            raise ValueError(f"Group {group_id} keeps nothing; tick a member to keep, or discard the whole group")
        changes = (
            after != current
            or (status is not None and status != g.status)
            or (note is not None and note != g.note)
            or discard != g.discard
        )
        if not changes:
            return  # nothing to record: no history step, and the group stays untouched
        if keeps:
            self.conn.executemany(
                "UPDATE members SET keep = ? WHERE group_id = ? AND path = ?", [(int(v), group_id, p) for p, v in keeps.items()]
            )
        sets: list[str] = ["updated_at = ?", "discard = ?"] + (["touched = 1"] if touch else [])
        args: list[object] = [now, int(discard)]
        if status is not None:
            sets.append("status = ?")
            args.append(status)
        if note is not None:
            sets.append("note = ?")
            args.append(note)
        self.conn.execute(f"UPDATE groups SET {', '.join(sets)} WHERE id = ?", (*args, group_id))
        self._snapshot(group_id, now)
        self.conn.commit()

    def discard(self, group_id: int) -> None:
        """Mark a group to have every member moved by Apply (keeps nothing, decided)."""
        self.decide(group_id, keeps={m.path: False for m in self.members(group_id)}, status="decided", discard=True)

    def keep_only(self, group_id: int, path: str, *, status: str | None = "decided") -> None:
        keeps = {m.path: m.path == path for m in self.members(group_id)}
        if path not in keeps:
            raise KeyError(path)
        self.decide(group_id, keeps=keeps, status=status)

    def reset_to_policy(self, group_id: int, policy: Policy, *, touch: bool = True) -> None:
        members = self.members(group_id)
        winner = policy.select([m.record for m in members])
        self.decide(group_id, keeps={m.path: m.path == winner.path for m in members}, touch=touch)

    def reset_pending_to_policy(self, policy: Policy) -> int:
        """Set the keeps of every pending group to ``policy``'s winner (status unchanged, one history step each, not
        counted as reviewing the group). Returns the number of pending groups."""
        n = 0
        for g in self.groups(status="pending"):
            self.reset_to_policy(g.id, policy, touch=False)
            n += 1
        return n

    def undo(self, group_id: int) -> bool:
        """Revert the last ``decide`` of a group. Returns False when there is nothing left to undo."""
        rows = self.conn.execute(
            "SELECT id, keep_json, status, note, discard FROM history WHERE group_id = ? ORDER BY id DESC LIMIT 2", (group_id,)
        ).fetchall()
        if len(rows) < 2:
            return False
        current, previous = rows
        if self.group(group_id).status == "applied":
            raise ValueError("An applied group cannot be changed; undo the apply first")
        for path, keep in json.loads(previous["keep_json"]).items():
            self.conn.execute("UPDATE members SET keep = ? WHERE group_id = ? AND path = ?", (int(keep), group_id, path))
        self.conn.execute(
            "UPDATE groups SET status = ?, note = ?, discard = ?, updated_at = ? WHERE id = ?",
            (previous["status"], previous["note"], previous["discard"], time.time(), group_id),
        )
        self.conn.execute("DELETE FROM history WHERE id = ?", (current["id"],))
        if self.conn.execute("SELECT COUNT(*) FROM history WHERE group_id = ?", (group_id,)).fetchone()[0] == 1:
            self.conn.execute("UPDATE groups SET touched = 0 WHERE id = ?", (group_id,))  # back at the starting point
        self.conn.commit()
        return True

    # -- applying ----------------------------------------------------------------------------------------------------

    def plan_apply(self, group_ids: Sequence[int] | None = None) -> list[Move]:
        """Files that applying the decided groups would move, with what is kept instead. A decided group keeping
        nothing is left out unless it is marked discard (see :meth:`groups` with ``keeps_nothing``)."""
        if group_ids is None:
            targets = [g.id for g in self.groups("decided")]
        else:
            targets = [gid for gid in group_ids if self.group(gid).status == "decided"]
        moves: list[Move] = []
        for gid in targets:
            members = self.members(gid)
            keep = tuple(m.path for m in members if m.keep)
            if not keep and not self.group(gid).discard:
                logger.debug("Group %d keeps nothing and is not marked discard; refusing to move all of its files", gid)
                continue
            moves.extend(Move(gid, m.path, keep) for m in members if not m.keep)
        return moves

    def mark_applied(self, group_id: int, moved: Iterable[tuple[str, str]]) -> None:
        now = time.time()
        self.conn.executemany(
            "INSERT OR REPLACE INTO applied (path, trash_path, group_id, at) VALUES (?, ?, ?, ?)",
            [(p, t, group_id, now) for p, t in moved],
        )
        self.conn.execute("UPDATE groups SET status = 'applied', touched = 1, updated_at = ? WHERE id = ?", (now, group_id))
        self._snapshot(group_id, now)
        self.conn.commit()

    def applied(self, group_id: int | None = None) -> list[tuple[str, str, int]]:
        sql = "SELECT path, trash_path, group_id FROM applied"
        rows = self.conn.execute(
            sql + (" WHERE group_id = ?" if group_id is not None else ""), (group_id,) if group_id is not None else ()
        ).fetchall()
        return [(r["path"], r["trash_path"], r["group_id"]) for r in rows]

    def mark_unapplied(self, group_id: int) -> None:
        now = time.time()
        self.conn.execute("DELETE FROM applied WHERE group_id = ?", (group_id,))
        self.conn.execute("UPDATE groups SET status = 'decided', updated_at = ? WHERE id = ?", (now, group_id))
        self._snapshot(group_id, now)
        self.conn.commit()


def changed_on_disk(image_dir: str, members: Sequence[MemberRow]) -> list[str]:
    """Member paths whose file is missing or has a different mtime/size than when the group was stored."""
    out = []
    for m in members:
        try:
            st = os.stat(os.path.join(image_dir, m.path))
        except OSError:
            out.append(m.path)
            continue
        if st.st_mtime != m.mtime or st.st_size != m.size:
            out.append(m.path)
    return out


def apply_decisions(
    review: ReviewDB,
    image_dir: str,
    trash_dir: str,
    *,
    dry_run: bool,
    group_ids: Sequence[int] | None = None,
    progress: Callable[[int], object] | None = None,
) -> tuple[int, list[int]]:
    """Move the losers of every decided group to ``trash_dir``. Returns ``(files moved, group ids refused)``.

    A group is refused, and left ``decided``, when any member changed on disk since it was stored; re-run
    ``update-db`` and re-match for those.
    """
    moves = review.plan_apply(group_ids)
    by_group: dict[int, list[Move]] = {}
    for mv in moves:
        by_group.setdefault(mv.group_id, []).append(mv)
    moved_total = 0
    refused: list[int] = []
    for gid, group_moves in by_group.items():
        members = review.members(gid)
        changed = changed_on_disk(image_dir, members)
        if changed:
            logger.warning(
                "Group %d: %d file(s) changed on disk since matching, skipping: %s", gid, len(changed), ", ".join(changed[:5])
            )
            refused.append(gid)
            continue
        done: list[tuple[str, str]] = []
        for mv in group_moves:
            if move_to_trash(
                image_dir,
                mv.path,
                trash_dir,
                dry_run=dry_run,
                reason=f"keeping {', '.join(mv.keep)}" if mv.keep else "whole group discarded",
            ):
                done.append((mv.path, os.path.join(trash_dir, mv.path)))
        moved_total += len(done)
        if not dry_run:
            review.mark_applied(gid, done)
        if progress is not None:
            progress(1)
    return moved_total, refused


def unapply_group(review: ReviewDB, image_dir: str, group_id: int) -> int:
    """Move a group's trashed files back where they were. Returns the number restored."""
    restored = 0
    for path, trash_path, _ in review.applied(group_id):
        dest = os.path.join(image_dir, path)
        if not os.path.exists(trash_path):
            logger.warning("Cannot restore %s: %s is gone", path, trash_path)
            continue
        if os.path.exists(dest):
            logger.warning("Cannot restore %s: destination exists", path)
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        os.replace(trash_path, dest)
        restored += 1
    review.mark_unapplied(group_id)
    return restored


def store_matches(
    review: ReviewDB,
    groups: Sequence[DuplicateGroup],
    records: Sequence[ImageRecord],
    policy: Policy,
    auto_threshold: float,
    pixel: dict[tuple[int, int], PixelVerdict] | None = None,
) -> UpsertStats:
    """Translate row-index groups from :func:`~.dedupe.group_edges` (and pixel results keyed by row pairs) into
    path-keyed rows and upsert them."""
    items = [([records[i] for i in g.members], [(records[i].path, records[j].path, d) for i, j, d in g.edges]) for g in groups]
    by_path = None if pixel is None else {(records[i].path, records[j].path): v for (i, j), v in pixel.items()}
    return review.upsert_groups(items, policy, auto_threshold, by_path)
