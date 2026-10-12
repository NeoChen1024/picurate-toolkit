"""Review sessions: a project's duplicate groups found at a loose threshold, and the reviewer's decisions.

Everything lives in ``review.sqlite`` in the project, a file of its own beside ``embeddings.sqlite``, so a
``--force-update`` rebuild of the embeddings never touches decisions, and a review of a large library (tens of
thousands of groups) is indexed rather than parsed. No Qt in here; the GUI and the CLI both drive this module.
Nothing is moved from here: the losers of decided groups count as excluded when the project is read.

Members are identified by item ID (see :mod:`picurate.common.images`), and a group by the BLAKE3 hash of its sorted
member IDs. Re-matching upserts by that key: an unchanged group keeps
its row, decision and history; a group whose membership changed is a new group, and the old row is marked
``stale``. A group nobody has changed yet (``touched`` is 0) gets its starting status re-derived on every run, so a
new rule (the pixel check, another auto threshold) reaches the groups still waiting for review. A new group that
shares members with a group the reviewer changed starts pending with that group's keeps (``carried_from``), and is
not re-derived: a human decision is only ever replaced by another one.

Starting status: with the pixel check, ``decided`` when every edge differs only by compression, else ``pending``;
without it, ``decided`` when every edge is within the auto threshold.

A decided group must keep at least one member, unless the reviewer marked it ``discard``: then every member is
excluded. Unticking every keep is not enough on its own, since that is as often a slip as an intent.

A file of another schema version is refused; there is no upgrade, matching runs again into a new file.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from blake3 import blake3

from ..common.images import ImageRecord, Roots
from .keeping import Policy
from .pixel_check import SAME, PixelVerdict

if TYPE_CHECKING:
    from .dedupe import DuplicateGroup  # imports torch; only a type here

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "5"
GROUP_KEY_CONTEXT = "picurate review group v1"
STATUSES = ("pending", "decided", "skipped", "stale")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS groups (
    id           INTEGER PRIMARY KEY,
    key          BLOB NOT NULL UNIQUE,
    n            INTEGER NOT NULL,
    min_distance REAL NOT NULL,
    max_distance REAL NOT NULL,
    status       TEXT NOT NULL,
    note         TEXT NOT NULL DEFAULT '',
    updated_at   REAL NOT NULL,
    pixel        TEXT,              -- 'same' when every edge differs only by compression, 'different', NULL: unchecked
    touched      INTEGER NOT NULL DEFAULT 0, -- changed by the reviewer since it was created
    discard      INTEGER NOT NULL DEFAULT 0, -- the reviewer chose to exclude every member, keeping none
    carried_from INTEGER                -- keeps taken over from this (now stale) group the reviewer had changed
);
CREATE INDEX IF NOT EXISTS groups_status ON groups (status, min_distance);
CREATE TABLE IF NOT EXISTS members (
    group_id           INTEGER NOT NULL REFERENCES groups (id) ON DELETE CASCADE,
    item               BLOB NOT NULL,
    source_id          BLOB NOT NULL,
    path               TEXT NOT NULL,
    keep               INTEGER NOT NULL,
    width              INTEGER NOT NULL,
    height             INTEGER NOT NULL,
    size               INTEGER NOT NULL,
    format             TEXT NOT NULL,
    mtime              REAL NOT NULL,
    distance_to_winner REAL,
    PRIMARY KEY (group_id, item)
);
CREATE INDEX IF NOT EXISTS members_item ON members (item);
CREATE TABLE IF NOT EXISTS edges (
    group_id INTEGER NOT NULL REFERENCES groups (id) ON DELETE CASCADE,
    item_a   BLOB NOT NULL,
    item_b   BLOB NOT NULL,
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
    keep_json TEXT NOT NULL,        -- {item ID hex: keep}
    status    TEXT NOT NULL,
    note      TEXT NOT NULL,
    discard   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS history_group ON history (group_id, id);
"""


def group_key(ids: Iterable[bytes]) -> bytes:
    return blake3(b"".join(sorted(ids)), derive_key_context=GROUP_KEY_CONTEXT).digest()


@dataclass(slots=True, frozen=True)
class GroupRow:
    id: int
    key: bytes
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
    carried_from: int | None = None


@dataclass(slots=True, frozen=True)
class MemberRow:
    item: bytes
    source_id: bytes
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
        return ImageRecord(self.path, self.mtime, self.size, self.width, self.height, self.format, self.source_id, self.item)

    def file(self, roots: Roots) -> str:
        return os.path.join(roots[self.source_id], self.path)


@dataclass(slots=True)
class UpsertStats:
    new: int = 0
    kept: int = 0
    stale: int = 0
    predecided: int = 0  # new groups that start decided
    redecided: int = 0  # untouched groups that went from pending to decided under the current rule
    reopened: int = 0  # untouched groups that went from decided back to pending
    pixel_same: int = 0  # groups of this run whose every edge differs only by compression
    carried: int = 0  # new groups that took over the keeps of a changed group sharing members


def initial_status(max_distance: float, pixel: str | None, auto_threshold: float) -> str:
    if pixel is not None:
        return "decided" if pixel == SAME else "pending"
    return "decided" if max_distance <= auto_threshold else "pending"


EdgeKey = tuple[bytes, bytes]


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
        self.conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        version = self.get_meta("schema_version")
        if version not in (None, SCHEMA_VERSION):
            self.conn.close()
            raise ValueError(
                f"Review database {path} uses schema version {version}, but this version of the tool needs {SCHEMA_VERSION}. "
                "Move it away and run matching again."
            )
        self.conn.executescript(_SCHEMA)
        if version is None:
            self.set_meta("schema_version", SCHEMA_VERSION)
        self.conn.commit()

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

    def begin_session(self, *, project_id: str, model_id: str, review_threshold: float, auto_threshold: float, policy: str) -> None:
        """Record what the groups were produced with. A later session with another model or project refuses to mix."""
        for key, value in (("project_id", project_id), ("model_id", model_id)):
            previous = self.get_meta(key)
            if previous is not None and previous != value:
                raise ValueError(f"Review database {self.path} holds groups of {key} '{previous}', not '{value}'. Move it away.")
        for k, v in (
            ("project_id", project_id),
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

    def _decided_before(self, items: Sequence[bytes], current: set[bytes]) -> tuple[int, dict[bytes, bool]] | None:
        """The most recently changed group the reviewer touched (or that carried keeps itself) sharing a member with
        ``items`` and not part of this run, with its keeps."""
        marks = ",".join("?" * len(items))
        rows = self.conn.execute(
            f"""SELECT DISTINCT g.id, g.key FROM groups g JOIN members m ON m.group_id = g.id
                WHERE m.item IN ({marks}) AND (g.touched OR g.carried_from IS NOT NULL)
                ORDER BY g.updated_at DESC, g.id DESC""",
            list(items),
        ).fetchall()
        for row in rows:
            if row["key"] not in current:
                keeps = self.conn.execute("SELECT item, keep FROM members WHERE group_id = ?", (row["id"],)).fetchall()
                return row["id"], {r["item"]: bool(r["keep"]) for r in keeps}
        return None

    def upsert_groups(
        self,
        groups: Iterable[tuple[Sequence[ImageRecord], Sequence[tuple[str, str, float]]]],
        policy: Policy,
        auto_threshold: float,
        pixel: dict[EdgeKey, PixelVerdict] | None = None,
    ) -> UpsertStats:
        """Store the groups of one matching run.

        Each item is ``(member records, edges as (id_a, id_b, distance))``; ``pixel`` holds the pixel check of
        every edge, or is ``None`` when the check is off. New groups get the policy's winner as the only kept member
        and start as :func:`initial_status` says. Existing groups keep their decision; file metadata, edges and
        pixel results are refreshed, and an untouched group's status is re-derived. Groups not in this run that
        were still open are marked ``stale``.
        """
        stats = UpsertStats()
        now = time.time()
        seen: set[str] = set()
        for records, edges in groups:
            key = group_key(r.id for r in records)
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
            row = self.conn.execute("SELECT id, status, touched, carried_from FROM groups WHERE key = ?", (key,)).fetchone()
            if row is None:
                status = start
                keeps = {r.id: r is winner for r in records}
                carried = self._decided_before([r.id for r in records], seen)
                if carried is not None:
                    carried, old_keeps = carried
                    overlap = {i: k for i, k in old_keeps.items() if i in keeps}
                    if any(overlap.values()):
                        keeps = {i: overlap.get(i, False) for i in keeps}
                    status = "pending"
                    stats.carried += 1
                cur = self.conn.execute(
                    "INSERT INTO groups (key, n, min_distance, max_distance, status, note, updated_at, pixel, carried_from)"
                    " VALUES (?, ?, ?, ?, ?, '', ?, ?, ?)",
                    (key, len(records), dmin, dmax, status, now, group_pixel, carried),
                )
                assert cur.lastrowid is not None
                gid = int(cur.lastrowid)
                self.conn.executemany(
                    "INSERT INTO members (group_id, item, source_id, path, keep, width, height, size, format, mtime, distance_to_winner)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            gid,
                            r.id,
                            r.source_id,
                            r.path,
                            int(keeps[r.id]),
                            r.width,
                            r.height,
                            r.size,
                            r.format,
                            r.mtime,
                            0.0 if r is winner else by_pair.get((r.id, winner.id)),
                        )
                        for r in records
                    ],
                )
                self._snapshot(gid, now)
                stats.new += 1
                if status == "decided" and carried is None:
                    stats.predecided += 1
            else:
                gid = int(row["id"])
                old_status = new_status = row["status"]
                if new_status == "stale":
                    new_status = "pending"
                rederive = not row["touched"] and row["carried_from"] is None and new_status in ("pending", "decided")
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
                        "UPDATE members SET width = ?, height = ?, size = ?, format = ?, mtime = ?, distance_to_winner = ?"
                        " WHERE group_id = ? AND item = ?",
                        (
                            r.width,
                            r.height,
                            r.size,
                            r.format,
                            r.mtime,
                            0.0 if r is winner else by_pair.get((r.id, winner.id)),
                            gid,
                            r.id,
                        ),
                    )
                self.conn.execute("DELETE FROM edges WHERE group_id = ?", (gid,))
                stats.kept += 1
            self.conn.executemany(
                "INSERT INTO edges (group_id, item_a, item_b, distance, pixel, pixel_peak, pixel_mean) VALUES (?, ?, ?, ?, ?, ?, ?)",
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
            row["carried_from"],
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
        groups with every keep unticked that are not marked discard (those cannot be decided);
        ``discarded`` only groups marked discard."""
        if status is None:
            statuses: tuple[str, ...] = ("pending", "decided", "skipped")
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
        rows = self.conn.execute("SELECT * FROM members WHERE group_id = ? ORDER BY path, item", (group_id,)).fetchall()
        return [
            MemberRow(
                r["item"],
                r["source_id"],
                r["path"],
                bool(r["keep"]),
                r["width"],
                r["height"],
                r["size"],
                r["format"],
                r["mtime"],
                r["distance_to_winner"],
            )
            for r in rows
        ]

    def edges(self, group_id: int) -> list[tuple[bytes, bytes, float]]:
        rows = self.conn.execute(
            "SELECT item_a, item_b, distance FROM edges WHERE group_id = ? ORDER BY distance", (group_id,)
        ).fetchall()
        return [(r["item_a"], r["item_b"], r["distance"]) for r in rows]

    def edge_pixels(self, group_id: int) -> dict[EdgeKey, PixelVerdict]:
        rows = self.conn.execute(
            "SELECT item_a, item_b, pixel, pixel_peak, pixel_mean FROM edges WHERE group_id = ? AND pixel IS NOT NULL", (group_id,)
        )
        return {(r["item_a"], r["item_b"]): PixelVerdict(r["pixel"], r["pixel_peak"], r["pixel_mean"]) for r in rows}

    def pixel_cache(self) -> dict[EdgeKey, tuple[PixelVerdict, tuple[float, int, float, int]]]:
        """Stored pixel results by edge, with ``(mtime_a, size_a, mtime_b, size_b)`` as recorded when they were
        measured, so a re-run can skip pairs whose files did not change. Verdicts are re-derived from peak and mean,
        so a changed rule applies without re-reading anything; unreadable pairs are not cached."""
        rows = self.conn.execute(
            """SELECT e.item_a, e.item_b, e.pixel, e.pixel_peak, e.pixel_mean, ma.mtime AS mta, ma.size AS sa, mb.mtime AS mtb, mb.size AS sb
               FROM edges e
               JOIN members ma ON ma.group_id = e.group_id AND ma.item = e.item_a
               JOIN members mb ON mb.group_id = e.group_id AND mb.item = e.item_b
               WHERE e.pixel IS NOT NULL AND e.pixel != 'error'"""
        )
        out = {}
        for r in rows:
            if r["pixel_peak"] is not None:
                verdict = PixelVerdict.measured(r["pixel_peak"], r["pixel_mean"])
            else:
                verdict = PixelVerdict(r["pixel"])
            out[(r["item_a"], r["item_b"])] = (verdict, (r["mta"], r["sa"], r["mtb"], r["sb"]))
        return out

    def groups_of(self, item: bytes) -> list[int]:
        return [r["group_id"] for r in self.conn.execute("SELECT group_id FROM members WHERE item = ?", (item,))]

    # -- deciding ----------------------------------------------------------------------------------------------------

    def _snapshot(self, group_id: int, now: float) -> None:
        keeps = {
            r["item"].hex(): bool(r["keep"])
            for r in self.conn.execute("SELECT item, keep FROM members WHERE group_id = ?", (group_id,))
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
        keeps: dict[bytes, bool] | None = None,
        status: str | None = None,
        note: str | None = None,
        discard: bool | None = None,
        touch: bool = True,
    ) -> None:
        """Change keep flags (by item ID), status, note and/or the discard mark of a group in one committed step that ``undo`` can
        revert. ``touch=False`` records the step without marking the group as reviewed (a bulk Policy pick).

        Ticking a keep clears the discard mark. Setting ``status="decided"`` on a group that would keep nothing
        without being marked discard raises ``ValueError``."""
        if status is not None and status not in STATUSES:
            raise ValueError(f"Unknown status {status!r}")
        g = self.group(group_id)
        now = time.time()
        current = {m.item: m.keep for m in self.members(group_id)}
        if keeps:
            unknown = set(keeps) - set(current)
            if unknown:
                raise KeyError(f"Not members of group {group_id}: {sorted(i.hex() for i in unknown)}")
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
                "UPDATE members SET keep = ? WHERE group_id = ? AND item = ?", [(int(v), group_id, i) for i, v in keeps.items()]
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
        """Mark a group to have every member excluded (keeps nothing, decided)."""
        self.decide(group_id, keeps={m.item: False for m in self.members(group_id)}, status="decided", discard=True)

    def keep_only(self, group_id: int, item: bytes, *, status: str | None = "decided") -> None:
        keeps = {m.item: m.item == item for m in self.members(group_id)}
        if item not in keeps:
            raise KeyError(item.hex())
        self.decide(group_id, keeps=keeps, status=status)

    def reset_to_policy(self, group_id: int, policy: Policy, *, touch: bool = True) -> None:
        members = self.members(group_id)
        winner = policy.select([m.record for m in members])
        self.decide(group_id, keeps={m.item: m.item == winner.id for m in members}, touch=touch)

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
        for item, keep in json.loads(previous["keep_json"]).items():
            self.conn.execute(
                "UPDATE members SET keep = ? WHERE group_id = ? AND item = ?", (int(keep), group_id, bytes.fromhex(item))
            )
        self.conn.execute(
            "UPDATE groups SET status = ?, note = ?, discard = ?, updated_at = ? WHERE id = ?",
            (previous["status"], previous["note"], previous["discard"], time.time(), group_id),
        )
        self.conn.execute("DELETE FROM history WHERE id = ?", (current["id"],))
        if self.conn.execute("SELECT COUNT(*) FROM history WHERE group_id = ?", (group_id,)).fetchone()[0] == 1:
            self.conn.execute("UPDATE groups SET touched = 0 WHERE id = ?", (group_id,))  # back at the starting point
        self.conn.commit()
        return True

    # -- exclusion -----------------------------------------------------------------------------------------------------

    def losers(self) -> dict[bytes, tuple[int, list[bytes]]]:
        """``{item: (group id, kept items)}`` for every member a decided group does not keep (every member of a
        discarded group). Decided groups that keep nothing without being discarded exclude nothing."""
        return losers(self.conn)


def store_matches(
    review: ReviewDB,
    groups: Sequence[DuplicateGroup],
    records: Sequence[ImageRecord],
    policy: Policy,
    auto_threshold: float,
    pixel: dict[tuple[int, int], PixelVerdict] | None = None,
) -> UpsertStats:
    """Translate row-index groups from :func:`~.dedupe.group_edges` (and pixel results keyed by row pairs) into
    ID-keyed rows and upsert them."""
    items = [([records[i] for i in g.members], [(records[i].id, records[j].id, d) for i, j, d in g.edges]) for g in groups]
    by_id = None if pixel is None else {(records[i].id, records[j].id): v for (i, j), v in pixel.items()}
    return review.upsert_groups(items, policy, auto_threshold, by_id)


def losers(conn: sqlite3.Connection, item: bytes | None = None) -> dict[bytes, tuple[int, list[bytes]]]:
    """See :meth:`ReviewDB.losers`; ``item`` limits it to one item. Also used on a read-only connection."""
    where = " AND m.item = ?" if item is not None else ""
    groups: dict[int, list[bytes]] = {}
    out: dict[bytes, tuple[int, list[bytes]]] = {}
    rows = conn.execute(
        f"""SELECT m.item, g.id FROM members m JOIN groups g ON g.id = m.group_id
            WHERE g.status = 'decided' AND NOT m.keep
              AND (g.discard OR EXISTS (SELECT 1 FROM members k WHERE k.group_id = g.id AND k.keep)){where}
            ORDER BY g.id""",
        () if item is None else (item,),
    ).fetchall()
    for member, gid in rows:
        if member in out:
            continue
        if gid not in groups:
            groups[gid] = [r[0] for r in conn.execute("SELECT item FROM members WHERE group_id = ? AND keep", (gid,))]
        out[member] = (gid, groups[gid])
    return out
