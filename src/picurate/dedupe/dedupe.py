"""Duplicate detection on top of :class:`~.similarity.DistanceIndex`, and moving losers to a trash directory."""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from ..common.images import ImageRecord
from .keeping import Policy
from .similarity import DistanceIndex, find_close_pairs_cross, find_close_pairs_self

logger = logging.getLogger(__name__)

Progress = Callable[[int], object] | None


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _log_matches(prefix: str, query_paths: Sequence[str], match_paths: Sequence[str], ii: np.ndarray, jj: np.ndarray, dd: np.ndarray) -> None:
    order = np.lexsort((jj, ii))
    ii, jj, dd = ii[order], jj[order], dd[order]
    for i in np.unique(ii):
        sel = ii == i
        matches = [(match_paths[int(j)], round(float(d), 4)) for j, d in zip(jj[sel], dd[sel])]
        logger.info("%s %s: %s", prefix, query_paths[int(i)], matches)


@dataclass(slots=True)
class DuplicateGroup:
    """A connected component of the "distance <= threshold" relation.

    ``members`` are row indices (sorted); ``edges`` the pairs that connected them, as ``(i, j, distance)`` with
    ``i < j``. Not every pair of members is an edge: A~B and B~C put A and C in one group even when A-C is above
    the threshold, which is exactly what a reviewer wants to see.
    """

    members: list[int]
    edges: list[tuple[int, int, float]]

    @property
    def min_distance(self) -> float:
        return min(d for _, _, d in self.edges)

    @property
    def max_distance(self) -> float:
        return max(d for _, _, d in self.edges)


def find_duplicate_edges(paths: Sequence[str], index: DistanceIndex, threshold: float, progress: Progress = None) -> list[tuple[int, int, float]]:
    """Every pair ``(i, j, d)`` with ``i < j`` and ``d <= threshold``, searching the upper triangle in blocks."""
    edges: list[tuple[int, int, float]] = []
    for start, end in index.iter_blocks():
        ii, jj, dd = find_close_pairs_self(index, threshold, start, end)
        if len(ii):
            _log_matches("Duplicates of", paths, paths, ii, jj, dd)
            edges.extend(zip(ii.tolist(), jj.tolist(), dd.tolist()))
        if progress:
            progress(end - start)
    return edges


def group_edges(n: int, edges: Sequence[tuple[int, int, float]]) -> list[DuplicateGroup]:
    """Merge ``edges`` over ``n`` rows into groups (union-find), each with the edges that belong to it."""
    uf = _UnionFind(n)
    for i, j, _ in edges:
        uf.union(i, j)
    by_root: dict[int, DuplicateGroup] = {}
    for i, j, d in edges:
        g = by_root.setdefault(uf.find(i), DuplicateGroup([], []))
        g.edges.append((i, j, d))
    for root, g in by_root.items():
        g.members = sorted({i for i, j, _ in g.edges} | {j for i, j, _ in g.edges})
    return sorted(by_root.values(), key=lambda g: g.members[0])


def find_duplicate_groups(paths: Sequence[str], index: DistanceIndex, threshold: float, progress: Progress = None) -> list[list[int]]:
    """Connected components of the "distance <= threshold" relation, as lists of row indices.

    The relation is not transitive: A may match B and B match C while A and C fall just outside the threshold, so all
    edges are collected first and merged with union-find. Only the upper triangle is searched, in blocks of rows.
    """
    return [g.members for g in group_edges(index.n, find_duplicate_edges(paths, index, threshold, progress))]


def find_cross_pairs(
    query_paths: Sequence[str], queries: DistanceIndex, base_paths: Sequence[str], base: DistanceIndex, threshold: float, progress: Progress = None
) -> list[tuple[int, int, float]]:
    """Every ``(query_row, base_row, distance)`` with ``distance <= threshold``."""
    pairs: list[tuple[int, int, float]] = []
    for start, end in queries.iter_blocks(other=base):
        ii, jj, dd = find_close_pairs_cross(queries, base, threshold, start, end)
        if len(ii):
            _log_matches("Already in base:", query_paths, base_paths, ii, jj, dd)
            pairs.extend(zip(ii.tolist(), jj.tolist(), dd.tolist()))
        if progress:
            progress(end - start)
    return pairs


def find_cross_duplicates(
    query_paths: Sequence[str], queries: DistanceIndex, base_paths: Sequence[str], base: DistanceIndex, threshold: float, progress: Progress = None
) -> dict[int, int]:
    """For every query row that has a match in ``base``: ``{query_row: number_of_matches}``."""
    hits: dict[int, int] = {}
    for i, _, _ in find_cross_pairs(query_paths, queries, base_paths, base, threshold, progress):
        hits[i] = hits.get(i, 0) + 1
    return hits


def move_to_trash(root_dir: str, relative_path: str, trash_dir: str, *, dry_run: bool, reason: str = "") -> bool:
    """Move one file into ``trash_dir`` preserving its relative path. Returns True when moved (or would be, in dry run)."""
    src = os.path.join(root_dir, relative_path)
    dst = os.path.join(trash_dir, relative_path)
    suffix = f" ({reason})" if reason else ""
    if dry_run:
        logger.info('[dry run] would move "%s" to trash%s', src, suffix)
        return True
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
    except OSError as e:
        logger.error('Error moving "%s" to trash: %s', src, e)
        return False
    logger.info('Moved "%s" to trash%s', src, suffix)
    return True


def trash_duplicate_groups(
    groups: Sequence[Sequence[int]], records: Sequence[ImageRecord], root_dir: str, trash_dir: str, policy: Policy, *, dry_run: bool
) -> int:
    """Apply ``policy`` to each group, keep the winner, move the rest. Returns the number of files moved."""
    moved = 0
    for group in groups:
        candidates = [records[i] for i in group]
        keep = policy.select(candidates)
        for rec in candidates:
            if rec is not keep and move_to_trash(root_dir, rec.path, trash_dir, dry_run=dry_run, reason=f"keeping {keep.path}"):
                moved += 1
    return moved
