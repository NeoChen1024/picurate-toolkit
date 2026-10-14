"""Keeping policies: decide which copy of a duplicate group survives.

A policy is an ordered list of criteria over :class:`~.db_store.ImageRecord` attributes. Candidates are sorted
lexicographically by the criteria and the first one is kept. Policies are declared in TOML; the built-in ones live in
``policies.toml`` next to this module and users can add or override policies with their own file. See that file for
the schema.
"""

from __future__ import annotations

import heapq
import os
import re
import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from importlib import resources
from typing import Any

from ..common.images import ImageRecord

NUMERIC_ATTRS = ("size", "mtime", "width", "height", "pixels")
STRING_ATTRS = ("path", "dirname", "basename", "ext")
CATEGORICAL_ATTRS = ("format",)


class PolicyError(ValueError):
    """A policy file or policy definition is invalid."""


def _attribute(rec: ImageRecord, attr: str) -> Any:
    if attr == "dirname":
        return os.path.dirname(rec.path)
    if attr == "basename":
        return os.path.basename(rec.path)
    if attr == "ext":
        return os.path.splitext(rec.path)[1].lower()
    return getattr(rec, attr)


@dataclass(frozen=True, slots=True)
class Criterion:
    attr: str
    prefer: str | None = None  # "max" | "min", numeric attrs
    patterns: tuple[re.Pattern[str], ...] = ()  # string attrs
    order: tuple[str, ...] = ()  # categorical attrs

    @classmethod
    def from_dict(cls, raw: dict[str, Any], where: str) -> Criterion:
        if not isinstance(raw, dict) or "attr" not in raw:
            raise PolicyError(f"{where}: each criterion must be a table with an 'attr' key, got {raw!r}")
        attr = raw["attr"]
        keys = set(raw) - {"attr"}
        if attr in NUMERIC_ATTRS:
            if keys != {"prefer"} or raw["prefer"] not in ("max", "min"):
                raise PolicyError(f"{where}: numeric attr {attr!r} takes exactly prefer = 'max' | 'min'")
            return cls(attr, prefer=raw["prefer"])
        if attr in STRING_ATTRS:
            if keys != {"match"}:
                raise PolicyError(f"{where}: string attr {attr!r} takes exactly match = 'regex' | ['regex', ...]")
            patterns = raw["match"] if isinstance(raw["match"], list) else [raw["match"]]
            try:
                compiled = tuple(re.compile(p) for p in patterns)
            except (re.error, TypeError) as e:
                raise PolicyError(f"{where}: invalid regex in match: {e}") from e
            if not compiled:
                raise PolicyError(f"{where}: match must contain at least one pattern")
            return cls(attr, patterns=compiled)
        if attr in CATEGORICAL_ATTRS:
            if keys != {"order"} or not isinstance(raw["order"], list) or not raw["order"]:
                raise PolicyError(f"{where}: categorical attr {attr!r} takes exactly order = ['A', 'B', ...]")
            return cls(attr, order=tuple(str(v).upper() for v in raw["order"]))
        raise PolicyError(f"{where}: unknown attr {attr!r}; expected one of {NUMERIC_ATTRS + STRING_ATTRS + CATEGORICAL_ATTRS}")

    def key(self, rec: ImageRecord) -> Any:
        """Sort key for one candidate; smaller is better."""
        value = _attribute(rec, self.attr)
        if self.prefer is not None:
            return -value if self.prefer == "max" else value
        if self.patterns:
            for rank, pattern in enumerate(self.patterns):
                if pattern.search(value):
                    return rank
            return len(self.patterns)
        try:
            return self.order.index(str(value).upper())
        except ValueError:
            return len(self.order)


@dataclass(frozen=True, slots=True)
class Policy:
    name: str
    criteria: tuple[Criterion, ...]
    description: str = ""
    variants: re.Pattern[str] | None = None  # on the basename, with groups 'work' and 'page'

    @classmethod
    def from_dict(cls, name: str, raw: dict[str, Any]) -> Policy:
        where = f"policy.{name}"
        criteria = raw.get("criteria")
        if not isinstance(criteria, list) or not criteria:
            raise PolicyError(f"{where}: 'criteria' must be a non-empty list")
        unknown = set(raw) - {"criteria", "description", "variants"}
        if unknown:
            raise PolicyError(f"{where}: unknown keys {sorted(unknown)}")
        variants = None
        if "variants" in raw:
            try:
                variants = re.compile(raw["variants"])
            except (re.error, TypeError) as e:
                raise PolicyError(f"{where}: invalid regex in variants: {e}") from e
            if not {"work", "page"} <= set(variants.groupindex):
                raise PolicyError(f"{where}: variants needs the named groups (?P<work>...) and (?P<page>...)")
        return cls(
            name,
            tuple(Criterion.from_dict(c, f"{where}.criteria[{i}]") for i, c in enumerate(criteria)),
            str(raw.get("description", "")),
            variants,
        )

    def sort(self, candidates: Sequence[ImageRecord]) -> list[ImageRecord]:
        """Candidates ordered best-first. Ties keep input order (sorted is stable)."""
        return sorted(candidates, key=lambda rec: tuple(c.key(rec) for c in self.criteria))

    def select(self, candidates: Sequence[ImageRecord]) -> ImageRecord:
        if not candidates:
            raise ValueError("select() needs at least one candidate")
        return self.sort(candidates)[0]

    def pages(self, candidates: Sequence[ImageRecord]) -> dict[bytes, tuple[str, str]]:
        """``(work, page)`` of the candidates that are pages of a work with at least two different pages among them."""
        if self.variants is None:
            return {}
        parsed = {}
        for rec in candidates:
            m = self.variants.search(os.path.basename(rec.path))
            if m is not None:
                parsed[rec.id] = (m["work"], m["page"])
        works: dict[str, set[str]] = {}
        for work, page in parsed.values():
            works.setdefault(work, set()).add(page)
        return {i: wp for i, wp in parsed.items() if len(works[wp[0]]) > 1}

    def keepers(self, candidates: Sequence[ImageRecord], edges: Iterable[tuple[bytes, bytes, float]]) -> dict[bytes, bytes]:
        """The kept candidate for each candidate, by ID. Without page variants that is :meth:`select`'s winner for
        all. When the group holds different pages of one work, each page keeps its best copy: the other candidates
        go with the page they are closest to along ``edges``, and each page's candidates are ranked separately."""
        pages = self.pages(candidates)
        if not pages:
            winner = self.select(candidates).id
            return {rec.id: winner for rec in candidates}
        graph: dict[bytes, list[tuple[float, bytes]]] = {}
        for a, b, d in edges:
            graph.setdefault(a, []).append((d, b))
            graph.setdefault(b, []).append((d, a))
        # Multi-source shortest paths from the page files; ties go to the smaller page key.
        label = dict(pages)
        best = {i: 0.0 for i in pages}
        heap = [(0.0, wp, i) for i, wp in pages.items()]
        heapq.heapify(heap)
        while heap:
            d, wp, i = heapq.heappop(heap)
            if d > best[i] or label[i] != wp:
                continue
            for w, j in graph.get(i, ()):
                if j in pages:
                    continue
                if (d + w, wp) < (best.get(j, float("inf")), label.get(j, wp)):
                    best[j] = d + w
                    label[j] = wp
                    heapq.heappush(heap, (d + w, wp, j))
        split: dict[tuple[str, str] | None, list[ImageRecord]] = {}
        for rec in candidates:
            split.setdefault(label.get(rec.id), []).append(rec)  # None: not connected to any page
        out = {}
        for part in split.values():
            winner = self.select(part).id
            out.update((rec.id, winner) for rec in part)
        return out


def _parse(data: dict[str, Any], source: str) -> dict[str, Policy]:
    policies = data.get("policy")
    if not isinstance(policies, dict):
        raise PolicyError(f"{source}: expected [policy.<name>] tables")
    return {name: Policy.from_dict(name, raw) for name, raw in policies.items()}


def builtin_policies() -> dict[str, Policy]:
    text = (resources.files(__package__) / "policies.toml").read_text(encoding="utf-8")
    return _parse(tomllib.loads(text), "built-in policies.toml")


def load_policies(user_file: str | None = None) -> dict[str, Policy]:
    """Built-in policies, with those from ``user_file`` (if given) added or overriding same-named ones."""
    policies = builtin_policies()
    if user_file is not None:
        try:
            with open(user_file, "rb") as f:
                data = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise PolicyError(f"{user_file}: {e}") from e
        policies.update(_parse(data, user_file))
    return policies
