"""Tag completion index over DraconicDragon's merged Danbooru/e621 tag list (``name,category,count,aliases``)."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from importlib.resources import files

TAG_LIST = files(__package__) / "data" / "danbooru_e621_merged_2026-04-01_pt20-ia-dd-ed-spc.csv"

# Danbooru categories 0-5, e621 categories 7-15 as merged into the list.
CATEGORIES = {
    0: "General",
    1: "Artist",
    3: "Copyright",
    4: "Character",
    5: "Meta",
    7: "General",
    8: "Artist",
    9: "Contributor",
    10: "Copyright",
    11: "Character",
    12: "Species",
    14: "Meta",
    15: "Lore",
}


def normalize(value: str) -> str:
    return value.lower().replace("\\(", "(").replace("\\)", ")").replace("_", " ")


class TagIndex:
    """Tags and their aliases, indexed by the first three letters of every word."""

    def __init__(self, path=TAG_LIST):
        self.tags = []
        self.prefixes = defaultdict(list)
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.reader(stream):
                if len(row) != 4:
                    continue
                name, kind, count, aliases = row
                try:
                    kind, count = int(kind), int(count)
                except ValueError:
                    continue
                names = [normalize(name)] + [normalize(a) for a in aliases.split(",") if a]
                index = len(self.tags)
                self.tags.append((name, kind, count, names))
                keys = {word[:3] for n in names for word in n.split() if word}
                for key in keys:
                    self.prefixes[key].append(index)

    def search(self, segment: str) -> list[tuple]:
        """Candidates for the end of ``segment`` (the text since the last comma or newline), trying each of its last
        five words as the start of the tag so that free-form modifiers before it are kept. Returns up to 30
        ``(rank, offset, name, category, count)``, best first; ``offset`` is where the replaced text starts."""
        words = list(re.finditer(r"\S+", segment))
        found = {}
        for match in words[-5:]:
            query = normalize(segment[match.start() :]).strip()
            if len(query) < 2:
                continue
            first = query.split()[0]
            ids = self.prefixes.get(first[:3], []) if len(first) >= 3 else []
            if len(first) == 2:
                ids = (i for key, values in self.prefixes.items() if key.startswith(first) for i in values)
            for i in ids:
                name, kind, count, variants = self.tags[i]
                best = None
                for pos, variant in enumerate(variants):
                    # Match from a word boundary; retain arbitrary modifiers outside the range.
                    starts = [0] + [m.end() for m in re.finditer(" ", variant)]
                    for start in starts:
                        if variant[start:].startswith(query):
                            rank = (0 if variant == query else 1 if start == 0 else 2, pos != 0, -len(query), -count)
                            if best is None or rank < best:
                                best = rank
                if best is not None and (i not in found or best < found[i][0]):
                    found[i] = (best, match.start(), name, kind, count)
        return sorted(found.values())[:30]
