"""Dataset projects: a directory that references external images and owns everything derived from them.

``project.json`` holds the settings and the source folders, ``items.jsonl`` maps stable item IDs to source images
(with their fingerprint and the caption they were imported with), and ``edits.jsonl`` is an append-only log of
complete per-image states. Source images and their caption files are only ever read.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path

from ..common.images import walk_directory_relative

SCHEMA_VERSION = 2
# What a project imports, and what an export refuses to overwrite: static formats a training pipeline reads.
EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def digest(path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_relative(value) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"Invalid project-relative path: {value}")
    return path


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class Project:
    """A project directory: ``project.json``, ``items.jsonl``, ``edits.jsonl``, accepted repairs and ``output/``."""

    def __init__(self, directory, create: bool = False):
        self.directory = Path(directory).expanduser().resolve()
        config = self.directory / "project.json"
        if create:
            if self.directory.exists() and any(self.directory.iterdir()):
                raise ValueError("Choose an empty directory for a new project")
            self.directory.mkdir(parents=True, exist_ok=True)
            atomic_json(
                config,
                {
                    "schema_version": SCHEMA_VERSION,
                    "project_id": uuid.uuid4().hex,
                    "sources": {},
                    "format": "PNG",
                    "spaces": True,
                    "escape": False,
                },
            )
        if not config.exists():
            raise ValueError("No project.json found. Create a project with New Project, then import images.")
        self.settings = json.loads(config.read_text(encoding="utf-8"))
        if self.settings.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported project version {self.settings.get('schema_version')!r}; it was left untouched.")
        self.log = self.directory / "edits.jsonl"
        self.items_file = self.directory / "items.jsonl"
        self.items = {}
        if self.items_file.exists():
            for line in self.items_file.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                safe_relative(item["image"])
                safe_relative(item["output_stem"])
                self.items[item["item_id"]] = item
        self.states = {}
        if self.log.exists():
            lines = self.log.read_bytes().splitlines(keepends=True)
            offset = 0
            for i, line in enumerate(lines):
                try:
                    state = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    if i != len(lines) - 1:
                        raise ValueError(f"Corrupt JSONL line {i + 1}")
                    # A torn last line from an interrupted write: set it aside and continue from the line before.
                    recovery = self.directory / f"edits-{uuid.uuid4().hex}.recovery"
                    recovery.write_bytes(line)
                    with self.log.open("r+b") as stream:
                        stream.truncate(offset)
                    break
                if state["item_id"] not in self.items:
                    raise ValueError("Edit refers to an unknown project item")
                self.states[state["item_id"]] = state
                offset += len(line)
            if self.log.stat().st_size and not self.log.read_bytes().endswith(b"\n"):
                with self.log.open("ab") as stream:
                    stream.write(b"\n")
        for name in ("masks", "patches", "output"):
            (self.directory / name).mkdir(parents=True, exist_ok=True)
        self.items_file.touch(exist_ok=True)
        self.log.touch(exist_ok=True)
        # Recover an interrupted directory swap without discarding the prior export.
        backup = self.directory / ".output-previous"
        if backup.exists():
            output = self.directory / "output"
            if not any(output.iterdir()):
                output.rmdir()
                backup.rename(output)
            else:
                shutil.rmtree(backup)

    def save_settings(self) -> None:
        atomic_json(self.directory / "project.json", self.settings)

    def source_path(self, item_id: str) -> Path:
        item = self.items[item_id]
        return Path(self.settings["sources"][item["source_id"]]) / safe_relative(item["image"])

    def import_images(
        self, paths, *, include_hidden: bool = False, progress: Callable[[int, int], None] = lambda done, total: None
    ) -> tuple[int, int]:
        """Import directories (recursively) and individual files as one batch. Same-stem ``.txt`` captions are copied
        into the item manifest; an image whose resolved path is already in the project is skipped, keeping its edits.
        Hidden files and directories inside an imported directory are skipped unless ``include_hidden``. Every new
        image is hashed (SHA-256), so this reads all of them once. Returns ``(imported, skipped)``."""
        sources = dict(self.settings["sources"])
        items = dict(self.items)
        existing = {str(self.source_path(key).resolve()) for key in items}
        names = {item["output_stem"].casefold() for item in items.values()}
        candidates = []
        for entry in paths:
            entry = Path(entry).expanduser().resolve()
            root = entry if entry.is_dir() else entry.parent
            if entry.is_dir():
                files = sorted(entry / name for name in walk_directory_relative(str(entry), include_hidden=include_hidden))
            else:
                files = [entry]
            source_id = next((key for key, value in sources.items() if Path(value) == root), None)
            if source_id is None:
                source_id = uuid.uuid4().hex
                sources[source_id] = str(root)
            candidates += [(root, source_id, path) for path in files if path.suffix.lower() in EXTENSIONS]
        imported = skipped = 0
        for done, (root, source_id, path) in enumerate(candidates):
            progress(done, len(candidates))
            if not path.is_file():
                continue
            # Never ingest generated artifacts from this project.
            if self.directory == path.resolve() or self.directory in path.resolve().parents:
                continue
            if str(path.resolve()) in existing:
                skipped += 1
                continue
            item_id = uuid.uuid4().hex
            relative = path.relative_to(root).as_posix()
            stem = str(Path(relative).with_suffix(""))
            if stem.casefold() in names:
                stem += "__" + item_id[:12]
            while stem.casefold() in names:
                stem += "_"
            caption_path = path.with_suffix(".txt")
            try:
                caption = caption_path.read_text(encoding="utf-8-sig") if caption_path.exists() else ""
            except UnicodeDecodeError as error:
                raise ValueError(f"Caption is not UTF-8: {caption_path}") from error
            stat = path.stat()
            fingerprint = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest(path)}
            items[item_id] = {
                "item_id": item_id,
                "source_id": source_id,
                "image": relative,
                "caption_source": caption_path.relative_to(root).as_posix() if caption_path.exists() else None,
                "source": fingerprint,
                "initial_caption": caption,
                "output_stem": stem,
            }
            names.add(stem.casefold())
            existing.add(str(path.resolve()))
            imported += 1
            if imported % 1000 == 0:
                self._publish(sources, items)  # an interrupted import resumes from here: re-imports are skipped
        progress(len(candidates), len(candidates))
        self._publish(sources, items)
        return imported, skipped

    def _publish(self, sources: dict, items: dict) -> None:
        # Publishing extra source roots before the manifest is safe if interrupted.
        self.settings["sources"] = dict(sources)
        self.save_settings()
        temporary = self.items_file.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for item in items.values():
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.items_file)
        self.items = dict(items)

    def relink(self, source_id: str, root) -> None:
        """Point a source root elsewhere, after checking every affected image's SHA-256."""
        root = Path(root).expanduser().resolve()
        for item in self.items.values():
            if item["source_id"] == source_id:
                path = root / item["image"]
                if not path.is_file() or digest(path) != item["source"]["sha256"]:
                    raise ValueError(f"Missing or different source image: {path}")
        self.settings["sources"][source_id] = str(root)
        self.save_settings()

    def state(self, item_id: str) -> dict:
        if item_id in self.states:
            return copy.deepcopy(self.states[item_id])
        item = self.items[item_id]
        return {
            "schema_version": SCHEMA_VERSION,
            "item_id": item_id,
            "image": item["image"],
            "revision": 0,
            "source": copy.deepcopy(item["source"]),
            "crop": None,
            "fills": [],
            "operations": [],
            "caption": item["initial_caption"],
            "review_status": "unreviewed",
        }

    def save(self, state: dict) -> None:
        if state["item_id"] not in self.items:
            raise ValueError("Cannot save an unknown item")
        saved = copy.deepcopy(state)
        saved["revision"] += 1
        with self.log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(saved, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        state["revision"] = saved["revision"]
        self.states[state["item_id"]] = saved

    def export_snapshot(self, states, fmt: str) -> dict:
        return {
            "project_id": self.settings["project_id"],
            "format": fmt,
            "items": [
                {
                    "state": copy.deepcopy(state),
                    "source_path": str(self.source_path(state["item_id"])),
                    "output_stem": self.items[state["item_id"]]["output_stem"],
                }
                for state in states
            ],
        }


def validate_source(path, state: dict) -> None:
    stat = Path(path).stat()
    fingerprint = state["source"]
    if fingerprint.get("sha256"):
        changed = digest(path) != fingerprint["sha256"]
    else:
        changed = any(fingerprint[k] != getattr(stat, "st_" + k) for k in ("size", "mtime_ns"))
    if changed:
        raise ValueError("Source file changed; review the edit recipe before continuing")
