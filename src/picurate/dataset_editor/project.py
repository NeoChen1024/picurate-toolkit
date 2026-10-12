"""Editor projects: external image references, persistent edit recipes and deterministic full-resolution rendering.

A project owns captions and edits; the images stay where they are and are never written to. ``items.jsonl`` maps
stable item IDs to source images, ``edits.jsonl`` is an append-only log of complete edit states, and ``output/`` is
the last export, replaced as a whole by the next one.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps, ImageStat

from ..common.images import walk_directory_relative

SCHEMA_VERSION = 2
# What the editor imports and what export refuses to overwrite: static formats a training pipeline reads.
EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def load_image(path) -> Image.Image:
    """Decode with EXIF orientation applied: crops and fills are recorded in oriented pixel coordinates."""
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def image_digest(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def digest(path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_relative(value) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"Invalid project-relative path: {value}")
    return path


def artifact_path(directory, name) -> Path:
    root = Path(directory).resolve()
    path = (root / safe_relative(name)).resolve()
    if root not in path.parents:
        raise ValueError("Repair artifact is outside the project")
    return path


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def save_repair(directory, item_id: str, image: Image.Image, rect, mask: Image.Image, patch: Image.Image, metadata: dict) -> dict:
    """Store an accepted LaMa repair under the project and return its fill entry."""
    identifier = uuid.uuid4().hex
    mask_name = f"masks/{item_id}-{identifier}.png"
    patch_name = f"patches/{item_id}-{identifier}.png"
    for name, artifact in ((mask_name, mask.convert("L")), (patch_name, patch.convert("RGB"))):
        artifact.save(artifact_path(directory, name), format="PNG")
    return {
        "method": "lama",
        "rect": list(rect),
        "input_hash": image_digest(image),
        "mask": mask_name,
        "patch": patch_name,
        "mask_sha256": digest(artifact_path(directory, mask_name)),
        "patch_sha256": digest(artifact_path(directory, patch_name)),
        "metadata": metadata,
    }


def render(im: Image.Image, state: dict, crop: bool = True, project_dir=None) -> Image.Image:
    """Apply the fills in order, then the crop. Mean fills sample a ring around their rectangle that excludes every
    filled rectangle so far; LaMa fills paste their accepted patch through their mask, and refuse if an earlier edit
    changed the pixels they were computed from."""
    if state.get("operations"):
        raise ValueError("This version cannot render recorded model operations")
    result = im.copy()
    boxes = []
    for fill in state.get("fills", []):
        x, y, w, h = fill["rect"]
        if fill.get("method") == "lama":
            if project_dir is None:
                raise ValueError("A project directory is required to load accepted repairs")
            if fill["input_hash"] != image_digest(result):
                raise ValueError(
                    "Stale LaMa repair: an earlier image edit changed. Undo that edit or remove and regenerate this repair."
                )
            mask_path = artifact_path(project_dir, fill["mask"])
            patch_path = artifact_path(project_dir, fill["patch"])
            if digest(mask_path) != fill["mask_sha256"] or digest(patch_path) != fill["patch_sha256"]:
                raise ValueError("A saved repair artifact changed")
            with Image.open(mask_path) as file:
                mask = file.convert("L")
            with Image.open(patch_path) as file:
                patch = file.convert("RGB")
            if mask.size != (w, h) or patch.size != (w, h):
                raise ValueError("Repair artifact dimensions do not match the ROI")
            result.paste(patch, (x, y), mask)
            continue
        boxes.append((x, y, w, h))
        ring = fill.get("ring_px", 4)
        left, top = max(0, x - ring), max(0, y - ring)
        right, bottom = min(im.width, x + w + ring), min(im.height, y + h + ring)
        patch = result.crop((left, top, right, bottom))
        mask = Image.new("L", patch.size, 255)
        draw = ImageDraw.Draw(mask)
        for bx, by, bw, bh in boxes:
            draw.rectangle((bx - left, by - top, bx + bw - 1 - left, by + bh - 1 - top), fill=0)
        stat = ImageStat.Stat(patch, mask)
        if not stat.count[0]:
            raise ValueError("No pixels available in the fill sampling ring")
        color = tuple(round(v) for v in stat.mean)
        result.paste(color, (x, y, x + w, y + h))
    if crop and state.get("crop"):
        x, y, w, h = state["crop"]
        result = result.crop((x, y, x + w, y + h))
    return result


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

    def import_images(self, paths) -> tuple[int, int]:
        """Import directories (recursively, hidden ones skipped) and individual files as one batch. Same-stem ``.txt``
        captions are copied into the item manifest; an image whose resolved path is already in the project is skipped,
        keeping its edits. Returns ``(imported, skipped)``."""
        sources = dict(self.settings["sources"])
        items = dict(self.items)
        existing = {str(self.source_path(key).resolve()) for key in items}
        names = {item["output_stem"].casefold() for item in items.values()}
        imported = skipped = 0
        for entry in paths:
            entry = Path(entry).expanduser().resolve()
            root = entry if entry.is_dir() else entry.parent
            files = sorted(entry / name for name in walk_directory_relative(str(entry))) if entry.is_dir() else [entry]
            source_id = next((key for key, value in sources.items() if Path(value) == root), None)
            if source_id is None:
                source_id = uuid.uuid4().hex
                sources[source_id] = str(root)
            for path in files:
                if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
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
                caption = caption_path.read_text(encoding="utf-8-sig") if caption_path.exists() else ""
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
        # Publishing extra source roots before the manifest is safe if interrupted.
        self.settings["sources"] = sources
        self.save_settings()
        temporary = self.items_file.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for item in items.values():
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.items_file)
        self.items = items
        return imported, skipped

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


def export_one(source, output, state: dict, fmt: str, output_stem: str, project_dir=None) -> list[str]:
    validate_source(source, state)
    image = render(load_image(source), state, project_dir=project_dir)
    target = Path(output) / (str(safe_relative(output_stem)) + (".png" if fmt == "PNG" else ".webp"))
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, format=fmt, **({"lossless": True, "method": 4} if fmt == "WEBP" else {}))
    target.with_suffix(".txt").write_text(state["caption"], encoding="utf-8")
    return [target.relative_to(output).as_posix(), target.with_suffix(".txt").relative_to(output).as_posix()]


def export_project(directory, snapshot: dict, cancelled=lambda: False, progress=lambda value: None) -> tuple[int, bool]:
    """Build the export in a staging directory and swap it in; unmanaged files are kept, and a failure or
    cancellation leaves the previous output intact. Returns ``(exported, cancelled)``."""
    directory = Path(directory)
    output = directory / "output"
    if output.is_symlink():
        raise ValueError("The project output directory must not be a symlink")
    backup = directory / ".output-previous"
    if backup.exists():
        raise ValueError("A previous output transaction needs recovery; reopen the project")
    stage = Path(tempfile.mkdtemp(prefix=".export-", dir=directory))
    try:
        if output.exists():
            if any(path.is_symlink() for path in output.rglob("*")):
                raise ValueError("Output contains symlinks; move them out before exporting")
            shutil.copytree(output, stage, dirs_exist_ok=True)
        manifest = stage / ".manifest.json"
        if manifest.exists():
            old = json.loads(manifest.read_text(encoding="utf-8"))
            if old.get("project_id") != snapshot["project_id"]:
                raise ValueError("Output manifest belongs to another project")
            for name in old["files"]:
                (stage / safe_relative(name)).unlink(missing_ok=True)
            manifest.unlink()
        files = []
        fmt = snapshot["format"]
        if fmt not in ("PNG", "WEBP"):
            raise ValueError("Unsupported export format")
        for i, item in enumerate(snapshot["items"]):
            if cancelled():
                return 0, True
            stem = stage / safe_relative(item["output_stem"])
            # Refuse to replace unmanaged captions/images in any supported format.
            if any((stem.parent / (stem.name + ext)).exists() for ext in EXTENSIONS | {".txt"}):
                raise ValueError(f"Unmanaged output file conflicts with {item['output_stem']}")
            files.extend(export_one(item["source_path"], stage, item["state"], fmt, item["output_stem"], directory))
            progress(i + 1)
        if cancelled():
            return 0, True
        atomic_json(manifest, {**snapshot, "files": files})
        output.rename(backup)
        try:
            stage.rename(output)
        except Exception:
            backup.rename(output)
            raise
        shutil.rmtree(backup)
        return len(snapshot["items"]), False
    finally:
        if stage.exists():
            shutil.rmtree(stage)
