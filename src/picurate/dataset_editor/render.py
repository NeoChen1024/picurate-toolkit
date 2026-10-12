"""Deterministic full-resolution rendering of a project's edit recipes, accepted repairs, and export transactions."""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path

from blake3 import blake3
from PIL import Image, ImageDraw, ImageOps, ImageStat

from ..common.images import file_digest
from ..project.project import EXTENSIONS, atomic_json, check_image_size, safe_relative, validate_source

logger = logging.getLogger(__name__)


def load_image(path) -> Image.Image:
    """Decode with EXIF orientation applied: crops and fills are recorded in oriented pixel coordinates."""
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def image_digest(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    return blake3(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def artifact_path(directory, name) -> Path:
    root = Path(directory).resolve()
    path = (root / safe_relative(name)).resolve()
    if root not in path.parents:
        raise ValueError("Repair artifact is outside the project")
    return path


def save_repair(directory, item_id: bytes, image: Image.Image, rect, mask: Image.Image, patch: Image.Image, metadata: dict) -> dict:
    """Store an accepted LaMa repair under the project and return its fill entry."""
    identifier = uuid.uuid4().hex
    mask_name = f"masks/{item_id.hex()}-{identifier}.png"
    patch_name = f"patches/{item_id.hex()}-{identifier}.png"
    for name, artifact in ((mask_name, mask.convert("L")), (patch_name, patch.convert("RGB"))):
        artifact.save(artifact_path(directory, name), format="PNG")
    return {
        "method": "lama",
        "rect": list(rect),
        "input_hash": image_digest(image),
        "mask": mask_name,
        "patch": patch_name,
        "mask_blake3": file_digest(artifact_path(directory, mask_name)).hex(),
        "patch_blake3": file_digest(artifact_path(directory, patch_name)).hex(),
        "metadata": metadata,
    }


def render(im: Image.Image, state: dict, crop: bool = True, project_dir=None) -> Image.Image:
    """Apply the fills in order, then the crop. Mean fills sample a ring around their rectangle that excludes every
    filled rectangle so far; LaMa fills paste their accepted patch through their mask, and refuse if an earlier edit
    changed the pixels they were computed from."""
    if state.get("operations"):
        raise ValueError("This version cannot render recorded model operations")
    check_image_size(im.size, state)
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
            if file_digest(mask_path).hex() != fill["mask_blake3"] or file_digest(patch_path).hex() != fill["patch_blake3"]:
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


def check_content(source, data: bytes, content) -> None:
    """Compare a source's bytes with ``content`` = ``[mtime, BLAKE3 hex]`` from the embedding database: other contents
    under the same mtime mean damage or a swapped file and are refused; with a new mtime, an edit, which only warns."""
    mtime, digest = content
    if blake3(data).hexdigest() != digest:
        if os.stat(source).st_mtime == mtime:
            raise ValueError(f"Source image contents changed without a new modification time (damaged?): {source}")
        logger.warning("Source image changed since it was encoded: %s", source)


def export_one(source, output, state: dict, fmt: str, output_stem: str, project_dir=None, content=None) -> list[str]:
    if warning := validate_source(source, state):
        logger.warning("%s", warning)
    if content is None:
        image = load_image(source)
    else:
        data = Path(source).read_bytes()
        check_content(source, data, content)
        image = load_image(io.BytesIO(data))
    image = render(image, state, project_dir=project_dir)
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
            files.extend(
                export_one(item["source_path"], stage, item["state"], fmt, item["output_stem"], directory, item.get("content"))
            )
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
