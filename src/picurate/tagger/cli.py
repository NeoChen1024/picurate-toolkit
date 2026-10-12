"""``picurate-tag``: Danbooru-tag captions with the PixAI tagger.

* ``run``: tag a directory and write captions next to its images in one go (no database).
* ``update``: tag a project's images into its ``tags.sqlite``; resumable, incremental by mtime.
* ``captions``: record how the project's captions are rendered from the scores and combined with imported ones.
* ``export``: write the project's captions next to the source images (or into one JSON file).
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import click
import torch
import tqdm
from blake3 import blake3
from click.core import ParameterSource

from ..common.files import write_text_atomic
from ..common.images import ImageRecord, Item, Roots, standalone_items, standalone_roots
from ..common.log import setup_logging
from ..common.pipeline import run_pipeline
from ..project.captions import COMBINE_MODES, TAGS, settings
from ..project.project import Project
from .captions import (
    META_MODES,
    RESOLUTION_MODES,
    STYLES,
    Caption,
    Captioner,
    CaptionOptions,
    Vocabulary,
    append_tags,
    normalize,
    parse_replacements,
    split_tag_list,
)
from .model import default_repo_id
from .tag_db import Binding, TagDB, TagDBMismatchError, default_score_floor
from .tagger import PRECISIONS, PendingProbs, Tagger

logger = logging.getLogger(__name__)

EXISTING_MODES = ("overwrite", "skip", "append")
_DEFAULT_ALWAYS_FIRST = {"anima": "", "sdxl": "1girl,1boy"}
SIDECAR_BATCH = 500


# -- helpers ---------------------------------------------------------------------------------------------------------


def _mtime(item: Item, roots: Roots) -> float | None:
    try:
        return os.path.getmtime(item.file(roots))
    except OSError:
        return None


def _caption_path(image_dir: str, rel: str, extension: str) -> str:
    return os.path.splitext(os.path.join(image_dir, rel))[0] + extension


def _write_json(path: str, results: dict[str, dict]) -> None:
    """``results``: image path -> {tags, image_size?}, as a .json object or .jsonl lines."""
    if path.endswith(".jsonl"):
        rows = [
            {
                "image_path": image,
                "caption": entry["tags"],
                **({"image_size": entry["image_size"]} if "image_size" in entry else {}),
            }
            for image, entry in results.items()
        ]
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    else:
        text = json.dumps(results, ensure_ascii=False, indent=4)
    write_text_atomic(path, text)
    logger.info("Captions saved to %s", path)


def _print_frequency(counts: Counter[str]) -> None:
    click.echo("Tag frequencies:")
    for tag, n in counts.most_common():
        click.echo(f"{tag}: {n}")


class CaptionWriter:
    """Writes captions as files next to the images, or collects them for one JSON/JSONL file at the end."""

    def __init__(self, image_dir: str, *, existing: str, extension: str, separator: str, output_path: str | None, debug: bool):
        self.image_dir = image_dir
        self.existing = existing
        self.extension = extension
        self.separator = separator
        self.output_path = output_path
        self.debug = debug
        self.results: dict[str, dict] = {}
        self.counts: Counter[str] = Counter()

    def has_caption(self, rel: str) -> bool:
        return self.output_path is None and os.path.exists(_caption_path(self.image_dir, rel, self.extension))

    def write(self, rec: ImageRecord, caption: Caption) -> None:
        path = os.path.join(self.image_dir, rec.path)
        if self.output_path:
            self.results[path] = {"tags": caption.text, "image_size": [rec.width, rec.height]}
        else:
            txt = _caption_path(self.image_dir, rec.path, self.extension)
            text = caption.text
            if self.existing == "append" and os.path.exists(txt):
                with open(txt, encoding="utf-8") as f:
                    text = append_tags(f.read(), caption.entries, self.separator)
            write_text_atomic(txt, text + "\n")
        self.counts.update(caption.entries)
        if self.debug:
            logger.info("%s:", path)
            for cat, items in caption.scored.items():
                if items:
                    logger.info("  %-9s %s", cat, ", ".join(f"{t} {p:.2f}" for t, p in items))
            logger.info("  caption   %s", caption.text)

    def close(self, frequency: bool) -> None:
        if self.output_path:
            _write_json(self.output_path, self.results)
        if frequency:
            _print_frequency(self.counts)


def _interrupted(what: str) -> click.exceptions.Exit:
    logger.warning("Interrupted. %s", what)
    return click.exceptions.Exit(130)


def _project(project_dir: str, lock: bool = True) -> Project:
    """Open the project; with ``lock``, hold its tag database for this command."""
    try:
        project = Project(project_dir)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    click.get_current_context().call_on_close(project.close)
    if lock:
        try:
            project.lock("tags")
        except ValueError as e:
            raise click.ClickException(str(e)) from e
    return project


# -- shared options --------------------------------------------------------------------------------------------------


def _options(*decorators: Callable) -> Callable:
    def apply(f: Callable) -> Callable:
        for d in reversed(decorators):
            f = d(f)
        return f

    return apply


def _threshold_help(category: str, default: str) -> str:
    return f"Minimum confidence for {category} tags (default: {default}). Above 1 disables them."


image_dir_arg = click.argument("image_dir", type=click.Path(exists=True, file_okay=False))
project_dir_arg = click.argument("project_dir", type=click.Path(exists=True, file_okay=False))
recursive_option = click.option("--recursive", "-r", is_flag=True, help="Also include images in subdirectories.")
hidden_option = click.option(
    "--include-hidden", is_flag=True, help="Also include hidden (dot) files and directories, which are skipped by default."
)

render_options = _options(
    click.option("--style", type=click.Choice(STYLES), default="anima", show_default=True,
                 help="anima: [prefix/meta/safety] [1girl..] [character] [series] [@artist] [general]. "
                      "sdxl: sd-scripts' WD14 output with --remove_underscore --character_tags_first --character_tag_expand "
                      "--use_rating_tags_as_last_tag --always_first_tags 1girl,1boy."),
    click.option("--caption-separator", default=", ", show_default=True, help="Separator between tags."),
    click.option("--threshold", "-t", type=float, default=None,
                 help="Minimum confidence for general/character/copyright/meta tags, unless set per category. Default: the checkpoint's per-category values."),
    click.option("--general-threshold", type=float, default=None, help=_threshold_help("general", "checkpoint, 0.17")),
    click.option("--character-threshold", type=float, default=None, help=_threshold_help("character", "checkpoint, 0.27")),
    click.option("--copyright-threshold", type=float, default=None, help=_threshold_help("copyright (series)", "checkpoint, 0.24")),
    click.option("--meta-threshold", type=float, default=None, help=_threshold_help("meta", "checkpoint, 0.17")),
    click.option("--artist-threshold", type=float, default=None,
                 help="Minimum confidence for artist tags; not affected by --threshold. Default: the checkpoint's value, 0.15. Above 1 disables them."),
    click.option("--no-artist", is_flag=True, help="Never emit artist tags (same as --artist-threshold 2)."),
    click.option("--override-artist-tag", default=None,
                 help="Comma-separated artist(s) written for every image instead of whatever the model predicts, e.g. 'nnn yryr' "
                      "(anima writes '@nnn yryr'; a leading @ is optional)."),
    click.option("--rating/--no-rating", default=True, show_default=True,
                 help="Emit the most likely rating (anima: safe/sensitive/nsfw/explicit; sdxl: general/sensitive/questionable/explicit)."),
    click.option("--meta", "meta_mode", type=click.Choice(META_MODES), default="filtered", show_default=True,
                 help="Meta tags: only those describing the image (official art, jpeg artifacts, *_(medium), ...), every one the model predicts (including Danbooru bookkeeping), or none."),
    click.option("--resolution", "resolution_mode", type=click.Choice(RESOLUTION_MODES), default="size", show_default=True,
                 help="lowres/highres/absurdres: computed from the image's pixel size by Danbooru's rules, as predicted by the model, or never."),
    click.option("--prefix", default="", help="Comma-separated tags put first, verbatim (anima: quality/year tags, e.g. 'masterpiece, best quality, year 2025')."),
    click.option("--always-first-tags", default=None,
                 help="Comma-separated tags moved to the front (after --prefix) when present, one at a time like sd-scripts, so the last listed ends up first. "
                      "Default: '1girl,1boy' for sdxl, none for anima."),
    click.option("--undesired-tags", default="", help="Comma-separated tags never to emit (either spelling: 'simple background' or 'simple_background')."),
    click.option("--tag-replacement", default=None,
                 help="'source,target;source2,target2' (escape literal , and ; with \\). Applied to Danbooru names before formatting; an empty target removes the tag."),
)  # fmt: skip

run_options = _options(
    click.option("--existing", type=click.Choice(EXISTING_MODES), default="overwrite", show_default=True,
                 help="What to do with an image that already has a caption file: replace it, leave it alone, or add the new tags it lacks."),
    click.option("--caption-extension", default=".txt", show_default=True, help="Extension of the caption file written next to each image."),
    click.option("--output-path", "-o", type=click.Path(dir_okay=False), default=None,
                 help="Write all captions to this .json (path -> {tags, image_size}) or .jsonl file instead of caption files."),
    click.option("--frequency", is_flag=True, help="Print how often each emitted tag occurred."),
    click.option("--debug", is_flag=True, help="Log each image's tags per category with confidences."),
)  # fmt: skip

model_options = _options(
    click.option("--model", "-m", default=default_repo_id, show_default=True, help="Hugging Face repo id, or a local directory with config.json + model.safetensors."),
    click.option("--revision", default=None, help="Hub revision (default: the snapshot this implementation was verified against, for the default model)."),
    click.option("--device", "-c", default="cuda" if torch.cuda.is_available() else "cpu", show_default=True),
    click.option("--dtype", type=click.Choice(tuple(PRECISIONS)), default=None, help="Model precision (default: fp16 on CUDA, fp32 otherwise)."),
    click.option("--compile", "compile_", is_flag=True, help="torch.compile for a fixed batch shape: ~1.5x faster on an RTX 4080 SUPER, slower start."),
    click.option("--batch-size", "-b", type=click.IntRange(1), default=4, show_default=True,
                 help="Images per forward pass. Speed barely depends on it; VRAM ~1 GiB + 0.15 GiB per image in fp16."),
    click.option("--workers", "-j", type=int, default=None, help="Image decoder processes (default: number of CPUs)."),
)  # fmt: skip

verbose_option = click.option("--verbose", "-v", is_flag=True)
RENDER_PARAMS = (
    "style", "caption_separator", "threshold", "general_threshold", "character_threshold", "copyright_threshold",
    "meta_threshold", "artist_threshold", "no_artist", "override_artist_tag", "rating", "meta_mode", "resolution_mode",
    "prefix", "always_first_tags", "undesired_tags", "tag_replacement",
)  # fmt: skip


def _render_options(kw: dict) -> CaptionOptions:
    """Pops the caption rendering options out of ``kw``."""
    threshold = kw.pop("threshold")
    thresholds = {c: kw.pop(f"{c}_threshold") for c in ("general", "character", "copyright", "meta")}
    thresholds = {c: v if v is not None else threshold for c, v in thresholds.items()}
    artist = kw.pop("artist_threshold")
    thresholds["artist"] = 2.0 if kw.pop("no_artist") else artist
    override_artist = kw.pop("override_artist_tag")
    override_artists = None if override_artist is None else [normalize(a.lstrip("@")) for a in split_tag_list(override_artist)]
    style = kw.pop("style")
    always_first = kw.pop("always_first_tags")
    try:
        return CaptionOptions(
            style=style,
            thresholds={k: v for k, v in thresholds.items() if v is not None},
            rating=kw.pop("rating"),
            meta=kw.pop("meta_mode"),
            resolution=kw.pop("resolution_mode"),
            prefix=split_tag_list(kw.pop("prefix")),
            always_first=split_tag_list(_DEFAULT_ALWAYS_FIRST[style] if always_first is None else always_first),
            override_artists=override_artists or None,
            undesired=frozenset(normalize(t) for t in split_tag_list(kw.pop("undesired_tags"))),
            replacements=parse_replacements(kw.pop("tag_replacement")),
            separator=kw.pop("caption_separator"),
        )
    except ValueError as e:
        raise click.UsageError(str(e)) from e


def _output_path(value: str | None) -> str | None:
    if value and not value.endswith((".json", ".jsonl")):
        raise click.BadParameter("must end in .json or .jsonl", param_hint="--output-path")
    return value


def _tagger(kw: dict) -> Tagger:
    return Tagger(
        kw.pop("model"),
        revision=kw.pop("revision"),
        device=kw.pop("device"),
        dtype=kw.pop("dtype"),
        compile=kw.pop("compile_"),
        batch_size=kw["batch_size"],
    )


def _log_thresholds(captioner: Captioner) -> None:
    logger.info("Thresholds: %s", ", ".join(f"{k} {v:g}" for k, v in captioner.thresholds.items()))


def _below_floor(captioner: Captioner, floor: float) -> dict[str, float]:
    return {c: t for c, t in captioner.thresholds.items() if t < floor}


# -- commands --------------------------------------------------------------------------------------------------------


@click.group(context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 132})
def cli() -> None:
    """Danbooru-tag captions with the PixAI tagger: for a directory in one go, or for a project from stored scores."""


@cli.command()
@image_dir_arg
@recursive_option
@hidden_option
@render_options
@run_options
@model_options
@verbose_option
def run(image_dir: str, recursive: bool, include_hidden: bool, verbose: bool, **kw) -> None:
    """Tag the images in IMAGE_DIR and write captions in one go, without a database.

    Caption files are written as each batch finishes, each under a temporary name renamed into place, so after an
    interruption '--existing skip' carries on and no caption is left half-written. A JSON output is only written at
    the end. For a project, use 'update', 'captions' and 'export'.
    """
    setup_logging(verbose)
    options = _render_options(kw)
    writer = CaptionWriter(
        image_dir,
        existing=kw.pop("existing"),
        extension=kw.pop("caption_extension"),
        separator=options.separator,
        output_path=_output_path(kw.pop("output_path")),
        debug=kw.pop("debug"),
    )
    frequency = kw.pop("frequency")
    images = standalone_items(image_dir, recursive=recursive, include_hidden=include_hidden)
    if writer.existing == "skip":
        images = [i for i in images if not writer.has_caption(i.path)]
    logger.info("%d images to tag", len(images))
    if not images:
        return

    tagger = _tagger(kw)
    captioner = Captioner(Vocabulary.from_config(tagger.config), options)
    _log_thresholds(captioner)

    def finish(records: list[ImageRecord], pending: PendingProbs) -> None:
        for rec, probs in zip(records, tagger.collect(pending)):
            writer.write(rec, captioner.caption(probs, rec.width, rec.height))

    with tqdm.tqdm(total=len(images), desc="Tagging", unit="image") as bar:
        try:
            done = run_pipeline(
                standalone_roots(image_dir), images, tagger.get_preprocessor(), tagger.submit, finish, progress=bar.update, **kw
            )
        except KeyboardInterrupt:
            hint = "The JSON output was not written." if writer.output_path else "Rerun with --existing skip to continue."
            raise _interrupted(hint) from None
        finally:
            tagger.close()
    writer.close(frequency)
    logger.info("Tagged %d of %d images", done, len(images))


@cli.command()
@project_dir_arg
@click.option(
    "--force-update",
    "-f",
    is_flag=True,
    help="Re-tag every image, ignoring modification times. Also rebinds the database to the requested model/dtype/floor, "
    "or rebuilds one of another schema version.",
)
@click.option(
    "--score-floor",
    type=click.FloatRange(0, 1),
    default=default_score_floor,
    show_default=True,
    help="Store every tag scoring at least this (ratings always). Captions cannot use a lower threshold; part of the database binding.",
)
@model_options
@verbose_option
def update(project_dir: str, force_update: bool, score_floor: float, verbose: bool, **kw) -> None:
    """Tag the new and changed images of the project at PROJECT_DIR into its tags.sqlite.

    Images that are excluded (a duplicate a decided group does not keep, excluded by a person, in the trash) or
    missing are not tagged; their rows are kept, and rows of images no longer in the project are dropped. Every batch
    is committed as it finishes, and images whose modification time matches their row are skipped, so an interrupted
    update (Ctrl-C, crash, power loss) resumes where it stopped when run again.
    """
    setup_logging(verbose)
    project = _project(project_dir)
    roots = {source: str(path) for source, path in project.sources.items()}
    images = [
        Item(state["item_id"], project.items[state["item_id"]]["source_id"], state["image"])
        for state in project.states(captions=False)
        if not state["excluded"] and not project.items[state["item_id"]]["missing"]
    ]
    try:
        store = TagDB(str(project.directory / TAGS), rebuild=force_update)
    except TagDBMismatchError as e:
        raise click.ClickException(str(e)) from e
    with store:
        index = store.load_index()
        gone = [item for item in index if item not in project.items]
        if gone:
            logger.info("Removing %d database entries of images no longer in the project", len(gone))
            store.delete(gone)
        candidates = images if force_update else [i for i in images if index.get(i.id) != _mtime(i, roots)]
        logger.info(
            "%d images to take part, %d already tagged, %d to tag", len(images), len(images) - len(candidates), len(candidates)
        )
        if not candidates:
            return

        tagger = _tagger(kw)
        vocab = Vocabulary.from_config(tagger.config)
        try:
            store.bind(Binding(tagger.model_id, tagger.dtype_name, score_floor), vocab, reset_on_mismatch=force_update)
        except TagDBMismatchError as e:
            tagger.close()
            raise click.ClickException(str(e)) from e
        too_low = _below_floor(Captioner(vocab, settings(store)[0]), score_floor)
        if too_low:
            logger.warning(
                "The recorded caption thresholds %s are below the score floor %g: tags between them are not stored",
                ", ".join(f"{c} {t:g}" for c, t in too_low.items()),
                score_floor,
            )

        stored = 0

        def finish(records: list[ImageRecord], pending: PendingProbs) -> None:
            nonlocal stored
            store.store(zip(records, tagger.collect(pending)), vocab)
            stored += len(records)

        with tqdm.tqdm(total=len(candidates), desc="Tagging", unit="image") as bar:
            try:
                run_pipeline(roots, candidates, tagger.get_preprocessor(), tagger.submit, finish, progress=bar.update, **kw)
            except KeyboardInterrupt:
                raise _interrupted(
                    f"{stored} images stored this run, {len(store)} in the database; run the same command again to resume."
                ) from None
            finally:
                tagger.close()
        logger.info("Stored %d images; %d in the database", stored, len(store))


@cli.command()
@project_dir_arg
@render_options
@click.option(
    "--combine",
    type=click.Choice(COMBINE_MODES),
    default="skip",
    show_default=True,
    help="With an imported caption: keep it (skip), use the tags (overwrite), add the tags it lacks compared exactly "
    "(append) or compared ignoring case, underscores and escapes (merge).",
)
@click.option("--preview", type=click.IntRange(0), default=5, show_default=True, help="Show the captions of this many images.")
@verbose_option
@click.pass_context
def captions(ctx: click.Context, project_dir: str, combine: str, preview: int, verbose: bool, **kw) -> None:
    """Record how the captions of the project at PROJECT_DIR are rendered from its tag scores and combined with
    imported captions, and show a few. No model, no GPU: the captions follow right away, in the editor and its export
    too, except where a person edited them.

    The options given replace everything recorded before; options left out take their defaults. Without any, the
    recorded options are only shown.
    """
    setup_logging(verbose)
    given = any(ctx.get_parameter_source(name) == ParameterSource.COMMANDLINE for name in (*RENDER_PARAMS, "combine"))
    project = _project(project_dir, lock=given)
    path = project.directory / TAGS
    options = _render_options(kw)
    if given:
        try:
            store = TagDB(str(path))
        except TagDBMismatchError as e:
            raise click.ClickException(str(e)) from e
        with store:
            if store.score_floor is not None:
                too_low = _below_floor(Captioner(store.vocabulary(), options), store.score_floor)
                if too_low:
                    raise click.UsageError(
                        f"The database only holds scores >= {store.score_floor:g} (its --score-floor); thresholds below that "
                        "would miss tags: " + ", ".join(f"{c} {t:g}" for c, t in too_low.items())
                    )
            store.set_caption_options({"options": options.to_json(), "combine": combine})
    elif path.exists():
        try:
            with TagDB(str(path), readonly=True) as store:
                options, combine = settings(store)
        except TagDBMismatchError as e:
            raise click.ClickException(str(e)) from e
    else:
        options, combine = settings(None)
    click.echo(f"combine: {combine}")
    for key, value in options.to_json().items():
        click.echo(f"{key}: {json.dumps(value, ensure_ascii=False)}")
    shown = [state["item_id"] for state in project.states(captions=False) if not state["excluded"]][:preview]
    for item in shown:
        state = project.state(item)
        stale = ", tags of an earlier version of the image" if state["tags_stale"] else ""
        click.echo(
            f"\n{project.original_path(item)} [{state['caption_source'] or 'no caption'}{stale}]\n{state['caption'].rstrip()}"
        )


@cli.command()
@project_dir_arg
@click.option("--dry-run", "-n", is_flag=True, help="List what would be written without writing anything.")
@click.option("--force", is_flag=True, help="Also replace caption files changed or created outside the project.")
@click.option(
    "--output-path",
    "-o",
    type=click.Path(dir_okay=False),
    default=None,
    help="Write all captions to this .json (path -> {tags}) or .jsonl file instead of caption files.",
)
@click.option("--frequency", is_flag=True, help="Print how often each caption entry occurs.")
@verbose_option
def export(project_dir: str, dry_run: bool, force: bool, output_path: str | None, frequency: bool, verbose: bool) -> None:
    """Write the captions of the project at PROJECT_DIR as .txt files next to its source images, for training on the
    source folders directly. Excluded and missing images and empty captions are left out.

    An existing caption file is replaced only when it holds the caption imported from it or the one the last export
    wrote; anything else was edited outside the project and is skipped and listed, unless --force. Each file is
    written under a temporary name and renamed into place.
    """
    setup_logging(verbose)
    output_path = _output_path(output_path)
    project = _project(project_dir)
    path = project.directory / TAGS
    try:
        store = TagDB(str(path)) if path.exists() or not (dry_run or output_path) else None
    except TagDBMismatchError as e:
        raise click.ClickException(str(e)) from e
    written = unchanged = 0
    stale, skipped, results = [], [], {}
    counts: Counter[str] = Counter()
    recorded = store.sidecars() if store is not None else {}
    pending: list[tuple[bytes, bytes]] = []
    try:
        states = [s for s in project.states() if not s["excluded"] and not project.items[s["item_id"]]["missing"]]
        for state in tqdm.tqdm(states, desc="Exporting", unit="image", disable=dry_run):
            item, caption = state["item_id"], state["caption"]
            if not caption.strip():
                continue
            image = project.original_path(item)
            if state["tags_stale"]:
                stale.append(image)
            counts.update(e.strip() for e in caption.replace("\n", ",").split(",") if e.strip())
            if output_path:
                results[str(image)] = {"tags": caption}
                continue
            target = image.with_suffix(".txt")
            text = caption if caption.endswith("\n") else caption + "\n"
            data = text.encode("utf-8")
            if target.exists():
                existing = target.read_bytes()
                if existing == data:
                    unchanged += 1
                    if recorded.get(item) != blake3(existing).digest():
                        pending.append((item, blake3(existing).digest()))
                    continue
                ours = blake3(existing).digest() == recorded.get(item) or (
                    project.items[item]["caption"] is not None
                    and existing.decode("utf-8", "replace").removeprefix("\ufeff") == project.items[item]["caption"]
                )
                if not ours and not force:
                    skipped.append(target)
                    continue
            if dry_run:
                click.echo(f"Would {'replace' if target.exists() else 'write'} {target}", err=True)
                written += 1
                continue
            write_text_atomic(target, text)
            written += 1
            pending.append((item, blake3(data).digest()))
            if len(pending) >= SIDECAR_BATCH:
                store.record_sidecars(pending)
                pending = []
    except KeyboardInterrupt:
        raise _interrupted(f"{written} caption files written; run the same command again to finish.") from None
    finally:
        if store is not None:
            if pending and not dry_run:
                store.record_sidecars(pending)
            store.close()
    if output_path and not dry_run:
        _write_json(output_path, results)
    for target in skipped:
        click.echo(f"Changed outside the project, skipped (--force replaces it): {target}", err=True)
    for image in stale:
        logger.warning("Caption rendered from tag scores of an earlier version of the image (run update): %s", image)
    if frequency:
        _print_frequency(counts)
    if output_path:
        click.echo(f"{'Would write' if dry_run else 'Wrote'} {len(results)} captions to {output_path}.", err=True)
    else:
        click.echo(
            f"{'Would write' if dry_run else 'Wrote'} {written} caption files, {unchanged} unchanged, {len(skipped)} skipped as "
            f"changed outside the project; {len(stale)} from tags of an earlier version of the image.",
            err=True,
        )


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
