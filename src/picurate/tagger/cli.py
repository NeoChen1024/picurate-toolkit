"""``picurate-tag``: Danbooru-tag captions for a directory of images with the PixAI tagger.

* ``run``: tag and write captions in one go (no database).
* ``update``: tag into a SQLite score database; resumable, incremental by mtime.
* ``export``: render captions from that database, no GPU, in any style/thresholds.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from collections.abc import Callable

import click
import torch
import tqdm

from ..common.images import ImageRecord, is_image_path, list_directory_relative
from ..common.log import setup_logging
from ..common.pipeline import run_pipeline
from .captions import (
    META_MODES,
    RESOLUTION_MODES,
    STYLES,
    Caption,
    Captioner,
    CaptionOptions,
    Vocabulary,
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


# -- helpers ---------------------------------------------------------------------------------------------------------


def _list_images(image_dir: str, recursive: bool, include_hidden: bool) -> list[str]:
    rels = list_directory_relative(image_dir, recursive=recursive, include_hidden=include_hidden)
    return sorted(r for r in rels if is_image_path(r))


def _caption_path(image_dir: str, rel: str, extension: str) -> str:
    return os.path.splitext(os.path.join(image_dir, rel))[0] + extension


def _merge_existing(path: str, entries: list[str], separator: str) -> list[str]:
    """Existing caption entries first, then the new ones it lacks (what sd-scripts' ``--append_tags`` does)."""
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read().strip()
    except FileNotFoundError:
        return entries
    existing = [t.strip() for t in content.split(separator.strip()) if t.strip()]
    have = set(existing)
    return existing + [t for t in entries if t not in have]


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
            entries = _merge_existing(txt, caption.entries, self.separator) if self.existing == "append" else caption.entries
            with open(txt, "w", encoding="utf-8") as f:
                f.write(self.separator.join(entries) + "\n")
        self.counts.update(caption.entries)
        if self.debug:
            logger.info("%s:", path)
            for cat, items in caption.scored.items():
                if items:
                    logger.info("  %-9s %s", cat, ", ".join(f"{t} {p:.2f}" for t, p in items))
            logger.info("  caption   %s", caption.text)

    def close(self, frequency: bool) -> None:
        if self.output_path:
            with open(self.output_path, "w", encoding="utf-8") as f:
                if self.output_path.endswith(".jsonl"):
                    for path, entry in self.results.items():
                        f.write(
                            json.dumps(
                                {"image_path": path, "caption": entry["tags"], "image_size": entry["image_size"]},
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                else:
                    json.dump(self.results, f, ensure_ascii=False, indent=4)
            logger.info("Captions saved to %s", self.output_path)
        if frequency:
            click.echo("Tag frequencies:")
            for tag, n in self.counts.most_common():
                click.echo(f"{tag}: {n}")


def _interrupted(what: str) -> click.exceptions.Exit:
    logger.warning("Interrupted. %s", what)
    return click.exceptions.Exit(130)


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
recursive_option = click.option("--recursive", "-r", is_flag=True, help="Also include images in subdirectories.")
hidden_option = click.option(
    "--include-hidden", is_flag=True, help="Also include hidden (dot) files and directories, which are skipped by default."
)

caption_options = _options(
    click.option("--style", type=click.Choice(STYLES), default="anima", show_default=True,
                 help="anima: [prefix/meta/safety] [1girl..] [character] [series] [@artist] [general]. "
                      "sdxl: sd-scripts' WD14 output with --remove_underscore --character_tags_first --character_tag_expand "
                      "--use_rating_tags_as_last_tag --always_first_tags 1girl,1boy."),
    click.option("--existing", type=click.Choice(EXISTING_MODES), default="overwrite", show_default=True,
                 help="What to do with an image that already has a caption file: replace it, leave it alone, or add the new tags it lacks."),
    click.option("--caption-extension", default=".txt", show_default=True, help="Extension of the caption file written next to each image."),
    click.option("--caption-separator", default=", ", show_default=True, help="Separator between tags."),
    click.option("--output-path", "-o", type=click.Path(dir_okay=False), default=None,
                 help="Write all captions to this .json (path -> {tags, image_size}) or .jsonl file instead of caption files."),
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


def _caption_setup(image_dir: str, kw: dict) -> tuple[CaptionOptions, CaptionWriter, bool]:
    """Pops the caption options out of ``kw`` and builds the options and the writer from them."""
    output_path = kw.pop("output_path")
    if output_path and not output_path.endswith((".json", ".jsonl")):
        raise click.BadParameter("must end in .json or .jsonl", param_hint="--output-path")
    threshold = kw.pop("threshold")
    thresholds = {c: kw.pop(f"{c}_threshold") for c in ("general", "character", "copyright", "meta")}
    thresholds = {c: v if v is not None else threshold for c, v in thresholds.items()}
    artist = kw.pop("artist_threshold")
    thresholds["artist"] = 2.0 if kw.pop("no_artist") else artist
    override_artist = kw.pop("override_artist_tag")
    override_artists = None if override_artist is None else [normalize(a.lstrip("@")) for a in split_tag_list(override_artist)]
    style = kw.pop("style")
    always_first = kw.pop("always_first_tags")
    separator = kw.pop("caption_separator")
    try:
        options = CaptionOptions(
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
            separator=separator,
        )
    except ValueError as e:
        raise click.UsageError(str(e)) from e
    writer = CaptionWriter(
        image_dir,
        existing=kw.pop("existing"),
        extension=kw.pop("caption_extension"),
        separator=separator,
        output_path=output_path,
        debug=kw.pop("debug"),
    )
    return options, writer, kw.pop("frequency")


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


# -- commands --------------------------------------------------------------------------------------------------------


@click.group(context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 132})
def cli() -> None:
    """Danbooru-tag captions with the PixAI tagger, written next to each image."""


@cli.command()
@image_dir_arg
@recursive_option
@hidden_option
@caption_options
@model_options
@verbose_option
def run(image_dir: str, recursive: bool, include_hidden: bool, verbose: bool, **kw) -> None:
    """Tag the images in IMAGE_DIR and write captions in one go, without a database.

    Caption files are written as each batch finishes, so after an interruption '--existing skip' carries on. A JSON
    output is only written at the end. For large or repeated jobs prefer 'update' + 'export'.
    """
    setup_logging(verbose)
    options, writer, frequency = _caption_setup(image_dir, kw)
    images = _list_images(image_dir, recursive, include_hidden)
    if writer.existing == "skip":
        images = [r for r in images if not writer.has_caption(r)]
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
            done = run_pipeline(image_dir, images, tagger.get_preprocessor(), tagger.submit, finish, progress=bar.update, **kw)
        except KeyboardInterrupt:
            hint = "The JSON output was not written." if writer.output_path else "Rerun with --existing skip to continue."
            raise _interrupted(hint) from None
        finally:
            tagger.close()
    writer.close(frequency)
    logger.info("Tagged %d of %d images", done, len(images))


@cli.command()
@image_dir_arg
@click.option(
    "--db", "-d", type=click.Path(dir_okay=False), required=True, help="SQLite score database for IMAGE_DIR (created if missing)."
)
@recursive_option
@hidden_option
@click.option(
    "--force-update",
    "-f",
    is_flag=True,
    help="Re-tag every image, ignoring modification times. Also rebinds the database to the requested model/dtype/floor.",
)
@click.option(
    "--clean-orphans/--no-clean-orphans",
    default=True,
    show_default=True,
    help="Remove database entries whose image no longer exists.",
)
@click.option(
    "--score-floor",
    type=click.FloatRange(0, 1),
    default=default_score_floor,
    show_default=True,
    help="Store every tag scoring at least this (ratings always). Exports cannot use a lower threshold; part of the database binding.",
)
@model_options
@verbose_option
def update(
    image_dir: str,
    db: str,
    recursive: bool,
    include_hidden: bool,
    force_update: bool,
    clean_orphans: bool,
    score_floor: float,
    verbose: bool,
    **kw,
) -> None:
    """Tag new and changed images of IMAGE_DIR into a score database.

    Every batch is committed as it finishes, and images whose modification time matches their row are skipped, so an
    interrupted update (Ctrl-C, crash, power loss) resumes where it stopped when run again.
    """
    setup_logging(verbose)
    images = _list_images(image_dir, recursive, include_hidden)
    with TagDB(db) as store:
        index = store.load_index()
        if clean_orphans:
            listed = set(images)
            orphans = [p for p in index if p not in listed and not os.path.exists(os.path.join(image_dir, p))]
            if orphans:
                logger.info("Removing %d database entries whose image is gone", len(orphans))
                store.delete(orphans)
        candidates = images if force_update else [r for r in images if index.get(r) != os.path.getmtime(os.path.join(image_dir, r))]
        logger.info("%d images, %d already tagged, %d to tag", len(images), len(images) - len(candidates), len(candidates))
        if not candidates:
            return

        tagger = _tagger(kw)
        vocab = Vocabulary.from_config(tagger.config)
        try:
            store.bind(Binding(tagger.model_id, tagger.dtype_name, score_floor), vocab, reset_on_mismatch=force_update)
        except TagDBMismatchError as e:
            tagger.close()
            raise click.ClickException(str(e)) from e

        stored = 0

        def finish(records: list[ImageRecord], pending: PendingProbs) -> None:
            nonlocal stored
            store.store(zip(records, tagger.collect(pending)), vocab)
            stored += len(records)

        with tqdm.tqdm(total=len(candidates), desc="Tagging", unit="image") as bar:
            try:
                run_pipeline(image_dir, candidates, tagger.get_preprocessor(), tagger.submit, finish, progress=bar.update, **kw)
            except KeyboardInterrupt:
                raise _interrupted(
                    f"{stored} images stored this run, {len(store)} in the database; run the same command again to resume."
                ) from None
            finally:
                tagger.close()
        logger.info("Stored %d images; %d in the database", stored, len(store))


@cli.command()
@image_dir_arg
@click.option("--db", "-d", type=click.Path(exists=True, dir_okay=False), required=True, help="Score database written by 'update'.")
@caption_options
@verbose_option
def export(image_dir: str, db: str, verbose: bool, **kw) -> None:
    """Write captions for the images of IMAGE_DIR from a score database (no model, no GPU).

    Images that changed on disk since they were tagged are skipped with a warning: run 'update' first.
    """
    setup_logging(verbose)
    options, writer, frequency = _caption_setup(image_dir, kw)
    with TagDB(db, create=False) as store:
        vocab = store.vocabulary()
        floor = store.score_floor or 0.0
        captioner = Captioner(vocab, options)
        too_low = {c: t for c, t in captioner.thresholds.items() if t < floor}
        if too_low:
            raise click.UsageError(
                f"The database only holds scores >= {floor:g} (its --score-floor); thresholds below that would miss tags: "
                + ", ".join(f"{c} {t:g}" for c, t in too_low.items())
            )
        _log_thresholds(captioner)
        written = stale = missing = skipped = 0
        with tqdm.tqdm(total=len(store), desc="Exporting", unit="image") as bar:
            for rec, probs in store.iter_scores(len(vocab.tags)):
                bar.update(1)
                try:
                    mtime = os.path.getmtime(os.path.join(image_dir, rec.path))
                except OSError:
                    missing += 1
                    continue
                if mtime != rec.mtime:
                    logger.warning("Changed since it was tagged, skipped: %s", rec.path)
                    stale += 1
                    continue
                if writer.existing == "skip" and writer.has_caption(rec.path):
                    skipped += 1
                    continue
                writer.write(rec, captioner.caption(probs, rec.width, rec.height))
                written += 1
    writer.close(frequency)
    logger.info(
        "Wrote %d captions%s%s%s",
        written,
        f", skipped {skipped} with existing captions" if skipped else "",
        f", {stale} changed since tagging (run update)" if stale else "",
        f", {missing} no longer on disk" if missing else "",
    )


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
