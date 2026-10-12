"""Command line interface: ``picurate-dedupe <subcommand>``."""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Callable, Generator

import click
import humanize
import numpy as np
import torch
import tqdm

from ..common.images import ImageRecord
from ..common.log import setup_logging
from .calibrate import VARIANTS, calibrate, default_variants, report
from .db_store import EmbeddingDB, load_database
from .dedupe import find_cross_pairs, find_duplicate_edges, group_edges, move_to_trash, trash_duplicate_groups
from .encoder import CLIPImageEncoder, default_model_id, default_seq_len, precision_choices
from .encoding_pipeline import update_database
from .keeping import PolicyError, load_policies
from .pixel_check import PixelVerdict, check_pairs, default_buffer_mib
from .review import ReviewDB, apply_decisions, default_review_path, store_matches
from .similarity import DistanceIndex, default_euclidean_distance_threshold, euclidean_distance

logger = logging.getLogger(__name__)

default_device = "cuda" if torch.cuda.is_available() else "cpu"


# -- shared options --------------------------------------------------------------------------------------------------


def _options(*decorators: Callable) -> Callable:
    def apply(f: Callable) -> Callable:
        for d in reversed(decorators):
            f = d(f)
        return f

    return apply


model_options = _options(
    click.option(
        "--model-id",
        "-m",
        default=default_model_id,
        show_default=True,
        help="timm model name of the image tower (any CLIP/SigLIP tower timm ships).",
    ),
    click.option("--device", "-c", default=default_device, show_default=True, help="Device to run the model and the matching on."),
    click.option(
        "--compile",
        "compile_",
        is_flag=True,
        help="torch.compile the model (dynamic shapes, 20-30 s warm-up). ~17%% faster for fixed-resolution towers, no gain for NaFlex towers yet.",
    ),
    click.option(
        "--seq-len",
        type=int,
        default=default_seq_len,
        show_default=True,
        help="NaFlex towers only: patch budget per image (16x16 patches, aspect ratio kept). 1024 matches a 512px fixed model in speed and VRAM; part of the model identity a DB is bound to.",
    ),
)

update_options = _options(
    click.option(
        "--batch-size", "-b", type=int, default=4, show_default=True, help="Images per model forward pass; raise if VRAM allows."
    ),
    click.option(
        "--workers",
        "-j",
        type=int,
        default=None,
        help="Image decoder processes, 2x this many reads are kept in flight (default: number of CPUs; raise it on network/slow storage). "
        "Independent of --batch-size. Also the number of pixel-check processes.",
    ),
    click.option(
        "--force-update",
        "-f",
        is_flag=True,
        help="Re-encode every image, ignoring modification times. Also rebinds the DB to --model-id.",
    ),
    click.option(
        "--clean-orphans/--no-clean-orphans",
        default=True,
        show_default=True,
        help="Remove DB entries whose image no longer exists.",
    ),
    click.option(
        "--include-hidden", is_flag=True, help="Also include hidden (dot) files and directories, which are skipped by default."
    ),
    click.option("--skip-update", is_flag=True, help="Do not touch the database, only match what is already in it."),
)

match_options = _options(
    click.option(
        "--threshold",
        "-t",
        type=float,
        default=default_euclidean_distance_threshold,
        show_default=True,
        help="Euclidean distance at or below which two images count as duplicates (lower = stricter).",
    ),
    click.option(
        "--trash-dir",
        type=click.Path(file_okay=False),
        default=None,
        help="Move duplicates here (mirroring their relative paths). Omit to only report.",
    ),
    click.option(
        "--dry-run",
        "-n",
        is_flag=True,
        help="Report what would be moved without moving. The DB is still refreshed unless --skip-update.",
    ),
)

pixel_options = _options(
    click.option(
        "--pixel-check/--no-pixel-check",
        default=True,
        show_default=True,
        help="Compare the pixels of every matched pair and only treat it as a duplicate when the two differ by lossy compression alone "
        "(not by a signature, an edit, a revision). Only ever blocks matches; in review mode it also decides which groups start decided.",
    ),
    click.option(
        "--pixel-buffer",
        type=click.IntRange(64),
        default=default_buffer_mib,
        show_default=True,
        help="MiB of decoded images the pixel check keeps across all its processes, so an image shared by several pairs is decoded once.",
    ),
)

image_dir_arg = click.option(
    "--image-dir", "-i", type=click.Path(exists=True, file_okay=False), required=True, help="Directory of images."
)
db_arg = click.option(
    "--db",
    "-d",
    type=click.Path(dir_okay=False),
    required=True,
    help="SQLite embedding database for --image-dir (created if missing).",
)


# -- helpers ---------------------------------------------------------------------------------------------------------


def _progress_factory(desc: str, unit: str) -> Callable[[int], Callable[[int], object]]:
    def factory(total: int) -> Callable[[int], object]:
        bar = tqdm.tqdm(total=total, desc=desc, unit=unit)
        return bar.update

    return factory


@contextlib.contextmanager
def _progress(total: int, desc: str, unit: str = "image") -> Generator[Callable[[int], object], None, None]:
    bar = tqdm.tqdm(total=total, desc=desc, unit=unit)
    try:
        yield bar.update
    finally:
        bar.close()


@contextlib.contextmanager
def _encoder(
    model_id: str, device: str, dtype: str | None = None, *, compile_: bool = False, seq_len: int = default_seq_len
) -> Generator[CLIPImageEncoder, None, None]:
    encoder = CLIPImageEncoder(model_id=model_id, device=device, dtype=dtype, compile=compile_, seq_len=seq_len)
    try:
        yield encoder
    finally:
        encoder.close()


def _update(
    encoder: CLIPImageEncoder,
    image_dir: str,
    db: str,
    *,
    force_update: bool,
    clean_orphans: bool,
    include_hidden: bool,
    batch_size: int,
    workers: int | None,
) -> None:
    logger.info("Updating database %s from %s", db, image_dir)
    try:
        update_database(
            encoder,
            image_dir,
            db,
            force_update=force_update,
            clean_orphans=clean_orphans,
            include_hidden=include_hidden,
            batch_size=batch_size,
            workers=workers,
            progress_factory=_progress_factory("Encoding", "image"),
        )
    except RuntimeError as e:
        raise click.ClickException(str(e)) from e


def _load_index(db: str, device: str, what: str = "database") -> tuple[list[ImageRecord], DistanceIndex]:
    records, embeddings = load_database(db)
    if not records:
        raise click.ClickException(f"No entries in the {what} {db}. Run an update first.")
    index = DistanceIndex(embeddings, device)
    logger.info(
        "Loaded %s: %d images, dim %d, %s on %s (%s)",
        what,
        index.n,
        index.dim,
        humanize.naturalsize(index.nbytes, binary=True),
        device,
        index.backend_name(),
    )
    return records, index


def _pixel_check(pairs: list[tuple[str, str]], workers: int | None, buffer_mib: int) -> list[PixelVerdict]:
    with _progress(len(pairs), "Pixel check", "pair") as progress:
        return check_pairs(pairs, workers=workers, buffer_bytes=buffer_mib << 20, progress=progress)


def _log_blocked(label_a: str, label_b: str, verdict: PixelVerdict) -> None:
    logger.info('Not a duplicate by pixels: "%s" and "%s": %s', label_a, label_b, verdict.describe())


def _policy(name: str, config: str | None):
    try:
        policies = load_policies(config)
    except (PolicyError, OSError) as e:
        raise click.ClickException(f"Cannot load keeping policies: {e}") from e
    if name not in policies:
        raise click.ClickException(f"Unknown keeping policy {name!r}. Available: {', '.join(sorted(policies))}")
    return policies[name]


# -- commands --------------------------------------------------------------------------------------------------------


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--verbose", "-v", is_flag=True, help="Debug logging.")
def cli(verbose: bool) -> None:
    """Image deduplication by CLIP embedding similarity."""
    setup_logging(verbose)
    torch.set_float32_matmul_precision("highest")  # distance math must not silently use TF32


@cli.command()
@image_dir_arg
@db_arg
@model_options
@update_options
@match_options
@click.option(
    "--keeping-logic", "-k", default="largest", show_default=True, help="Which copy of a duplicate group to keep (a policy name)."
)
@click.option(
    "--keeping-config",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="TOML file adding/overriding keeping policies.",
)
@click.option(
    "--review-threshold",
    type=float,
    default=None,
    help="Review mode: match at this looser distance and store every group in the review database for the GUI / review-apply "
    "instead of moving anything. Groups whose every pair passes the pixel check start decided (keeping-policy winner kept); "
    "with --no-pixel-check, groups entirely within --threshold do.",
)
@click.option(
    "--review-db", type=click.Path(dir_okay=False), default=None, help="Review database path (default: <db>.review.sqlite)."
)
@pixel_options
def dedupe(
    image_dir: str,
    db: str,
    model_id: str,
    device: str,
    compile_: bool,
    seq_len: int,
    batch_size: int,
    workers: int | None,
    force_update: bool,
    clean_orphans: bool,
    include_hidden: bool,
    skip_update: bool,
    threshold: float,
    trash_dir: str | None,
    dry_run: bool,
    keeping_logic: str,
    keeping_config: str | None,
    review_threshold: float | None,
    review_db: str | None,
    pixel_check: bool,
    pixel_buffer: int,
) -> None:
    """Find duplicates within one image directory."""
    policy = _policy(keeping_logic, keeping_config)  # validate before spending time on encoding
    if review_threshold is not None:
        if trash_dir is not None:
            raise click.ClickException("--review-threshold stores groups for review and moves nothing; drop --trash-dir.")
        if review_threshold < threshold:
            raise click.ClickException(f"--review-threshold ({review_threshold}) must be at least --threshold ({threshold}).")
    if not skip_update:
        with _encoder(model_id, device, compile_=compile_, seq_len=seq_len) as encoder:
            _update(
                encoder,
                image_dir,
                db,
                force_update=force_update,
                clean_orphans=clean_orphans,
                include_hidden=include_hidden,
                batch_size=batch_size,
                workers=workers,
            )

    records, index = _load_index(db, device)
    paths = [r.path for r in records]
    if review_threshold is not None:
        with _progress(index.n, "Matching") as progress:
            edges = find_duplicate_edges(paths, index, review_threshold, progress)
        index.release()
        review_path = review_db or default_review_path(db)
        with EmbeddingDB(db) as store, ReviewDB(review_path) as review:
            try:
                review.begin_session(
                    image_dir=image_dir,
                    model_id=store.model_id or "",
                    review_threshold=review_threshold,
                    auto_threshold=threshold,
                    policy=policy.name,
                )
            except ValueError as e:
                raise click.ClickException(str(e)) from e
            review.set_meta("pixel_check", "on" if pixel_check else "off")
            pixel = _review_pixel_check(review, edges, records, image_dir, workers, pixel_buffer) if pixel_check else None
            stats = store_matches(review, group_edges(index.n, edges), records, policy, threshold, pixel)
            counts = review.counts()
        rule = "every pair differs by compression only" if pixel_check else f"within {threshold:g}"
        logger.info(
            "Review database %s: %d groups this run (%d new, %d of them start decided because %s; %d seen before, of which %d untouched ones "
            "were re-decided and %d reopened), %d went stale; now %d pending, %d decided, %d skipped, %d applied",
            review_path,
            stats.new + stats.kept,
            stats.new,
            stats.predecided,
            rule,
            stats.kept,
            stats.redecided,
            stats.reopened,
            stats.stale,
            counts["pending"],
            counts["decided"],
            counts["skipped"],
            counts["applied"],
        )
        return
    with _progress(index.n, "Matching") as progress:
        edges = find_duplicate_edges(paths, index, threshold, progress)
    index.release()
    if pixel_check and edges:
        verdicts = _pixel_check(
            [(os.path.join(image_dir, paths[i]), os.path.join(image_dir, paths[j])) for i, j, _ in edges], workers, pixel_buffer
        )
        for (i, j, _), v in zip(edges, verdicts):
            if not v.same:
                _log_blocked(paths[i], paths[j], v)
        blocked = sum(1 for v in verdicts if not v.same)
        edges = [e for e, v in zip(edges, verdicts) if v.same]
        logger.info(
            "Pixel check: %d of %d matched pairs differ by more than compression and are not merged", blocked, len(verdicts)
        )
    groups = [g.members for g in group_edges(index.n, edges)]

    duplicates = sum(len(g) - 1 for g in groups)
    moved = 0
    if trash_dir is not None and groups:
        moved = trash_duplicate_groups(groups, records, image_dir, trash_dir, policy, dry_run=dry_run)
    logger.info(
        "Done: %d images, %d duplicates in %d groups%s%s",
        index.n,
        duplicates,
        len(groups),
        f", {moved} files moved" if trash_dir is not None else "",
        " (dry run, nothing moved)" if dry_run else "",
    )


def _review_pixel_check(
    review: ReviewDB,
    edges: list[tuple[int, int, float]],
    records: list[ImageRecord],
    image_dir: str,
    workers: int | None,
    buffer_mib: int,
) -> dict[tuple[int, int], PixelVerdict]:
    """Pixel verdicts for every edge, reusing the review database's results for pairs whose files are unchanged."""
    cached = review.pixel_cache()
    out: dict[tuple[int, int], PixelVerdict] = {}
    todo: list[tuple[int, int]] = []
    for i, j, _ in edges:
        a, b = records[i], records[j]
        hit = cached.get((a.path, b.path))
        if hit is not None and hit[1] == (a.mtime, a.size, b.mtime, b.size):
            out[(i, j)] = hit[0]
        else:
            todo.append((i, j))
    logger.info("Pixel check: %d pairs to compare, %d unchanged since the last run", len(todo), len(edges) - len(todo))
    if todo:
        verdicts = _pixel_check(
            [(os.path.join(image_dir, records[i].path), os.path.join(image_dir, records[j].path)) for i, j in todo],
            workers,
            buffer_mib,
        )
        out.update(zip(todo, verdicts))
    same = sum(1 for v in out.values() if v.same)
    logger.info("Pixel check: %d of %d pairs differ by compression only", same, len(out))
    return out


@cli.command("dedupe-import")
@click.option(
    "--base-image-dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Directory of images already in the collection.",
)
@click.option("--base-db", type=click.Path(dir_okay=False), required=True, help="Embedding database for --base-image-dir.")
@click.option(
    "--import-image-dir", type=click.Path(exists=True, file_okay=False), required=True, help="Directory of images to be imported."
)
@click.option("--import-db", type=click.Path(dir_okay=False), required=True, help="Embedding database for --import-image-dir.")
@model_options
@update_options
@match_options
@pixel_options
def dedupe_import(
    base_image_dir: str,
    base_db: str,
    import_image_dir: str,
    import_db: str,
    model_id: str,
    device: str,
    compile_: bool,
    seq_len: int,
    batch_size: int,
    workers: int | None,
    force_update: bool,
    clean_orphans: bool,
    include_hidden: bool,
    skip_update: bool,
    threshold: float,
    trash_dir: str | None,
    dry_run: bool,
    pixel_check: bool,
    pixel_buffer: int,
) -> None:
    """Remove images from an import directory that already exist in a base directory."""
    if not skip_update:
        with _encoder(model_id, device, compile_=compile_, seq_len=seq_len) as encoder:
            _update(
                encoder,
                base_image_dir,
                base_db,
                force_update=force_update,
                clean_orphans=clean_orphans,
                include_hidden=include_hidden,
                batch_size=batch_size,
                workers=workers,
            )
            _update(
                encoder,
                import_image_dir,
                import_db,
                force_update=force_update,
                clean_orphans=clean_orphans,
                include_hidden=include_hidden,
                batch_size=batch_size,
                workers=workers,
            )

    base_records, base_index = _load_index(base_db, device, "base database")
    import_records, import_index = _load_index(import_db, device, "import database")
    base_paths = [r.path for r in base_records]
    import_paths = [r.path for r in import_records]
    with _progress(import_index.n, "Matching") as progress:
        pairs = find_cross_pairs(import_paths, import_index, base_paths, base_index, threshold, progress)
    base_index.release()
    import_index.release()
    if pixel_check and pairs:
        verdicts = _pixel_check(
            [(os.path.join(import_image_dir, import_paths[i]), os.path.join(base_image_dir, base_paths[j])) for i, j, _ in pairs],
            workers,
            pixel_buffer,
        )
        for (i, j, _), v in zip(pairs, verdicts):
            if not v.same:
                _log_blocked(import_paths[i], f"base: {base_paths[j]}", v)
        logger.info(
            "Pixel check: %d of %d matched pairs differ by more than compression and do not count",
            sum(1 for v in verdicts if not v.same),
            len(verdicts),
        )
        pairs = [p for p, v in zip(pairs, verdicts) if v.same]
    hits: dict[int, int] = {}
    for i, _, _ in pairs:
        hits[i] = hits.get(i, 0) + 1

    moved = 0
    if trash_dir is not None:
        for i in sorted(hits):
            if move_to_trash(import_image_dir, import_paths[i], trash_dir, dry_run=dry_run):
                moved += 1
    logger.info(
        "Done: %d import images, %d already present in base (%d matches)%s%s",
        import_index.n,
        len(hits),
        sum(hits.values()),
        f", {moved} files moved" if trash_dir is not None else "",
        " (dry run, nothing moved)" if dry_run else "",
    )


@cli.command("update-db")
@image_dir_arg
@db_arg
@model_options
@update_options
def update_db(
    image_dir: str,
    db: str,
    model_id: str,
    device: str,
    compile_: bool,
    seq_len: int,
    batch_size: int,
    workers: int | None,
    force_update: bool,
    clean_orphans: bool,
    include_hidden: bool,
    skip_update: bool,
) -> None:
    """Only (re)encode images into the database, without matching."""
    if not skip_update:
        with _encoder(model_id, device, compile_=compile_, seq_len=seq_len) as encoder:
            _update(
                encoder,
                image_dir,
                db,
                force_update=force_update,
                clean_orphans=clean_orphans,
                include_hidden=include_hidden,
                batch_size=batch_size,
                workers=workers,
            )
    records, embeddings = load_database(db)
    logger.info("%s holds %d embeddings of shape %s", db, len(records), embeddings.shape[1:])


@cli.command("encode-test")
@model_options
@click.option(
    "--dtype",
    type=click.Choice(precision_choices, case_sensitive=False),
    default=None,
    help="Model precision (default: fp16 on CUDA, fp32 otherwise).",
)
@click.argument("image_paths", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False))
def encode_test(model_id: str, device: str, compile_: bool, seq_len: int, dtype: str | None, image_paths: tuple[str, ...]) -> None:
    """Encode a few images and print their pairwise distance matrix (for picking a threshold)."""
    import PIL.Image

    with _encoder(model_id, device, dtype, compile_=compile_, seq_len=seq_len) as encoder:
        images = []
        for p in image_paths:
            try:
                with PIL.Image.open(p) as img:
                    images.append(img.convert("RGB"))
            except (OSError, PIL.UnidentifiedImageError) as e:
                logger.warning("Skipping %s: %s", p, e)
        if not images:
            raise click.ClickException("No readable images given.")
        features = encoder.encode_pil_images(images)
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    click.echo(f"Feature matrix shape: {features.shape}")
    click.echo("Euclidean distance matrix:")
    click.echo(euclidean_distance(features, features))


review_db_option = click.option(
    "--review-db", type=click.Path(dir_okay=False), default=None, help="Review database path (default: <db>.review.sqlite)."
)


@cli.command("review-status")
@db_arg
@review_db_option
def review_status(db: str, review_db: str | None) -> None:
    """Print what the review database holds."""
    path = review_db or default_review_path(db)
    if not os.path.exists(path):
        raise click.ClickException(f"No review database at {path}. Run dedupe --review-threshold first.")
    with ReviewDB(path) as review:
        counts = review.counts()
        click.echo(
            f"{path}: model {review.get_meta('model_id')}, review threshold {review.review_threshold}, auto threshold {review.auto_threshold}, policy {review.get_meta('policy')}"
        )
        pixel = review.pixel_counts()
        click.echo(f"  pixel check: {review.get_meta('pixel_check') or 'off'}")
        for status, n in counts.items():
            same = pixel.get((status, "same"), 0)
            different = pixel.get((status, "different"), 0)
            extra = f"  ({same} compression only, {different} with pixel differences)" if same or different else ""
            click.echo(f"  {status:8s} {n}{extra}")
        moves = review.plan_apply()
        click.echo(f"  {len(moves)} files would be moved by review-apply")
        discarded = review.groups("decided", discarded=True)
        if discarded:
            click.echo(f"  {len(discarded)} decided groups are discarded whole (every member is moved)")
        keepless = review.groups("decided", keeps_nothing=True)
        if keepless:
            click.echo(
                f"  {len(keepless)} decided groups keep nothing and are skipped: {', '.join(str(g.id) for g in keepless[:20])}{' …' if len(keepless) > 20 else ''}"
            )


@cli.command("review-apply")
@image_dir_arg
@db_arg
@review_db_option
@click.option(
    "--trash-dir",
    type=click.Path(file_okay=False),
    required=True,
    help="Move the losers of decided groups here (mirroring their relative paths).",
)
@click.option("--dry-run", "-n", is_flag=True, help="Report what would be moved without moving.")
def review_apply(image_dir: str, db: str, review_db: str | None, trash_dir: str, dry_run: bool) -> None:
    """Move the losers of every decided group to the trash directory."""
    path = review_db or default_review_path(db)
    if not os.path.exists(path):
        raise click.ClickException(f"No review database at {path}. Run dedupe --review-threshold first.")
    with ReviewDB(path) as review:
        decided = review.groups("decided")
        keepless = review.groups("decided", keeps_nothing=True)
        if keepless:
            logger.warning(
                "%d decided groups keep nothing and are skipped (keep a member, or discard the whole group with R in the GUI): %s",
                len(keepless),
                ", ".join(str(g.id) for g in keepless),
            )
        with _progress(len(decided), "Applying", "group") as progress:
            moved, refused = apply_decisions(review, image_dir, trash_dir, dry_run=dry_run, progress=progress)
    logger.info(
        "Done: %d files %s, %d groups refused because files changed on disk%s",
        moved,
        "would be moved" if dry_run else "moved",
        len(refused),
        " (dry run)" if dry_run else "",
    )


@cli.command("calibrate")
@image_dir_arg
@db_arg
@model_options
@update_options
@click.option("--samples", "-s", type=int, default=200, show_default=True, help="Number of images to sample from the database.")
@click.option(
    "--seed",
    type=int,
    default=0,
    show_default=True,
    help="Sampling seed; the same seed on an unchanged directory picks the same images.",
)
@click.option(
    "--variants",
    default=",".join(default_variants),
    show_default=True,
    help=f"Comma-separated lossy copies to synthesize per sample. Available: {', '.join(VARIANTS)}.",
)
@click.option(
    "--threshold",
    "-t",
    "thresholds",
    type=float,
    multiple=True,
    help="Threshold(s) to evaluate against the distributions (default: the dedupe default).",
)
def calibrate_cmd(
    image_dir: str,
    db: str,
    model_id: str,
    device: str,
    compile_: bool,
    seq_len: int,
    batch_size: int,
    workers: int | None,
    force_update: bool,
    clean_orphans: bool,
    include_hidden: bool,
    skip_update: bool,
    samples: int,
    seed: int,
    variants: str,
    thresholds: tuple[float, ...],
) -> None:
    """Measure how far lossy copies and different images sit, and suggest a threshold.

    Samples images from the database, synthesizes variants (JPEG re-save, downscale, ...) and prints histograms of
    their distance to the stored embedding next to the distance to the nearest different image.
    """
    variant_names = [v.strip() for v in variants.split(",") if v.strip()]
    unknown = [v for v in variant_names if v not in VARIANTS]
    if unknown:
        raise click.ClickException(f"Unknown variants {', '.join(unknown)}; available: {', '.join(VARIANTS)}")
    with _encoder(model_id, device, compile_=compile_, seq_len=seq_len) as encoder:
        if not skip_update:
            _update(
                encoder,
                image_dir,
                db,
                force_update=force_update,
                clean_orphans=clean_orphans,
                include_hidden=include_hidden,
                batch_size=batch_size,
                workers=workers,
            )
        with EmbeddingDB(db) as store:
            if store.model_id != encoder.model_id:
                raise click.ClickException(
                    f"Database {db} was encoded with model '{store.model_id}', but '{encoder.model_id}' is loaded."
                )
        records, embeddings = load_database(db)
        if not records:
            raise click.ClickException(f"No entries in the database {db}. Run an update first.")
        index = DistanceIndex(embeddings, device)
        n = min(samples, len(records))
        with _progress(n, "Calibrating") as progress:
            result = calibrate(
                encoder,
                image_dir,
                records,
                embeddings,
                index,
                samples=samples,
                seed=seed,
                variants=variant_names,
                batch_size=batch_size,
                progress=progress,
            )
        index.release()
    for line in report(result, thresholds or (default_euclidean_distance_threshold,)):
        click.echo(line)


if __name__ == "__main__":
    cli()
