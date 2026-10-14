"""Command line interface: ``picurate-dedupe <subcommand>``."""

from __future__ import annotations

import contextlib
import logging
import tempfile
from collections.abc import Callable, Generator
from pathlib import Path

import click
import humanize
import numpy as np
import torch
import tqdm

from ..common.images import ImageRecord, Item, Roots, standalone_items, standalone_roots
from ..common.log import setup_logging
from ..project.project import EMBEDDINGS, REVIEW, Project
from .calibrate import VARIANTS, calibrate, default_variants, report
from .db_store import EmbeddingDB, ModelMismatchError, load_database
from .dedupe import find_cross_pairs, find_duplicate_edges, group_edges, move_to_trash
from .encoder import CLIPImageEncoder, default_model_id, default_seq_len, precision_choices
from .encoding_pipeline import update_database
from .keeping import PolicyError, load_policies
from .pixel_check import PixelVerdict, check_pairs, default_buffer_mib
from .review import ReviewDB, store_matches
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
        help="The image tower: the SigLIP2 so400m NaFlex checkpoint by name, or the path of a clip-train export file.",
    ),
    click.option("--device", "-c", default=default_device, show_default=True, help="Device to run the model and the matching on."),
    click.option(
        "--compile",
        "compile_",
        is_flag=True,
        help="torch.compile the model (dynamic shapes, 20-30 s warm-up).",
    ),
    click.option(
        "--seq-len",
        type=int,
        default=default_seq_len,
        show_default=True,
        help="Patch budget per image (16x16 patches, aspect ratio kept). 1024 matches a 512px fixed model in speed and VRAM; part of the model identity a DB is bound to.",
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
    click.option("--skip-update", is_flag=True, help="Do not touch the embedding database, only match what is already in it."),
)

threshold_option = click.option(
    "--threshold",
    "-t",
    type=float,
    default=default_euclidean_distance_threshold,
    show_default=True,
    help="Euclidean distance at or below which two images count as duplicates (lower = stricter).",
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

project_arg = click.argument("project_dir", type=click.Path(exists=True, file_okay=False))


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


def _project(project_dir: str) -> Project:
    try:
        return Project(project_dir)
    except ValueError as e:
        raise click.ClickException(str(e)) from e


def _lock_review(project: Project, *, check_only: bool = False) -> None:
    """Take the review database for this run; ``check_only`` fails early without holding it (through encoding)."""
    try:
        project.lock("review")
    except ValueError as e:
        raise click.ClickException(f"{e}; close it and run again") from e
    if check_only:
        project.unlock("review")


def _scope(project: Project) -> tuple[Roots, list[Item]]:
    """The project's roots, and the items taking part: present (not marked missing by the last sync, not in the
    trash) and not excluded by a person. Losers of duplicate groups take part, so their groups stay stable."""
    roots = {source: str(path) for source, path in project.sources.items()}
    out = project.human_excluded() | project.trashed
    items = [
        Item(item, row["source_id"], row["path"]) for item, row in project.items.items() if not row["missing"] and item not in out
    ]
    return roots, items


def _embeddings(project: Project) -> str:
    return str(project.directory / EMBEDDINGS)


def _update_project(
    encoder: CLIPImageEncoder,
    project: Project,
    *,
    force_update: bool,
    batch_size: int,
    workers: int | None,
    reuse_from: str | None = None,
) -> None:
    """Encode the project's new and changed images into its embedding database. Rows of items still in the project
    are kept even when they take no part now (missing, excluded by a person); rows of removed items are dropped."""
    roots, items = _scope(project)
    logger.info("Updating %s", _embeddings(project))
    try:
        update_database(
            encoder,
            roots,
            items,
            _embeddings(project),
            force_update=force_update,
            batch_size=batch_size,
            workers=workers,
            progress_factory=_progress_factory("Encoding", "image"),
            reuse_from=reuse_from,
            keep_rows=project.items.keys(),
        )
    except RuntimeError as e:
        raise click.ClickException(str(e)) from e


def _update_dir(
    encoder: CLIPImageEncoder, image_dir: str, db: str, *, include_hidden: bool, batch_size: int, workers: int | None
) -> None:
    logger.info("Updating %s from %s", db, image_dir)
    try:
        update_database(
            encoder,
            standalone_roots(image_dir),
            standalone_items(image_dir, include_hidden=include_hidden),
            db,
            batch_size=batch_size,
            workers=workers,
            progress_factory=_progress_factory("Encoding", "image"),
        )
    except RuntimeError as e:
        raise click.ClickException(str(e)) from e


def _load(db: str) -> tuple[list[ImageRecord], np.ndarray]:
    try:
        return load_database(db)
    except ModelMismatchError as e:
        raise click.ClickException(str(e)) from e


def _load_index(
    db: str, device: str, what: str = "database", only: set[bytes] | None = None
) -> tuple[list[ImageRecord], DistanceIndex]:
    """Load ``db`` onto ``device``; ``only`` keeps the rows of these item IDs."""
    records, embeddings = _load(db)
    if only is not None:
        rows = [i for i, r in enumerate(records) if r.id in only]
        records, embeddings = [records[i] for i in rows], embeddings[rows]
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


def _review(path) -> ReviewDB:
    try:
        return ReviewDB(str(path))
    except ValueError as e:
        raise click.ClickException(str(e)) from e


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
    """Image deduplication by CLIP embedding similarity, on a project (see picurate-project)."""
    setup_logging(verbose)
    torch.set_float32_matmul_precision("highest")  # distance math must not silently use TF32


@cli.command()
@project_arg
@model_options
@update_options
@threshold_option
@click.option(
    "--review-threshold",
    type=float,
    default=None,
    help="Match at this looser distance; groups beyond what starts decided wait for review in picurate-curate "
    "(default: --threshold, so only the groups decided automatically are stored).",
)
@click.option(
    "--keeping-logic", "-k", default="largest", show_default=True, help="Which copy of a duplicate group to keep (a policy name)."
)
@click.option(
    "--keeping-config",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="TOML file adding/overriding keeping policies.",
)
@pixel_options
def dedupe(
    project_dir: str,
    model_id: str,
    device: str,
    compile_: bool,
    seq_len: int,
    batch_size: int,
    workers: int | None,
    force_update: bool,
    skip_update: bool,
    threshold: float,
    review_threshold: float | None,
    keeping_logic: str,
    keeping_config: str | None,
    pixel_check: bool,
    pixel_buffer: int,
) -> None:
    """Find duplicate groups among the images of the project at PROJECT_DIR.

    New and changed images are encoded into the project's embeddings.sqlite, and the groups go to its review.sqlite.
    Groups whose every pair differs by compression alone (with --no-pixel-check: lie within --threshold) start decided,
    keeping the policy's winner; the others, and groups holding several pages of one work (a policy's variants
    pattern), wait for review in picurate-curate. The images a decided group does not
    keep count as excluded: the editor does not export them. Nothing is moved here. Images a person excluded take no part; run 'picurate-project sync' first to
    pick up new files. Needs picurate-curate to be closed.
    """
    policy = _policy(keeping_logic, keeping_config)  # validate before spending time on encoding
    if review_threshold is None:
        review_threshold = threshold
    elif review_threshold < threshold:
        raise click.ClickException(f"--review-threshold ({review_threshold}) must be at least --threshold ({threshold}).")
    project = _project(project_dir)
    _lock_review(project, check_only=True)
    review_path = project.directory / REVIEW
    if review_path.exists():
        _review(review_path).close()  # refuse another schema version before encoding
    if not skip_update:
        with _encoder(model_id, device, compile_=compile_, seq_len=seq_len) as encoder:
            _update_project(encoder, project, force_update=force_update, batch_size=batch_size, workers=workers)
    _lock_review(project)
    roots, items = _scope(project)
    records, index = _load_index(_embeddings(project), device, only={item.id for item in items})
    with _progress(index.n, "Matching") as progress:
        edges = find_duplicate_edges([r.path for r in records], index, review_threshold, progress)
    index.release()
    with EmbeddingDB(_embeddings(project)) as store, _review(review_path) as review:
        try:
            review.begin_session(
                project_id=project.settings["project_id"],
                model_id=store.model_id or "",
                review_threshold=review_threshold,
                auto_threshold=threshold,
                policy=policy.name,
            )
        except ValueError as e:
            raise click.ClickException(str(e)) from e
        review.set_meta("pixel_check", "on" if pixel_check else "off")
        pixel = _review_pixel_check(review, edges, records, roots, workers, pixel_buffer) if pixel_check else None
        stats = store_matches(review, group_edges(index.n, edges), records, policy, threshold, pixel)
        counts = review.counts()
        excluded = review.excluded_count()
    rule = "every pair differs by compression only" if pixel_check else f"within {threshold:g}"
    logger.info(
        "%s: %d groups this run (%d new, %d of them start decided because %s, %d took over the keeps of a changed group; "
        "%d seen before, of which %d untouched ones were re-decided and %d reopened), %d went stale; now %d pending, "
        "%d decided, %d skipped; %d images excluded as duplicates",
        review_path,
        stats.new + stats.kept,
        stats.new,
        stats.predecided,
        rule,
        stats.carried,
        stats.kept,
        stats.redecided,
        stats.reopened,
        stats.stale,
        counts["pending"],
        counts["decided"],
        counts["skipped"],
        excluded,
    )


def _review_pixel_check(
    review: ReviewDB,
    edges: list[tuple[int, int, float]],
    records: list[ImageRecord],
    roots: Roots,
    workers: int | None,
    buffer_mib: int,
) -> dict[tuple[int, int], PixelVerdict]:
    """Pixel verdicts for every edge, reusing the review database's results for pairs whose files are unchanged."""
    cached = review.pixel_cache()
    out: dict[tuple[int, int], PixelVerdict] = {}
    todo: list[tuple[int, int]] = []
    for i, j, _ in edges:
        a, b = records[i], records[j]
        hit = cached.get((a.id, b.id))
        if hit is not None and hit[1] == (a.mtime, a.size, b.mtime, b.size):
            out[(i, j)] = hit[0]
        else:
            todo.append((i, j))
    logger.info("Pixel check: %d pairs to compare, %d unchanged since the last run", len(todo), len(edges) - len(todo))
    if todo:
        verdicts = _pixel_check([(records[i].file(roots), records[j].file(roots)) for i, j in todo], workers, buffer_mib)
        out.update(zip(todo, verdicts))
    same = sum(1 for v in out.values() if v.same)
    logger.info("Pixel check: %d of %d pairs differ by compression only", same, len(out))
    return out


@cli.command("dedupe-import")
@click.option(
    "--import-image-dir",
    "-i",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Directory of new images, before they are added to the library.",
)
@project_arg
@click.option(
    "--import-db",
    type=click.Path(dir_okay=False),
    default=None,
    help="Embedding database for --import-image-dir, kept for the next run or for update-db --reuse-from "
    "(default: a temporary one).",
)
@click.option(
    "--trash-dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Move new images already in the project here (mirroring their relative paths). Omit to only report.",
)
@click.option("--dry-run", "-n", is_flag=True, help="Report what would be moved without moving.")
@click.option("--include-hidden", is_flag=True, help="Also include hidden (dot) files and directories of the import directory.")
@model_options
@update_options
@threshold_option
@pixel_options
def dedupe_import(
    import_image_dir: str,
    project_dir: str,
    import_db: str | None,
    trash_dir: str | None,
    dry_run: bool,
    include_hidden: bool,
    model_id: str,
    device: str,
    compile_: bool,
    seq_len: int,
    batch_size: int,
    workers: int | None,
    force_update: bool,
    skip_update: bool,
    threshold: float,
    pixel_check: bool,
    pixel_buffer: int,
) -> None:
    """Find the images of a new directory that the project at PROJECT_DIR already has, and move them to --trash-dir.

    The project's images that are excluded (by a person or as duplicates) do not count. Duplicates within the import
    directory are not looked for. The project's embeddings are brought up to date first unless --skip-update.
    """
    project = _project(project_dir)
    with contextlib.ExitStack() as stack:
        if import_db is None:
            import_db = str(Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="picurate-import-"))) / "import.sqlite")
        with _encoder(model_id, device, compile_=compile_, seq_len=seq_len) as encoder:
            if not skip_update:
                _update_project(encoder, project, force_update=force_update, batch_size=batch_size, workers=workers)
            _update_dir(encoder, import_image_dir, import_db, include_hidden=include_hidden, batch_size=batch_size, workers=workers)
        roots, items = _scope(project)
        excluded = {state["item_id"] for state in project.states(captions=False) if state["excluded"]}
        base_records, base_index = _load_index(
            _embeddings(project), device, "project database", only={item.id for item in items} - excluded
        )
        import_records, import_index = _load_index(import_db, device, "import database")
    base_paths = [r.path for r in base_records]
    import_paths = [r.path for r in import_records]
    with _progress(import_index.n, "Matching") as progress:
        pairs = find_cross_pairs(import_paths, import_index, base_paths, base_index, threshold, progress)
    base_index.release()
    import_index.release()
    import_roots = standalone_roots(import_image_dir)
    if pixel_check and pairs:
        verdicts = _pixel_check(
            [(import_records[i].file(import_roots), base_records[j].file(roots)) for i, j, _ in pairs], workers, pixel_buffer
        )
        for (i, j, _), v in zip(pairs, verdicts):
            if not v.same:
                _log_blocked(import_paths[i], f"project: {base_records[j].file(roots)}", v)
        logger.info(
            "Pixel check: %d of %d matched pairs differ by more than compression and do not count",
            sum(1 for v in verdicts if not v.same),
            len(verdicts),
        )
        pairs = [p for p, v in zip(pairs, verdicts) if v.same]
    hits: dict[int, list[int]] = {}
    for i, j, _ in pairs:
        hits.setdefault(i, []).append(j)

    moved = 0
    for i in sorted(hits):
        reason = f"already in the project as {base_records[hits[i][0]].file(roots)}"
        if trash_dir is None:
            logger.info('"%s" is %s', import_records[i].file(import_roots), reason)
        elif move_to_trash(import_image_dir, import_paths[i], trash_dir, dry_run=dry_run, reason=reason):
            moved += 1
    logger.info(
        "Done: %d import images, %d already in the project (%d matches)%s%s",
        import_index.n,
        len(hits),
        sum(len(v) for v in hits.values()),
        f", {moved} files moved" if trash_dir is not None else "",
        " (dry run, nothing moved)" if dry_run else "",
    )


@cli.command("update-db")
@project_arg
@model_options
@update_options
@click.option(
    "--reuse-from",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Another database of the same model: images whose content (BLAKE3) it already holds are copied from it instead "
    "of encoded. Every new or changed image is read once more to hash it.",
)
def update_db(
    project_dir: str,
    model_id: str,
    device: str,
    compile_: bool,
    seq_len: int,
    batch_size: int,
    workers: int | None,
    force_update: bool,
    skip_update: bool,
    reuse_from: str | None,
) -> None:
    """Only (re)encode the project's images into its embedding database, without matching."""
    project = _project(project_dir)
    if not skip_update:
        with _encoder(model_id, device, compile_=compile_, seq_len=seq_len) as encoder:
            _update_project(
                encoder, project, force_update=force_update, batch_size=batch_size, workers=workers, reuse_from=reuse_from
            )
    records, embeddings = _load(_embeddings(project))
    logger.info("%s holds %d embeddings of shape %s", _embeddings(project), len(records), embeddings.shape[1:])


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


@cli.command("review-status")
@project_arg
def review_status(project_dir: str) -> None:
    """Print what the project's review database holds."""
    project = _project(project_dir)
    path = project.directory / REVIEW
    if not path.exists():
        raise click.ClickException(f"No review database at {path}. Run dedupe first.")
    with _review(path) as review:
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
        click.echo(f"  {review.excluded_count()} images excluded as duplicates")
        discarded = review.groups("decided", discarded=True)
        if discarded:
            click.echo(f"  {len(discarded)} decided groups are discarded whole (every member is excluded)")
        keepless = review.groups(("pending", "skipped"), keeps_nothing=True)
        if keepless:
            click.echo(
                f"  {len(keepless)} open groups keep nothing: {', '.join(str(g.id) for g in keepless[:20])}{' …' if len(keepless) > 20 else ''}"
            )


@cli.command("calibrate")
@project_arg
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
    project_dir: str,
    model_id: str,
    device: str,
    compile_: bool,
    seq_len: int,
    batch_size: int,
    workers: int | None,
    force_update: bool,
    skip_update: bool,
    samples: int,
    seed: int,
    variants: str,
    thresholds: tuple[float, ...],
) -> None:
    """Measure how far lossy copies and different images sit, and suggest a threshold.

    Samples images of the project at PROJECT_DIR from its embedding database, synthesizes variants (JPEG re-save, downscale, ...) and prints histograms of
    their distance to the stored embedding next to the distance to the nearest different image.
    """
    variant_names = [v.strip() for v in variants.split(",") if v.strip()]
    unknown = [v for v in variant_names if v not in VARIANTS]
    if unknown:
        raise click.ClickException(f"Unknown variants {', '.join(unknown)}; available: {', '.join(VARIANTS)}")
    project = _project(project_dir)
    db = _embeddings(project)
    with _encoder(model_id, device, compile_=compile_, seq_len=seq_len) as encoder:
        if not skip_update:
            _update_project(encoder, project, force_update=force_update, batch_size=batch_size, workers=workers)
        with EmbeddingDB(db) as store:
            if store.model_id != encoder.model_id:
                raise click.ClickException(
                    f"Database {db} was encoded with model '{store.model_id}', but '{encoder.model_id}' is loaded."
                )
        roots, items = _scope(project)
        records, embeddings = _load(db)
        present = {item.id for item in items}
        rows = [i for i, r in enumerate(records) if r.id in present]
        records, embeddings = [records[i] for i in rows], embeddings[rows]
        if not records:
            raise click.ClickException(f"No entries in the database {db}. Run an update first.")
        index = DistanceIndex(embeddings, device)
        n = min(samples, len(records))
        with _progress(n, "Calibrating") as progress:
            result = calibrate(
                encoder,
                roots,
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
