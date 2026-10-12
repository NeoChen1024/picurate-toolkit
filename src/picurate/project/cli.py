"""``picurate-project``: create a dataset project and import existing image/caption pairs into it."""

from __future__ import annotations

import datetime
import re

import click
import tqdm

from ..common.log import setup_logging
from .project import MOVE_TYPES, Project

AGE_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 7 * 86400}


def _open(project_dir: str, *, lock: bool = False) -> Project:
    try:
        project = Project(project_dir)
        if lock:
            project.lock()
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    return project


def _age(value: str | None) -> float | None:
    if value is None:
        return None
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([smhdw])", value.strip())
    if match is None:
        raise click.BadParameter("a number with a unit: s, m, h, d or w (e.g. 30d)", param_hint="--older-than")
    return float(match[1]) * AGE_UNITS[match[2]]


class _OrderedFilters(click.Command):
    """Keeps the command-line order of --include and --exclude, which decides between them (first match wins)."""

    def make_parser(self, ctx: click.Context):
        parser = super().make_parser(ctx)
        parse_args = parser.parse_args

        def parse(args):
            opts, largs, order = parse_args(args=args)
            ctx.meta["filter_order"] = [param.name for param in order if param.name in ("include", "exclude")]
            return opts, largs, order

        parser.parse_args = parse
        return parser


def _filters(ctx: click.Context, include: tuple[str, ...], exclude: tuple[str, ...]) -> list[tuple[str, str]]:
    values = {"include": iter(include), "exclude": iter(exclude)}
    return [("+" if name == "include" else "-", next(values[name])) for name in ctx.meta.get("filter_order", [])]


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """Dataset projects: the images stay where they are; captions and edits live in the project directory."""


@main.command()
@click.argument("project_dir", type=click.Path(file_okay=False))
def init(project_dir: str) -> None:
    """Create an empty project in PROJECT_DIR (new or empty)."""
    try:
        project = Project(project_dir, create=True)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"Created project {project.directory}", err=True)


@main.command("import", cls=_OrderedFilters)
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("sources", nargs=-1, required=True, type=click.Path(exists=True))
@click.option(
    "--include-hidden", is_flag=True, help="Also import hidden (dot) files and directories, which are skipped by default."
)
@click.option("--recursive", "-r", is_flag=True, help="Also import images in subdirectories.")
@click.option("--include", multiple=True, metavar="PATTERN", help="rsync-style include rule for paths below a directory source.")
@click.option("--exclude", multiple=True, metavar="PATTERN", help="rsync-style exclude rule; rules apply in command-line order.")
@click.option("--no-captions", is_flag=True, help="Import the images only, ignoring caption files next to them.")
@click.option("--verbose", "-v", is_flag=True, help="Debug logging.")
@click.pass_context
def import_(
    ctx: click.Context,
    project_dir: str,
    sources: tuple[str, ...],
    recursive: bool,
    include_hidden: bool,
    include: tuple[str, ...],
    exclude: tuple[str, ...],
    no_captions: bool,
    verbose: bool,
) -> None:
    """Import images from SOURCES (directories, with -r also their subdirectories, or single files) into the project at
    PROJECT_DIR.

    A same-stem UTF-8 .txt next to an image becomes its imported caption, unless --no-captions. Images already in the
    project are skipped with their edits kept, so an interrupted import can simply be run again.

    \b
    --include/--exclude work like rsync's, on paths relative to each directory given; the first rule that matches
    decides, and an excluded directory is not descended into. A pattern without '/' matches the file or directory
    name, one with '/' or '**' the path (a leading '/' anchors it to the directory); a trailing '/' matches
    directories only, '/***' a directory and everything in it. Only PNG files, in every subdirectory:
      picurate-project import P -r --include '*/' --include '*.png' --exclude '*' SOURCE
    The rules are recorded with the source folder for 'sync'.
    """
    setup_logging(verbose)
    try:
        project = Project(project_dir)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    with tqdm.tqdm(unit="img", desc="Importing", dynamic_ncols=True) as bar:

        def progress(done: int, total: int) -> None:
            bar.total = total
            bar.update(done - bar.n)

        try:
            imported, skipped = project.import_images(
                sources,
                captions=not no_captions,
                recursive=recursive,
                include_hidden=include_hidden,
                filters=_filters(ctx, include, exclude),
                progress=progress,
            )
        except ValueError as e:
            raise click.ClickException(str(e)) from e
    click.echo(f"Imported {imported} images, skipped {skipped} already in the project; {len(project.items)} in total.", err=True)


@main.command()
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--verbose", "-v", is_flag=True, help="Debug logging.")
def sync(project_dir: str, verbose: bool) -> None:
    """Bring the project at PROJECT_DIR up to date with its source folders: import new images of imported
    directories (recursive and hidden as they were imported), record changed images, mark vanished ones missing.
    Captions and decisions are kept. Needs the project to be closed in the editor."""
    setup_logging(verbose)
    project = _open(project_dir, lock=True)
    with tqdm.tqdm(unit="img", desc="Syncing", dynamic_ncols=True) as bar:

        def progress(done: int, total: int) -> None:
            bar.total = total
            bar.update(done - bar.n)

        result = project.sync(progress)
    for root in result["unavailable"]:
        click.echo(f"Source folder not found, skipped (relink it in the editor): {root}", err=True)
    for path in result["trash_gone"]:
        click.echo(f"Gone from the trash; gc removes it from the project: {path}", err=True)
    click.echo(
        f"Added {result['added']}, changed {result['changed']}, newly missing {result['missing']}, back {result['back']}; "
        f"{len(project.items)} images, {sum(row['missing'] for row in project.items.values())} missing.",
        err=True,
    )


@main.command()
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--limit", "-n", type=click.IntRange(1), default=30, show_default=True, help="How many operations to list.")
def history(project_dir: str, limit: int) -> None:
    """List the most recent operations of the project at PROJECT_DIR, newest first: ID, time, kind, number of field
    changes still in the history, and description (for an editor visit, the image)."""
    project = _open(project_dir)
    for op in project.ops(limit):
        at = datetime.datetime.fromtimestamp(op["at"]).strftime("%Y-%m-%d %H:%M:%S")
        click.echo(f"{op['id']:>6}  {at}  {op['kind']:<9} {op['changes']:>5}  {op['description'] or ''}")


@main.command()
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("op", type=int)
def rollback(project_dir: str, op: int) -> None:
    """Undo operation OP (see 'history') in the project at PROJECT_DIR: every field it changed gets its earlier value
    back. Fields changed again since are left alone and listed. The rollback is itself an operation, so it can be
    rolled back too. Needs the project to be closed in the editor."""
    project = _open(project_dir, lock=True)
    try:
        new, skipped, restored, moved = project.rollback(op)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    for item, field in skipped:
        if field == "trashed":
            click.echo(f"Cannot move back (gone from the trash, or its place is taken): {project.trash_path(item)}", err=True)
        else:
            click.echo(f"Changed since, left alone: {project.items[item]['path']} {field}", err=True)
    click.echo(
        f"Restored {restored} fields as operation {new}; {len(skipped)} left alone"
        + (f"; moved {moved} images out of or into the trash." if moved else "."),
        err=True,
    )


@main.command()
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--trash-excluded",
    is_flag=True,
    help="Move excluded images (with their imported caption files) into the project's trash/, as one operation that "
    "rollback moves back.",
)
@click.option(
    "--move-type",
    type=click.Choice(MOVE_TYPES),
    default="all",
    show_default=True,
    help="With --trash-excluded: images excluded as duplicates (dedupe), by a person (human), or both.",
)
@click.option("--keep-revisions", type=click.IntRange(0), default=None, help="Keep this many newest history entries per field.")
@click.option("--older-than", default=None, help="Only prune entries of operations older than this (e.g. 12h, 30d, 2w).")
@click.option("--dry-run", "-n", is_flag=True, help="Report what would be moved or pruned without changing anything.")
@click.option("--verbose", "-v", is_flag=True, help="Debug logging.")
def gc(
    project_dir: str,
    trash_excluded: bool,
    move_type: str,
    keep_revisions: int | None,
    older_than: str | None,
    dry_run: bool,
    verbose: bool,
) -> None:
    """Move excluded images into the trash, and prune old history in the project at PROJECT_DIR, with the LaMa masks
    and patches nothing refers to afterwards.

    \b
    --trash-excluded moves to PROJECT_DIR/trash/<source folder>-<id>/<path>; nothing is deleted. Images later
    deleted from the trash by hand no longer exist: every gc run removes them from the project, with their decisions
    and history.

    With both --keep-revisions and --older-than, an entry goes only when it is beyond the newest --keep-revisions of
    its field and older than --older-than. Pruned operations can no longer be rolled back; current decisions are
    never touched. Needs the project to be closed in the editor.
    """
    setup_logging(verbose)
    if not trash_excluded and keep_revisions is None and older_than is None:
        raise click.UsageError("Give --trash-excluded, --keep-revisions, --older-than or a combination.")
    project = _open(project_dir, lock=True)
    if trash_excluded:
        if dry_run:
            for state in project.trash_candidates(move_type):
                item = state["item_id"]
                click.echo(f"Would move {project.original_path(item)} -> {project.trash_path(item)}", err=True)
        else:
            with tqdm.tqdm(unit="img", desc="Moving to trash", dynamic_ncols=True) as bar:

                def progress(done: int, total: int) -> None:
                    bar.total = total
                    bar.update(done - bar.n)

                op = project.trash_excluded(move_type, progress)
            if op:
                moved = project.db.execute("SELECT COUNT(*) FROM history WHERE op = ?", (op,)).fetchone()[0]
                click.echo(f"Moved {moved} images to {project.directory / 'trash'} as operation {op}.", err=True)
            else:
                click.echo("No excluded images to move.", err=True)
    result = project.gc(keep_revisions=keep_revisions, older_than=_age(older_than), dry_run=dry_run)
    for path in result["removed"]:
        click.echo(
            f"{'Would remove' if dry_run else 'Removed'} from the project (gone from the trash): {path}",
            err=True,
        )
    for path in result["artifacts"]:
        click.echo(f"{'Would remove' if dry_run else 'Removed'} {path}", err=True)
    click.echo(
        f"{'Would prune' if dry_run else 'Pruned'} {result['entries']} history entries, {len(result['ops'])} operations and "
        f"{len(result['artifacts'])} artifacts; {len(result['removed'])} items gone from the trash.",
        err=True,
    )


if __name__ == "__main__":
    main()
