"""``picurate-project``: create a dataset project and import existing image/caption pairs into it."""

from __future__ import annotations

import click
import tqdm

from ..common.log import setup_logging
from .project import Project


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


@main.command("import")
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("sources", nargs=-1, required=True, type=click.Path(exists=True))
@click.option(
    "--include-hidden", is_flag=True, help="Also import hidden (dot) files and directories, which are skipped by default."
)
@click.option("--recursive", "-r", is_flag=True, help="Also import images in subdirectories.")
@click.option("--no-captions", is_flag=True, help="Import the images only, ignoring caption files next to them.")
@click.option("--verbose", "-v", is_flag=True, help="Debug logging.")
def import_(
    project_dir: str, sources: tuple[str, ...], recursive: bool, include_hidden: bool, no_captions: bool, verbose: bool
) -> None:
    """Import images from SOURCES (directories, with -r also their subdirectories, or single files) into the project at
    PROJECT_DIR.

    A same-stem UTF-8 .txt next to an image becomes its imported caption, unless --no-captions. Images already in the project are skipped
    with their edits kept, so an interrupted import can simply be run again.
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
                sources, captions=not no_captions, recursive=recursive, include_hidden=include_hidden, progress=progress
            )
        except ValueError as e:
            raise click.ClickException(str(e)) from e
    click.echo(f"Imported {imported} images, skipped {skipped} already in the project; {len(project.items)} in total.", err=True)


if __name__ == "__main__":
    main()
