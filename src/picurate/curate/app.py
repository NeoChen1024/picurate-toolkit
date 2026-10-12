"""``picurate-curate``: decide about a project's images in the GUI."""

from __future__ import annotations

import sys

import click

from ..common.log import setup_logging
from ..dedupe.keeping import PolicyError, load_policies


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--keeping-config",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="TOML file adding/overriding the keeping policies offered in the Policy box.",
)
@click.option(
    "--prefetch",
    type=click.IntRange(0),
    default=3,
    show_default=True,
    help="Groups to load ahead in the direction you are moving (each keeps two full images, and in diff mode their diff, in memory).",
)
@click.option("--verbose", "-v", is_flag=True, help="Debug logging.")
def main(project_dir: str, keeping_config: str | None, prefetch: int, verbose: bool) -> None:
    """Decide about the images of the project at PROJECT_DIR: which copies of duplicate groups (found by
    `picurate-dedupe dedupe`) to keep, and which images to cull by text query, tag condition or example image. Needs no
    other picurate-curate or dedupe run on the project."""
    setup_logging(verbose)
    try:
        policies = load_policies(keeping_config)
    except (PolicyError, OSError) as e:
        raise click.ClickException(f"Cannot load keeping policies: {e}") from e

    from PySide6.QtWidgets import QApplication

    from .window import CurateWindow

    app = QApplication.instance() or QApplication(sys.argv[:1])
    try:
        window = CurateWindow(project_dir, policies=policies, prefetch=prefetch)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
