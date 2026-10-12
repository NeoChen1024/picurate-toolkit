"""``picurate-dedupe-review``: open a review database in the GUI."""

from __future__ import annotations

import os
import sys

import click

from ...common.log import setup_logging
from ..keeping import PolicyError, load_policies
from ..review import default_review_path


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--db",
    "-d",
    type=click.Path(dir_okay=False),
    default=None,
    help="Embedding database; its review database is <db>.review.sqlite.",
)
@click.option("--review-db", type=click.Path(dir_okay=False), default=None, help="Review database path (overrides --db).")
@click.option(
    "--image-dir",
    "-i",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Image directory (default: the one recorded in the review database).",
)
@click.option(
    "--trash-dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Where Apply moves the losers (asked for on first Apply otherwise).",
)
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
def main(
    db: str | None,
    review_db: str | None,
    image_dir: str | None,
    trash_dir: str | None,
    keeping_config: str | None,
    prefetch: int,
    verbose: bool,
) -> None:
    """Review duplicate groups found by `picurate-dedupe dedupe --review-threshold`."""
    setup_logging(verbose)
    path = review_db or (default_review_path(db) if db else None)
    if path is None:
        raise click.UsageError("Give --db or --review-db.")
    if not os.path.exists(path):
        raise click.ClickException(f"No review database at {path}. Run dedupe --review-threshold first.")
    try:
        policies = load_policies(keeping_config)
    except (PolicyError, OSError) as e:
        raise click.ClickException(f"Cannot load keeping policies: {e}") from e

    from PySide6.QtWidgets import QApplication

    from .window import ReviewWindow

    app = QApplication.instance() or QApplication(sys.argv[:1])
    try:
        window = ReviewWindow(path, image_dir=image_dir, trash_dir=trash_dir, policies=policies, prefetch=prefetch)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
