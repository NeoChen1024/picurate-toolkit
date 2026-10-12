"""``picurate-dataset-editor``: open the editor, optionally on an existing project directory."""

from __future__ import annotations

import sys

import click


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("project_dir", required=False, type=click.Path(exists=True, file_okay=False))
def main(project_dir: str | None) -> None:
    """Edit captions, crops and watermark fills of a training dataset; PROJECT_DIR is a project, not the images."""
    from PySide6.QtWidgets import QApplication

    from .window import Window

    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("picurate-dataset-editor")
    window = Window(project_dir)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
