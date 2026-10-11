"""Logging setup that cooperates with tqdm progress bars.

Library modules log through the standard ``logging`` module and never touch progress bars. The CLI installs
:class:`TqdmHandler`, which routes records through ``tqdm.write`` so they appear above any active bar instead of
tearing it.
"""

from __future__ import annotations

import logging

import tqdm


class TqdmHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.tqdm.write(self.format(record))
        except Exception:  # pragma: no cover - logging must never raise
            self.handleError(record)


def setup_logging(verbose: bool = False, packages: tuple[str, ...] = ("clip_image_deduper",)) -> None:
    for package in packages:
        root = logging.getLogger(package)
        root.setLevel(logging.DEBUG if verbose else logging.INFO)
        if not any(isinstance(h, TqdmHandler) for h in root.handlers):
            handler = TqdmHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            root.addHandler(handler)
        root.propagate = False
