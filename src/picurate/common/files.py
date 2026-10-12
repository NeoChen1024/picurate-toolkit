"""Writing files next to the user's images without ever leaving a half-written one."""

from __future__ import annotations

import os
from pathlib import Path


def write_text_atomic(path, text: str) -> None:
    """Write ``text`` (UTF-8) to a temporary name next to ``path`` and rename it into place, so an interruption leaves
    either the old file or the new one."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.picurate-partial")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
