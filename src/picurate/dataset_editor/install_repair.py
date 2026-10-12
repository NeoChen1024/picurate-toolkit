"""Install Hi-SAM and SAM 2 from pinned upstream source revisions into the running interpreter's environment.

Run after installing the ``repair`` extra: ``python -m picurate.dataset_editor.install_repair``. Neither project
publishes a usable package (Hi-SAM has no packaging at all, SAM 2's build wants to compile a CUDA extension the editor
does not need), so both are fetched with git and installed with uv, without dependencies. No weights are downloaded;
the first use fetches pinned files through the Hugging Face Hub.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

SOURCES = [
    ("Hi-SAM", "https://github.com/ymy-k/Hi-SAM", "69009434d4dba5541f228d8f5acb0754c333d417"),
    ("sam2", "https://github.com/facebookresearch/sam2", "2b90b9f5ceec907a1c18123530e92e794ad901a4"),
]
HI_SAM_PYPROJECT = """[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"
[project]
name = "hi-sam"
version = "0.0.1"
[tool.setuptools.packages.find]
include = ["hi_sam*"]
"""


def run(*args, **kwargs) -> None:
    subprocess.run(list(map(str, args)), check=True, **kwargs)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="picurate-repair-install-") as temporary:
        for name, url, revision in SOURCES:
            source = Path(temporary) / name
            run("git", "init", "-q", source)
            run("git", "-C", source, "fetch", "--depth", "1", url, revision)
            run("git", "-C", source, "checkout", "--detach", "FETCH_HEAD")
            if name == "Hi-SAM":
                (source / "pyproject.toml").write_text(HI_SAM_PYPROJECT)
            run(
                "uv", "pip", "install", "--python", sys.executable, "--no-deps", "--no-build-isolation", source,
                env={**os.environ, "SAM2_BUILD_CUDA": "0"},
            )  # fmt: skip
    print("Installed Hi-SAM and SAM 2.")


if __name__ == "__main__":
    main()
