# picurate-toolkit — Agent guide

Deduplication, tagging and dataset editing for image collections of millions of files. Batch work happens on the
command line; a GUI exists only where a human has to look. The tools are separate commands until the workflow shows
how they should feed each other.

## Sources of truth

- [README](README.md): what the project is, the tools, installation and a quick start.
- [Deduplication](docs/dedupe.md), [duplicate review](docs/dedupe-review.md) and [tagging](docs/tagger.md): behavior,
  databases and the reasoning behind defaults.
- [TODO](docs/TODO.md): agreed future work.
- Each command's `--help` owns the complete option list.

## Documentation

- Describe current behavior and explicitly agreed plans; keep them apart. Give each topic one primary home and link
  to it elsewhere instead of repeating it.
- Keep what a user needs to use the tools and judge their results: database contents, matching and pixel-check rules,
  caption layouts, and the measured numbers that justify a default (threshold distances, memory per image,
  throughput order of magnitude). Leave out source walkthroughs and details that only restate the code.
- No decision histories, comparisons with previous versions or defaults, debugging notes, benchmark logs or test
  transcripts. Report validation evidence in the task response. Git history is the archive: remove superseded plans
  and stale references when behavior changes.
- No local paths, hostnames or detailed personal hardware. "An RTX 4080-class GPU" is enough context for a number.
- The README is written in the author's first-person voice, with opinions. Keep that personality; do not polish it
  into neutral product copy. Keep it short and move detail into `docs/`.

## Data safety

- Image collections and datasets are the user's originals. Tools only move files into a trash directory the user
  named; nothing is ever deleted, and dry runs move nothing. Never alter source images as a side effect.
- Never rewrite existing databases or caption files just to adopt a new default. Databases are bound to the model
  (and its settings) they were made with; a mismatch is refused unless the user forces a rebuild.
- Never scan the filesystem broadly (`find /`, recursive searches of the home directory): the machine has spinning
  disks. Search the repository or the directory the user named, and ask if something is not there.

## Scope

- Prefer clear computation and fast iteration over compatibility and speculative infrastructure. Do not keep
  obsolete interfaces alive for tests or for old project files unless asked.
- Defaults stay conservative: automatic actions only where they are safe; borderline cases go to a human (the review
  GUI) instead of a looser default.
- Do not add provenance bundles, generic recovery frameworks or plugin systems without an explicit request. Keep the
  mechanisms the workflows need (resumable per-batch commits, model binding, undo in the review database).

## Language and style

- Project files are in English, including documentation and comments. Conversation with the user need not be.
- Python 3.11+, Black (132 columns) and isort (Black profile) per `pyproject.toml`.
- CLIs use Click and install with the `picurate-` prefix. Progress uses tqdm, with logging routed through it
  (`picurate.common.log`); progress and diagnostics go to stderr.
- GUIs use PySide6 and live in the `gui` extra; command-line tools never import Qt.

## Code ownership and dependencies

- One import package, `picurate` under `src/`: `common` (image walking, the decode pipeline, logging), `dedupe`
  (including its review GUI), `tagger`, and `dataset_editor` once ported. `src/clip_training` holds the separate
  CLIP fine-tuning scripts.
- Look in `picurate.common` before writing another image walker, loader or pipeline. Share mechanisms with identical
  semantics; keep each tool's policy in the tool.
- Runtime dependencies live in `pyproject.toml` with minimum versions only; heavy optional ones go in extras. No
  lockfile.
- Code imported from elsewhere (for example the author's scripts repository) names its origin in the commit
  message. Original code and documentation are GPL-3.0-or-later; keep third-party notices.

## Local toolchain and verification

- Use the repository-root `.venv`, managed by uv: `uv pip install --python .venv/bin/python -e '.[gui]'`. Never
  fall back to the system Python or create other environments.
- Do not add or expand automated tests unless the user asks for them. The suite covers only contracts whose breakage
  would go unnoticed: which files get moved, database binding, resume after interruption, numerical exactness, and
  caption layouts that must match other tools. Remove tests that only restate option validation or widget details.
  Run it with `QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tests`.
- Validate changes on real images and inspect the actual output (moved files, captions, database rows). A round trip
  through the same code is not independent evidence. Report what was actually checked.
