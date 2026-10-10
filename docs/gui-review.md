# Duplicate review GUI (PySide6)

Status: implemented (`src/clip_image_deduper/gui`, `clip-image-deduper-review`). The text below is the design it
follows; deviations are noted inline.

A reviewer for the cases the CLI must not decide on its own: pairs that are clearly related but not bit-identical
(lossy copies, downscales, crops, variants of the same picture). The CLI stays the batch tool; the GUI reads the same
SQLite database, shows duplicate groups side by side, records the reviewer's decisions and applies them by moving
files to the trash directory, exactly like `dedupe --trash-dir`.

Conventions are borrowed from the picture dataset editor in `../scripts/python-src/neoscripts/pic-dataset-editor`
(PyQt6 there, PySide6 here; the API differs only in `Signal`/`Slot` names): single-letter shortcuts that never
intercept text inputs, Z/X navigation, C to cycle a status, wheel to navigate and Ctrl+wheel to zoom, a `QThreadPool`
of `QRunnable` jobs with a generation counter for anything that touches the disk, an in-memory thumbnail cache, an
F1 help dialog that is the single source of truth for the key map, and offscreen `unittest` checks.

## Scope

In:

* Review groups found by the existing matcher at a **review threshold** (looser than the automatic one), including
  the pair distances inside each group.
* Compare any two members of a group at full resolution with synchronized zoom/pan, instant A/B flip, and a
  difference overlay.
* Decide per group which members to keep; decisions persist immediately and survive restarts.
* Apply decisions: move the losers to a trash directory, never delete. Dry-run preview first.
* Keeping policies (`policies.toml`) pre-select a winner so the common case is one keystroke.

Out (for now): editing images, cross-directory `dedupe-import` review, running the encoder from the GUI (run
`update-db` first; the GUI refuses a database whose model does not match and says so), remote/web UI.

## Data model

`src/clip_image_deduper/review.py` (no Qt, testable without a display):

* `DuplicateGroup`: `members: list[int]` (DB row indices), `edges: list[tuple[int, int, float]]` (pairs under the
  review threshold with their distance), `min_distance`, `max_distance`. `find_duplicate_groups` currently throws
  the distances away after union-find; it gains a sibling that keeps the edges (or returns `DuplicateGroup`s), and
  the CLI keeps using the index lists.
* `Decision`: per group, `keep: set[path]`, `status: "pending" | "decided" | "skipped"`, `note: str`, timestamp.
  Keyed by the sorted tuple of member paths, so a group is recognized again after the matcher re-runs, as long as
  its membership did not change; a changed group comes back as pending with the old decision shown as a hint.
* `ReviewSession`: a **separate SQLite file**, `<db>.review.sqlite`, next to the embedding database. A review of a
  100k-image library can hold tens of thousands of groups and hundreds of thousands of members and edges; that is
  hundreds of times what the dataset editor's JSONL handles, and a JSONL would have to be fully parsed on open and
  rewritten on close. SQLite gives indexed lookups, per-decision commits (WAL, `synchronous = NORMAL`, as in
  `db_store.py`) and a file that survives `--force-update` rebuilding the embedding database. Tables:

  ```
  meta      key, value                       schema_version, review_threshold, auto_threshold, policy, model_id,
                                             image_dir, created_at
  groups    id, key, n, min_distance, max_distance, status, note, updated_at, pixel, touched
                                             key = sha256 of the sorted member paths; UNIQUE
                                             pixel: same / different / NULL (unchecked); touched: changed by
                                             the reviewer (an untouched group's status is re-derived per run)
  members   group_id, path, keep, width, height, size, format, mtime, distance_to_winner
                                             PRIMARY KEY (group_id, path); INDEX (path)
  edges     group_id, path_a, path_b, distance, pixel, pixel_peak, pixel_mean
                                             pixel check per pair; reused on re-runs when the files are unchanged
  history   id, group_id, at, keep_json, status, note
                                             append-only; undo pops the last row of a group
  applied   path, trash_path, group_id, at   one row per moved file; "Undo apply" reads it back
  ```

  Re-matching upserts groups by `key`: an unchanged group keeps its row, decision and history; a group whose
  membership changed gets a new key and the old row is marked `stale` (kept for reference, filtered out by default).
  The `members.path` index is what makes "which group is this file in" and the file-changed guard cheap.
* Thumbnails and full images are read from the image directory; nothing from the review is written there.

Matching runs once at session start (or on "Re-match" after changing the threshold) on the GPU through the existing
`DistanceIndex`, in a background job with the usual progress callback. 100k images take a second; the GUI does not
need incremental matching.

## Layout

`QMainWindow`, toolbar on top, `QSplitter` with three panes, status bar at the bottom.

```
┌ toolbar: Open DB · Re-match · Threshold [0.10 ▲▼] · Policy [largest ▾] · Apply… · Help ┐
├───────────────┬─────────────────────────────────────────────┬───────────────────────────┤
│ groups        │ comparison canvas                           │ group panel               │
│ (QListView)   │  ┌──────────────┐ ┌──────────────┐          │  members (QListWidget):   │
│  ▣ 3 · 0.42   │  │   A (keep)   │ │     B        │          │   1 ▣ a/foo.png  4000×6000│
│  ▢ 2 · 0.18   │  │              │ │              │          │     PNG 12.3 MiB  d=0.00  │
│  ✓ 2 · 1.90   │  └──────────────┘ └──────────────┘          │   2 ▢ b/foo.jpg  2000×3000│
│  ✗ 4 · 0.07   │  side-by-side | flip | diff                 │     JPEG 1.1 MiB  d=0.42  │
│  ...          │                                             │  distance matrix (small)  │
│               │                                             │  note: [__________]       │
│               │                                             │  status: pending (C)      │
├───────────────┴─────────────────────────────────────────────┴───────────────────────────┤
│ Group 12 / 418 · 37 decided · 2 skipped · 71 files to trash · Ready                     │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

* **Groups list**: one row per group with a status badge (pending red, decided green, skipped cyan, applied grey,
  drawn by a `QStyledItemDelegate` like the editor's U/A/S badges), member count and the group's minimum distance.
  Filter box above it: all / pending / decided / by distance band. Sorted by min distance by default, so the
  near-certain ones come first and the questionable band is at the end.
* **Canvas**: two `QGraphicsView`s sharing one transform (zoom and pan are applied to both), with three modes:
  side by side, flip (one view, A/B swapped on a key), diff (absolute difference, amplified, of A and B resampled to
  the same size). Member numbers 1-9 pick which image is A; Shift+number picks B. Fit / 100% as toolbar actions.
* **Group panel**: numbered members with a keep checkbox, resolution, format, file size, mtime, distance to the
  policy's winner, and the full pairwise matrix for groups up to 6; above that the policy's pre-selection is shown
  and can be re-applied. Free-text note and the status.
* **Status bar**: fixed position label on the left (`Group i / n`), counters, last message.

Large images: thumbnails via `QImageReader.setScaledSize` (fast JPEG downscale on decode); the canvas decodes the
full file off-thread with Pillow and the same leniency flags as the encoder (`MAX_IMAGE_PIXELS = None`, truncated
files allowed), converts to `QImage` and keeps the full images in a small LRU. The window remembers the direction
of the last navigation and prefetches the A/B pair of the next `--prefetch` groups (default 3) that way in a
separate thread pool, so a steady Next/Next/Next never waits for the disk and the current group's own reads never
queue behind prefetch reads (cold reads from a network share take seconds each); in diff mode the diffs of those
pairs are prefetched too. Turning around drops the queued prefetches that are now behind. Thumbnail cache ~300 entries like the editor. Files that fail to load show a `[!]` and keep the group reviewable.

## Key map

Single-letter keys are swallowed by an application-level event filter only when no text or numeric input has focus,
as in the editor. Everything is also reachable from the toolbar / menu, and F1 lists it all.

| key | action |
|---|---|
| Z / X, wheel up / down, Alt+Left / Right | previous / next group |
| Shift+Z / Shift+X | previous / next **pending** group |
| 1-9 | show member n as A; Shift+1-9 as B |
| Space (hold) | flip: show B in place of A while held |
| D | cycle canvas mode: side by side → flip → diff |
| K | toggle keep on the member currently shown as A |
| Shift+K | keep only A (everything else to trash) |
| P | reset the keeps of all pending groups to the policy chosen in the toolbar (confirmation dialog; U undoes per group) |
| Shift+P | reset this group's keeps to the policy's choice |
| Enter | mark group decided and go to the next pending one |
| C | cycle status pending → decided → skipped |
| S | skip (same as cycling to skipped) and advance |
| U | undo the last decision change (per-group history, like the editor's Undo Image) |
| F | focus the note field; Esc returns to the canvas |
| Ctrl+wheel | zoom (both views); Space+drag or middle drag pans |
| Home / End, 0 | fit to view / 100% |
| Ctrl+S | no-op kept for muscle memory: every decision is committed to the review database as it is made |
| Ctrl+Enter | Apply… (dry-run dialog listing every move, then confirm) |
| F1 | help |

`Delete` is deliberately unbound: nothing in the GUI deletes, it only marks and then moves to trash on Apply.

## Apply

The Apply dialog lists `keep → trash` per decided group, with totals (files, bytes), runs `move_to_trash` from
`dedupe.py` for each loser in a background job with progress, and records `applied` in the session with the trash
paths in the `applied` table so an "Undo apply" can move them back as long as the trash directory is untouched. Groups whose files changed
on disk since matching (mtime or missing) are refused and flagged; re-run `update-db` and Re-match.

## Thresholds

The review threshold is the user's choice, made after looking at the `calibrate` histograms, and given once to
`dedupe --review-threshold`; it is recorded in the review database's `meta` and shown by the GUI and
`review-status`. Which groups start pre-decided with the policy's choice is the pixel check's call: those whose
every pair differs by compression only, at any distance up to the review threshold. Groups with a pixel difference
start pending even within the automatic threshold, because that is where artist revisions and pages of one pixiv
set hide. With `--no-pixel-check` the automatic threshold of the CLI (0.1 by default) decides instead, as
`dedupe` would have merged on its own. Changing either value means re-running the
match from the CLI; decisions of groups whose membership did not change survive that.

`calibrate` deliberately does not feed the GUI: its suggestion only exists when the variant and different-image
distributions do not overlap, and it answers "how loose to catch these synthetic variants", not "how loose is
worth a human look", which may well be looser.

## Code layout and packaging

```
src/clip_image_deduper/review.py          groups with edges, Decision, ReviewSession on SQLite (no Qt)
src/clip_image_deduper/gui/__init__.py
src/clip_image_deduper/gui/app.py         main(), QApplication, CLI args (--db, --image-dir, --trash-dir, --threshold)
src/clip_image_deduper/gui/window.py      QMainWindow, toolbar, event filter, key map, help text
src/clip_image_deduper/gui/models.py      GroupListModel, MemberListModel, thumbnail cache + jobs
src/clip_image_deduper/gui/canvas.py      CompareCanvas (two linked views, flip, diff)
src/clip_image_deduper/gui/jobs.py        Job(QRunnable) + Signals, image loading helpers
tests/test_review.py                      session persistence, group identity, decision undo (no Qt)
tests/test_gui.py                         QT_QPA_PLATFORM=offscreen: key map, focus rules, flip/diff, apply dry-run
```

`pyproject.toml`: optional dependency group `gui = ["PySide6>=6.7"]`, script entry
`clip-image-deduper-review = "clip_image_deduper.gui.app:main"`. The core package keeps zero Qt imports so the CLI
install stays small. Tests for the GUI run under `QT_QPA_PLATFORM=offscreen` and are skipped when PySide6 is not
installed.

## Deviations from the plan

* Apply runs synchronously with a progress dialog rather than in a job: moves are quick and must not interleave
  with decisions.
* Ctrl+S is unbound (every decision commits immediately).

## Milestones (as planned)

1. `review.py`: groups with edges, the review SQLite schema, decisions, undo, re-match upsert by group key. Tests.
   CLI gains `--review-threshold` on `dedupe` that writes groups in the band to the review database instead of
   trashing them, so the GUI has something to open, and a `review-status` subcommand that prints the counters.
2. Window skeleton: open DB + image dir, group list with badges, member panel, status bar, navigation keys, session
   persistence. Static side-by-side canvas.
3. Canvas: linked zoom/pan, flip, diff, full-resolution off-thread loading, LRU.
4. Decisions: keep toggles, policy pre-selection, C/Enter/S/U, filters.
5. Apply dialog with dry-run and undo-apply; file-changed guards.
6. Help dialog, README section.
