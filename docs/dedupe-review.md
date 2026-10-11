# Duplicate review (`picurate-dedupe-review`)

A single threshold cannot tell a lossy copy from an artist revision or the next page of a pixiv set. The review GUI
is for those groups: the CLI finds them, a human decides, and the decisions are applied by moving files to the trash
directory exactly like `dedupe --trash-dir`. It needs the `gui` extra (`uv pip install -e '.[gui]'`).

```shell
picurate-dedupe dedupe -i pictures -d pictures.sqlite --review-threshold 1.5   # stores groups, moves nothing
picurate-dedupe-review -d pictures.sqlite --trash-dir trash                    # decide, then Apply
picurate-dedupe review-status -d pictures.sqlite                               # or from the CLI:
picurate-dedupe review-apply -i pictures -d pictures.sqlite --trash-dir trash
```

## Matching into the review database

`dedupe --review-threshold T` matches at `T` and writes every group into `pictures.review.sqlite` (next to the
embedding database) instead of moving anything. `T` is your choice after looking at the `calibrate` histograms; it
may well be looser than anything `calibrate` would suggest, since it answers "worth a human look", not "safe to
merge".

* Groups whose every pair passes the pixel check start *decided*, with the keeping policy's winner ticked, whatever
  their embedding distance.
* Every other group starts *pending*, even one within the normal `--threshold`: that is how artist revisions and
  pages of one pixiv set reach a human.
* With `--no-pixel-check`, groups entirely within `--threshold` start decided instead, as `dedupe` would have merged
  them on its own.

Groups are keyed by their member paths. Re-running the match (after adding images, or with another threshold) keeps
the decisions of groups whose membership did not change and marks the others stale. Groups nobody has touched get
their starting status re-derived on every run, and pixel results are reused for files that did not change.

## The window

Group list on the left (surest first, a status badge per row), two linked image views in the middle (side by side,
flip while holding Space, or an amplified difference), members with thumbnails and keep checkboxes on the right.
Groups that passed the pixel check are marked ≈; the Show box filters by status, by pixel result, and finds groups
that keep nothing or are discarded.

The keys you need most (F1 lists all of them):

| key | action |
|---|---|
| Z / X | previous / next group (Shift: next pending one) |
| 1-9, Shift+1-9 | show member n as A / as B |
| Space (hold), D | flip A/B; cycle side by side → flip → diff |
| K, Shift+K | toggle keep on A; keep only A |
| Enter | mark decided and go to the next pending group |
| R | discard the whole group (every member is moved); R again takes it back |
| P, Shift+P | reset keeps of all pending groups / this group to the toolbar's policy |
| U | undo the last change of this group |
| Ctrl+Enter | Apply |

Single-letter keys never fire while a text field has focus. The window loads the next `--prefetch` groups (default
3) in the direction you are moving, diffs included in diff mode, so stepping through groups does not wait for the
disk.

## Safety

* Every decision is committed to the review database as it is made, with a per-group history for undo.
* A group can only be decided while it keeps a member. Moving every member takes R on purpose; a group whose keeps
  were all unticked is flagged and skipped by Apply.
* **Apply** lists every move first, then moves the losers of decided groups to the trash directory. Groups whose
  files changed on disk since matching are refused. An applied group can be undone while the trash directory is
  untouched. Nothing is ever deleted, and `Delete` is not bound to anything.

## Review database

A separate SQLite file, so it survives `--force-update` rebuilding the embedding database and stays fast with tens
of thousands of groups:

```
meta      key, value                       schema_version, review_threshold, auto_threshold, policy, model_id, image_dir
groups    id, key, n, min_distance, max_distance, status, note, updated_at, pixel, touched, discard
                                           key = sha256 of the sorted member paths; pixel = same / different / NULL;
                                           touched = changed by the reviewer; discard = move every member
members   group_id, path, keep, width, height, size, format, mtime, distance_to_winner
edges     group_id, path_a, path_b, distance, pixel, pixel_peak, pixel_mean
history   id, group_id, at, keep_json, status, note, discard      append-only; undo pops a group's last row
applied   path, trash_path, group_id, at                          one row per moved file, for undoing an apply
```

`status` is `pending`, `decided`, `skipped`, `applied` or `stale`. Cross-directory `dedupe-import` results are not
reviewable yet, and the GUI does not run the encoder: update the database from the CLI first.
