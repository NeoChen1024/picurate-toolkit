# Curation (`picurate-curate`)

A single threshold cannot tell a lossy copy from an artist revision or the next page of a pixiv set. `picurate-curate`
is where a human decides about a [project](project.md)'s images, in two tabs: **Duplicates**, for the groups
[`picurate-dedupe dedupe`](dedupe.md) found, and **Culling**, for removing whole kinds of images by text query, tag or
example. It needs the `gui` extra (`uv pip install -e '.[gui]'`).

```shell
picurate-dedupe dedupe library --review-threshold 1.5   # stores groups, moves nothing
picurate-curate library                                 # decide
```

Nothing is moved from here. The images a decided group does not keep count as excluded: the editor marks them and
does not export them, and [`picurate-project gc --trash-excluded`](project.md#trash) moves them out of the library. A person's decision in the [editor](dataset-editor.md) (force keep or force exclude) overrides
a group's, and the member panel shows it, as it shows a member excluded by another group. The window holds the
project's review database, so a `dedupe` run waits until it is closed; the editor can stay open alongside.

## Duplicates

Group list on the left (surest first, a status badge per row), two linked image views in the middle (side by side,
flip while holding Space, or an amplified difference), members with thumbnails, source directories and keep
checkboxes on the right. Groups that passed the pixel check are marked ≈; the Show box filters by status, by pixel
result, and finds groups that keep nothing or are discarded. A group that took over the keeps of a group you had changed says so.

The keys you need most (F1 lists all of them):

| key | action |
|---|---|
| Z / X | previous / next group (Shift: next pending one) |
| 1-9, Shift+1-9 | show member n as A / as B |
| Space (hold), D | flip A/B; cycle side by side → flip → diff |
| K, Shift+K | toggle keep on A; keep only A |
| Ctrl+A | keep every member (not duplicates after all) |
| Enter | mark decided and go to the next pending group |
| R | discard the whole group (every member is excluded); R again takes it back |
| P, Shift+P | reset keeps of all pending groups / this group to the toolbar's policy |
| U | undo the last change of this group |

Single-letter keys and Ctrl+A never fire while a text field has focus. The window loads the next `--prefetch` groups (default
3) in the direction you are moving, diffs included in diff mode, so stepping through groups does not wait for the
disk.

Every decision is committed to the review database as it is made, with a per-group history for undo. A group can only
be decided while it keeps a member: excluding every member takes R on purpose, and a group whose keeps were all
unticked is flagged and excludes nothing.

## Culling

Finds the images to drop from a project by what they show, and excludes them as human decisions. It works on what
the other stages stored, loaded when the tab is first opened: the embeddings of
[`picurate-dedupe update-db`](dedupe.md) (or `dedupe`) and the tag scores of
[`picurate-tag update`](tagger.md#tagging-a-project). Without one of them its criteria are not offered.

Criteria, any number, each with its own threshold:

* **Text query**: cosine similarity between SigLIP2's text embedding of the query (lowercased) and each image's
  stored embedding. Matching images score about 0.10-0.18 and the rest 0.02-0.06; the threshold starts at 0.12.
  Needs embeddings of the default model, `naflexvit_so400m_patch16_siglip.v2_webli`; the text tower
  (`google/siglip2-so400m-patch16-naflex`, the same checkpoint) is downloaded on first use.
* **Tag condition**: `tag` or `tag >= probability`, the tag's score in `tags.sqlite`; without a probability, the
  tagger's threshold for the tag's category. Images not tagged yet never match.
* **Example image**: cosine similarity to the current image's stored embedding (E), or to image files from anywhere,
  encoded with the image model. Two unrelated images already score about 0.7 (99.9th percentile 0.91), so the
  threshold starts at 0.90.

The selection is the images matching any (or all) of the ticked criteria, plus those put in by hand and minus those
left out (Space on the selected cells). The grid shows the selection or every image, sorted by the score of the
criterion selected in the list; T moves its threshold to the current image's score. Badges mark what already
excludes an image: **T** trash, **H** a person, **D** a duplicate group; **K** marks one a person kept.

**Exclude** writes a force exclude decision for every selected image no person decided about yet, as one operation
of the project's history: the reason is the query (`culling: any of text "monochrome manga page" >= 0.120; tag comic
>= 0.50`), or `culling: picked by hand`. Images a person kept or excluded stay as they are. **Undo** rolls the
operation back, as `picurate-project rollback` does. It needs the project's decisions to itself for that moment, so
it refuses while the [editor](dataset-editor.md) has the project open. Criteria and choices by hand are not saved.

Scores are computed on the CPU from the embeddings held in memory as FP16 (2.3 KiB per image): about 0.1-0.2 s per
criterion for a million images on an 8-core desktop CPU. Thresholds apply instantly. The GPU, when there is one,
only encodes query texts and example files.

## Review database

`review.sqlite` in the project, a file of its own so it survives `--force-update` rebuilding the embedding database
and stays fast with tens of thousands of groups. Members are [item IDs](project.md#sources-and-identity), stored with
the source ID and path they are derived from. A file of another schema version is refused: move it away and match
again.

```
meta      key, value                       schema_version, project_id, model_id, review_threshold, auto_threshold,
                                           policy, pixel_check
groups    id, key, n, min_distance, max_distance, status, note, updated_at, pixel, touched, discard, carried_from
                                           key = BLAKE3 of the sorted member IDs; pixel = same / different / NULL;
                                           touched = changed by the reviewer; discard = exclude every member;
                                           carried_from = the changed group whose keeps it started with
members   group_id, item, source_id, path, keep, width, height, size, format, mtime, distance_to_winner
edges     group_id, item_a, item_b, distance, pixel, pixel_peak, pixel_mean
history   id, group_id, at, keep_json, status, note, discard      append-only; undo pops a group's last row
```

`status` is `pending`, `decided`, `skipped` or `stale`. An item is excluded as a duplicate when a decided group does
not keep it, provided the group keeps a member or is discarded. `dedupe-import` results are not reviewable, and the
GUI does not run the encoder: run `dedupe` from the CLI first.
