# TODO

## Everything on projects

Import → deduplication → culling → optional batch tagging → the dataset editor (fixing tags, adding
natural-language passages for Anima-style captions). Tagging runs once: before culling when its scores should also
drive tag matching, otherwise after it, so that excluded images are not tagged.

A project is also how a library is cleaned: a library is a project over the library folders, and deduplicating it
is deduplicating that project. The standalone directory modes of `dedupe`, `update-db`, `calibrate`, the review
GUI and the tagger's `update`/`export` go away; `encode-test`, the tagger's `run` and the import side of
`dedupe-import` stay plain directories. Sources stay read-only in the sense that image contents are never modified;
moving a discarded image into the project's trash is allowed.

Project identity, stat-only import and sync, storage, per-field decisions with history, rollback and GC, databases
keyed by item ID, and content hashes in the embedding database (reuse, source checks) are in place ([projects](project.md)). Still open:

* [ ] **Stages write only their own files; reading combines them** (one implementation in `picurate.project`, used by
  the editor, the tagger and export):
  * Excluded = (losers of decided groups in `review.sqlite`) overridden by the human exclusion decision (force keep
    or force exclude), with the reason (review group and the kept image, or the human reason).
  * Caption = the human caption if edited; otherwise the imported caption and the one rendered from `tags.sqlite`,
    combined by `skip` (default: keep an existing caption), `overwrite`, `append` or `merge` (union without
    duplicates, compared like the editor's duplicate highlighting). The caption options (style, thresholds, this mode)
    are the tagger's arguments, recorded in `tags.sqlite`.
* [ ] **Deduplication on projects only** (`--project`): `embeddings.sqlite` and `review.sqlite` in the project. Items
  excluded by a human are not encoded or matched; review losers still take part, so groups stay stable; rows of items
  no longer in the project are dropped. `--review-threshold` defaults to `--threshold`: automatic groups are written
  as decided, and the losers of decided groups count as excluded. Nothing is moved by `dedupe` or the GUI;
  Apply, undo-apply, `review-apply` and the `applied` table go away.
* [ ] **`picurate-curate PROJECT_DIR`**, one GUI for what a human decides about a project's images, in a new
  `picurate.curate` package (the `dedupe` package keeps the CLI and the databases). It replaces
  `picurate-dedupe-review`. Shared: the project, thumbnail and image loading, and every image marked with what
  excluded it (a duplicate group, a culling query, a human). The Duplicates tab is the current review window, keys
  unchanged. The Culling tab (below) loads embeddings, the text tower and tag scores only when first opened.
* [ ] **`dedupe-import`: a directory against a project.** `-i NEW_DIR --project LIB --trash-dir T` moves images of
  `NEW_DIR` already in `LIB` (its items not excluded) to `T`. `--import-db` becomes optional (without it the import
  side is encoded in memory). Duplicates within `NEW_DIR` stay out of scope.
* [ ] **Trash in GC.** `picurate-project gc --trash-excluded` moves excluded images to `P/trash/<source>/<path>`
  (`--only dedupe` limits it to review losers), as one operation that rollback moves back; `-n` lists first. A
  trashed image whose file was later removed from the trash by hand no longer exists: GC removes the item with its
  state and history, the stages drop its rows on their next update, and a rollback skips and lists it.
* [ ] **Tagging on projects** (`update --project`): scores into the project's `tags.sqlite`, excluded items skipped.
  Captions are derived on read (above); `export --project` can still write `.txt` next to the source images, for
  training on a library folder directly.
* [ ] **Exclusion in the editor**: thumbnails marked excluded with the reason, and a control for automatic / force
  keep / force exclude.
* [ ] **Culling tab** of `picurate-curate`, interactive, on the project's embeddings and tag scores:
  * Manual selection; image similarity to example images (cosine); text similarity to typed queries; tag matching.
    Several examples, queries and tag conditions combine; the threshold is applied live to computed scores.
  * Embeddings are held in RAM as FP16 (the same as deduplication needs) with per-row FP32 norms, and multiplied on
    the CPU in blocks converted to FP32 small enough to stay in cache: about 150 ms per query vector for a million
    images. The GPU only encodes the query images and texts.
  * Applying writes human exclusion decisions with the query as the reason, as one operation.
  * Text similarity needs SigLIP2's text tower from transformers (`google/siglip2-so400m-patch16-naflex`). First
    check that its image tower gives the same embeddings as timm's, and that text queries rank real images sensibly.

## Other

* [ ] Store a mean-centered copy or normalized embeddings to make thresholds model-independent? (`calibrate` measures
  the scale per model for now.)
* [ ] Train a custom model for anime image comparison, and later anime semantic search? Notes in
  [training-clip.md](training-clip.md), scripts under `src/clip_training`.
