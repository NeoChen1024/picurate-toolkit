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
keyed by item ID, content hashes in the embedding database (reuse, source checks), deduplication on projects with
the exclusion combined on read, `dedupe-import` against a project, `picurate-curate` with its Duplicates tab, and the trash
in GC are in place ([projects](project.md)). Still open:

* [ ] **Captions combined on read** (one implementation in `picurate.project`, used by the editor, the tagger and
  export): the human caption if edited; otherwise the imported caption and the one rendered from `tags.sqlite`,
  combined by `skip` (default: keep an existing caption), `overwrite`, `append` or `merge` (union without
  duplicates, compared like the editor's duplicate highlighting). The caption options (style, thresholds, this mode)
  are the tagger's arguments, recorded in `tags.sqlite`.
* [ ] **Tagging on projects** (`update --project`): scores into the project's `tags.sqlite`, excluded items skipped.
  Captions are derived on read (above); `export --project` can still write `.txt` next to the source images, for
  training on a library folder directly.
* [ ] **Culling tab** of `picurate-curate`, interactive, on the project's embeddings and tag scores, loaded only when
  the tab is first opened; it marks every image with what excluded it, like the Duplicates tab:
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
