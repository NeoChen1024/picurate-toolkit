# TODO

## Dataset curation on one project

Import → deduplication → culling → optional batch tagging → the dataset editor (fixing tags, adding
natural-language passages for Anima-style captions). Tagging runs once: before culling when its scores should also
drive tag matching, otherwise after it, so that excluded images are not tagged. Every stage stays usable on its own,
without a project.

Project identity, storage, per-field decisions with history, rollback and GC, and databases keyed by item ID are in
place ([projects](project.md)). Still open:

* [ ] **Stages write only their own files; reading combines them** (one implementation in `picurate.project`, used by
  the editor, the tagger and export):
  * Excluded = (losers of decided groups in `review.sqlite`) overridden by the human exclusion decision.
  * Caption = the human caption if edited; otherwise the imported caption and the one rendered from `tags.sqlite`,
    combined by `skip` (default: keep an existing caption), `overwrite`, `append` or `merge` (union without
    duplicates, compared like the editor's duplicate highlighting). The caption options (style, thresholds, this mode)
    are the tagger's arguments, recorded in `tags.sqlite`.
* [ ] **Deduplication in a project** (`--project`): `embeddings.sqlite` and `review.sqlite` in the project, items
  instead of a directory walk, items excluded by a human skipped (review losers still take part, so groups stay
  stable). Nothing is moved: automatic groups are written to the review database as decided, and the losers of
  decided groups count as excluded, with no Apply step. The standalone behaviour (`--trash-dir`) is unchanged.
* [ ] **Tagging in a project** (`--project`): scores into the project's `tags.sqlite`, excluded items skipped, no
  `.txt` written next to the sources; captions are derived on read (above).
* [ ] **Culling GUI**, interactive, on the project's embeddings and tag scores:
  * Manual selection; image similarity to example images (cosine); text similarity to typed queries; tag matching.
    Several examples, queries and tag conditions combine; the threshold is applied live to computed scores.
  * Embeddings are held in RAM as FP16 (the same as deduplication needs) with per-row FP32 norms, and multiplied on
    the CPU in blocks converted to FP32 small enough to stay in cache: about 150 ms per query vector for a million
    images. The GPU only encodes the query images and texts.
  * Applying writes human exclusion decisions with the query as the reason, as one operation.
  * Text similarity needs SigLIP2's text tower from transformers (`google/siglip2-so400m-patch16-naflex`). First
    check that its image tower gives the same embeddings as timm's, and that text queries rank real images sensibly.
* [ ] Later: reuse embeddings across projects and collections by content hash and model.

## Other

* [ ] Store a mean-centered copy or normalized embeddings to make thresholds model-independent? (`calibrate` measures
  the scale per model for now.)
* [ ] Train a custom model for anime image comparison, and later anime semantic search? Notes in
  [training-clip.md](training-clip.md), scripts under `src/clip_training`.
