# TODO

## Everything on projects

Import → deduplication → culling → optional batch tagging → the dataset editor (fixing tags, adding
natural-language passages for Anima-style captions). Tagging runs once: before culling when its scores should also
drive tag matching, otherwise after it, so that excluded images are not tagged.

A project is also how a library is cleaned: a library is a project over the library folders, and deduplicating it
is deduplicating that project. Only `encode-test`, the tagger's `run` and the import side of `dedupe-import` work
on plain directories. Sources stay read-only in the sense that image contents are never modified;
moving a discarded image into the project's trash is allowed.

Project identity, stat-only import and sync, storage, per-field decisions with history, rollback and GC, databases
keyed by item ID, content hashes in the embedding database (reuse, source checks), deduplication on projects with
the exclusion combined on read, `dedupe-import` against a project, `picurate-curate` with its Duplicates tab, the trash
in GC, and tagging on projects with captions combined on read are in place ([projects](project.md),
[tagging](tagger.md#tagging-a-project)). Still open:

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
