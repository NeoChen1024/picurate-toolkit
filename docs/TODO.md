# TODO

## Dataset curation on one project

Import → deduplication → culling → optional batch tagging → the dataset editor (fixing tags, adding
natural-language passages for Anima-style captions). Tagging runs once: before culling when its scores should also
drive tag matching, otherwise after it, so that excluded images are not tagged. Every stage stays usable on its own,
without a project.

* [ ] **Identity.** An image is `(source_id, relative path)`. `source_id` is 256 random bits (never all zero)
  created when a source folder is first imported, kept in `project.json` (as hex, the only place IDs are text) and
  unchanged by relinking; outside a project the single image directory has the all-zero `source_id`, so there is
  one code path. The item ID is BLAKE3 in key-derivation mode with context `picurate item_id v1` over the 32 bytes
  of `source_id`, `"\0"` and the relative path (UTF-8 with surrogateescape), all 256 bits. IDs are stored as BLOBs;
  hex is only for display. Rows in every database keep `(id, source_id, path)`. IDs are only unique within one
  database: `dedupe-import` keeps the two sides apart.
* [ ] **Hashing.** BLAKE3 for every hash the project makes (content fingerprints, repair inputs and artifacts,
  review group keys); downloads are still checked with whatever their publisher uses (Big-LaMa: SHA-256).
* [ ] **Project storage.** `project.json` keeps the settings and the source folders; `project.sqlite` replaces
  `items.jsonl` and `edits.jsonl`:
  * `items`, written by import, committed per batch.
  * Human decisions, stored per field (caption, crop, fills, review status, exclusion: none / force keep / force
    exclude, with a reason): a current-state table and an append-only history table. History rows carry an
    operation ID, so one edit session, one culling apply or one batch replacement can be rolled back as a unit;
    fields changed again since are skipped and listed. A rollback is itself a new history entry. Auto-save while
    typing updates the current state; history is written when leaving an image or closing.
  * An unchanged caption is stored as NULL, meaning "use the derived one".
  * `picurate-project gc --keep-revisions N | --older-than AGE` prunes history and the LaMa artifacts nothing kept
    refers to, with `-n` to list first. Never automatic.
  * The editor holds an `flock` on the project so two editors cannot overwrite each other.
  * Schema version 3; older projects are refused, not converted.
* [ ] **Stages write only their own files; reading combines them** (one implementation in `picurate.project`, used by
  the editor, the tagger and export):
  * Excluded = (losers of decided groups in `review.sqlite`) overridden by the human exclusion decision.
  * Caption = the human caption if edited; otherwise the imported caption and the one rendered from `tags.sqlite`,
    combined by the project setting `skip` (default: keep an existing caption), `overwrite`, `append` or `merge`
    (union without duplicates, compared like the editor's duplicate highlighting).
* [ ] **Deduplication in a project** (`--project`): `embeddings.sqlite` and `review.sqlite` in the project, items
  instead of a directory walk, excluded items skipped. Nothing is moved: automatic groups are written to the review
  database as decided, and the review GUI's Apply marks losers excluded instead of moving them. The standalone
  behaviour (`--trash-dir`) is unchanged.
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
* [ ] **Import:** `--no-captions` imports images only.
* [ ] **Existing standalone databases** (embeddings, review, tags) keyed by path are refused with a hint; an explicit
  migration command adds the IDs in one transaction (cheap: the ID is a hash of the path, nothing is re-encoded).
* [ ] Later: reuse embeddings across projects and collections by content hash and model.

## Other

* [ ] Store a mean-centered copy or normalized embeddings to make thresholds model-independent? (`calibrate` measures
  the scale per model for now.)
* [ ] Train a custom model for anime image comparison, and later anime semantic search? Notes in
  [training-clip.md](training-clip.md), scripts under `src/clip_training`.
