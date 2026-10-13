# TODO

## Other

* [ ] **Fine-tuned towers in the deduper**: load a `clip-train export` file, identified by the name in its metadata
  (databases are bound to that name).

* [ ] Store a mean-centered copy or normalized embeddings to make thresholds model-independent? (`calibrate` measures
  the scale per model for now.)
* [ ] Train a custom model for anime image comparison, and later anime semantic search? Notes in
  [training-clip.md](training-clip.md), scripts under `src/clip_training`.
