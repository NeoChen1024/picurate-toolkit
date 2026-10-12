# TODO

* [ ] Dataset curation on one [project](project.md), each stage still usable on its own: import (done) →
  deduplication and culling (removing images that do not belong in the dataset) → optional batch tagging → the
  dataset editor (fixing tags, adding natural-language passages for Anima-style captions). How dedupe, culling and
  tagging store their results in the project is still open.
* [ ] Store a mean-centered copy or normalized embeddings to make thresholds model-independent? (`calibrate` measures
  the scale per model for now.)
* [ ] Train a custom model for anime image comparison, and later anime semantic search? Notes in
  [training-clip.md](training-clip.md), scripts under `src/clip_training`.
