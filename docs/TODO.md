# TODO

## Other

* [ ] **CLIP training without transformers**: the timm image tower (its NaFlex data loader) with the text tower of
  `picurate.common.siglip_text`, and SigLIP's sigmoid loss with the checkpoint's `logit_scale`/`logit_bias`, written
  here. The position embedding stays resampled bicubically like timm does (the official code uses bilinear).
  `src/clip_training/train.py` still uses transformers' `CLIPModel`.

* [ ] Store a mean-centered copy or normalized embeddings to make thresholds model-independent? (`calibrate` measures
  the scale per model for now.)
* [ ] Train a custom model for anime image comparison, and later anime semantic search? Notes in
  [training-clip.md](training-clip.md), scripts under `src/clip_training`.
