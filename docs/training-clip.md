# Fine-tuning CLIP/SigLIP for anime images

Working notes for the scripts under `src/clip_training`. Nothing here is implemented yet beyond the dataset
preprocessing stub; this records what was researched and decided (2026-10) so the implementation does not have to
rediscover it.

## Two routes, not one

Tag-based contrastive fine-tuning and deduplication pull in opposite directions. Training the image tower to match
Danbooru tags makes embeddings converge on "same tags = close". That shrinks the distance between a picture and its
re-encoded or downscaled copy (good), but it shrinks the distance between two different pictures of the same
character in the same pose just as much (bad: the 0.1 threshold works today precisely because the pretrained tower is
sensitive to the details tags do not describe). So:

1. **Dedupe route** (image tower only, no captions). Self-supervised image-image fine-tuning, SimCLR/DINO style, where
   the positives are the variants the deduper should merge: JPEG re-saves at several qualities, downscales, WebP/AVIF
   re-encodes, small crops and borders, watermarks and signatures, mild colour/levels edits. Negatives are other
   images. The 75-token problem does not exist on this route. Use the real duplicate groups the current deduper finds
   (and the near misses between 0.1 and ~1) as a validation set: the goal is a bigger gap between "variant" and
   "different picture", not a smaller absolute distance.
2. **Search route** (both towers). Image-text alignment on tags and captions for anime semantic search. Everything in
   the "Dataset" section below is about this route.

The two routes produce two independent models. Both start from the same SigLIP2 so400m checkpoint, but neither
fine-tune has to preserve the other's behaviour: the dedupe tower is free to drift from the text alignment, and the
search model is free to lose the detail sensitivity the deduper needs.

## Model and precision

* Base: SigLIP2 so400m, NaFlex variant (the deduper's default, see [dedupe.md](dedupe.md)). NaFlex matters here: anime illustrations
  are mostly portrait, and squashing them to a square is exactly the kind of distortion a fine-tune should not learn.
* **Precision floor: bf16 (or fp16 with loss scaling). Nothing below.** No 8-bit optimizer states, no 4-bit/8-bit
  weights, no fp8 activations. Memory is to be found by offloading and recomputation, not by quantising.
* Memory budget for the full so400m image tower (~430M parameters), bf16 weights with fp32 master weights and AdamW
  states: roughly 0.9 + 1.7 + 3.4 GB ≈ 6 GB before activations. On an 8 GiB GPU that means at least one of:
  * activation checkpointing (cheap, always on);
  * optimizer state offload to CPU (DeepSpeed ZeRO-Offload, or torch's own `fused` CPU AdamW on the offloaded
    master copy) — the LLM-style trick that fits here, since the state is touched once per step;
  * freezing the text tower (search route) or the lower half of the image tower;
  * LoRA on attention projections with the base frozen, which also solves most of the forgetting problem.
* Optimizer: [schedule_free](https://github.com/facebookresearch/schedule_free) AdamW. No schedule to tune and the
  same state size as AdamW (Prodigy keeps two extra buffers). Trap: it trains on one weight sequence and evaluates
  on another; every eval, checkpoint and the final export must happen after `optimizer.eval()`.
* Loss: SigLIP's sigmoid loss, which only looks at pairs and tolerates small batches far better than the softmax
  CLIP loss (which needs hundreds to thousands of negatives per step). If batches must still grow beyond what fits,
  use gradient caching (GradCache) rather than smaller precision.
* Forgetting (search route): low learning rate with warm-up, or LoRA, and compare against the original with WiSE-FT
  interpolation (`w = (1-a)·w_orig + a·w_ft`) before deciding a fine-tune is an improvement. Evaluate zero-shot
  retrieval on a held-out set at every checkpoint, not just the training loss. The dedupe route has no alignment to
  keep; its only metric is the variant/different gap on held-out groups.
* Framework: train with `transformers` (`Siglip2Model`, `attn_implementation="sdpa"` or flash-attention 2,
  gradient checkpointing built in), then convert the image tower to timm's `naflexvit_*` layout for the deduper.
  timm's checkpoints were themselves converted from the Google release, so the mapping exists; the conversion
  script belongs in `src/clip_training`.
* Preprocessing at training time must match inference: aspect ratio kept, bicubic, mean/std 0.5, the same patch
  budget (`--seq-len`, 1024 by default). A tower trained on 256-patch inputs and evaluated at 1024 is a different
  model.

## Text side: token limits

* The SigLIP/SigLIP2 text tower takes **64 tokens**, padded to max length as trained; the classic CLIP tower takes
  77 (75 content tokens). Do not hard-code either, read `model_max_length` from the tokenizer of the checkpoint.
* CLIP's BPE splits `long_hair` into several tokens; replace `_` with a space and drop the `\(` escapes before
  counting. SigLIP2's Gemma tokenizer (256k vocabulary) is much more token-efficient on tag words, which partly
  offsets its shorter limit.
* Reported in the literature: the effective text length of CLIP is under 20 tokens, with later positions barely
  contributing. Stuffing every tag in is therefore less useful than it looks.
* Length extension (Long-CLIP: keep the first 20 positions, interpolate the rest ×4 to 248; TULIP: relative
  positions) only applies to the CLIP architecture, needs its own fine-tune, and is overkill for tag lists, which are
  unordered sets. Prefer the sampling approach below.

## Dataset handling

The current `dataset_preprocessing.py` splits an over-long tag list into several `(image, chunk)` rows. Two problems:

1. **False negatives.** Two chunks of the same image in one batch are treated as negatives by the contrastive loss.
2. **Partial descriptions.** Each chunk describes part of the image; the model learns "some tags ↔ whole image",
   which is no better than a random subset.

Planned replacement: preprocessing only cleans and categorises; the caption is assembled per training step.

* **Categorise tags** using Danbooru's categories: artist, copyright, character, general, meta, rating.
  * Drop meta tags (`highres`, `absurdres`, `commentary_request`, `bad_id`, `translated`, ...): they describe the
    file, not the picture.
  * Keep artist tags only when the artist has enough samples to learn a style from; otherwise they are noise.
* **Priority fill per step**: character and copyright always, rating, then general tags sampled at random until the
  token budget is full. Different subset and order every epoch: this is tag dropout plus shuffle, and it doubles as
  augmentation.
* **Frequency filtering**: drop general tags below N occurrences in the dataset (unlearnable); down-weight the
  near-universal ones (`1girl`, `solo`, `looking at viewer`) so captions do not all start the same way.
* **Mix in natural language**: for a fraction of steps use a VLM-generated sentence instead of the tag list.
  Re-captioning is a documented gain for retrieval and keeps the text tower from only understanding comma lists.
* **Count tokens on the assembled string**, once, with the real tokenizer. The current script tokenises each tag
  separately and counts BOS/EOS every time, which over-counts by two per tag.
* Deduplicate the training set with this tool first; near-duplicate pairs across batches are false negatives too.

## Existing resources

* [OysterQAQ/DanbooruCLIP](https://huggingface.co/OysterQAQ/DanbooruCLIP): a `transformers` CLIP fine-tuned on
  Danbooru and Pixiv. Useful as a baseline to see what tag fine-tuning does to the dedupe distance distribution
  before investing in training.
* [Anime-2026](https://dl.acm.org/doi/10.1145/3805622.3810619): 1.5M character images, 4k text queries, baselines;
  the retrieval benchmark for the search route.
* Papers: [SigLIP](https://arxiv.org/abs/2303.15343) (sigmoid loss), [Long-CLIP](https://arxiv.org/abs/2403.15378),
  [TULIP](https://arxiv.org/abs/2410.10034), [FineLIP](https://arxiv.org/abs/2504.01916),
  [A Picture is Worth More Than 77 Text Tokens](https://openaccess.thecvf.com/content/CVPR2024/papers/Urbanek_A_Picture_is_Worth_More_Than_77_Text_Tokens_Evaluating_CVPR_2024_paper.pdf).

## Suggested order of work

1. Run the deduper with DanbooruCLIP and with the default SigLIP2 on the same library and compare the distance
   distributions of known duplicate groups vs same-character pairs. This decides how much tag fine-tuning hurts.
2. Dedupe route: build the variant-augmentation dataset from the library, fine-tune with LoRA first, measure the
   variant/different gap on held-out groups.
3. Search route: rewrite dataset preprocessing as described, then train with the text tower frozen first.
