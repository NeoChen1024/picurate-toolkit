# Fine-tuning CLIP/SigLIP for anime images

Working notes for the scripts under `src/clip_training`: what was researched and decided, and the parts that exist so
far: a training-step benchmark (`clip-train-bench`), image packs (`clip-pack`), the training batches made from them
(`clip-data`) and the training runs (`clip-train`).

## Two routes, not one

Tag-based contrastive fine-tuning and deduplication pull in opposite directions. Training the image tower to match
Danbooru tags makes embeddings converge on "same tags = close". That shrinks the distance between a picture and its
re-encoded or downscaled copy (good), but it shrinks the distance between two different pictures of the same
character in the same pose just as much (bad: the 0.1 threshold works today precisely because the pretrained tower is
sensitive to the details tags do not describe). So:

1. **Dedupe route** (image tower only, no captions). Self-supervised image-image fine-tuning, SimCLR/DINO style, where
   the positives are the variants the deduper should merge; negatives are other images. Duplicates met in practice
   are mostly the same picture at another, not integer-related resolution, compressed again; crops and watermarks are
   rarer. The 75-token problem does not exist on this route. Use the real duplicate groups the current deduper finds
   (and the near misses between 0.1 and ~1) as a validation set: the goal is a bigger gap between "variant" and
   "different picture", not a smaller absolute distance.
2. **Search route** (both towers). Image-text alignment on tags and captions for anime semantic search. Its captions
   are described under "Training batches" below.

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
  * optimizer state offload to the host (`optim.offload`, below): the state is touched once per step;
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
* Framework: no `transformers`, no timm. Both towers are ours (`picurate.common.siglip_image` and `siglip_text`),
  built from the one `google/siglip2-so400m-patch16-naflex` checkpoint. The image tower keeps timm's parameter names
  and computes the same function as timm's `naflexvit_so400m_patch16_siglip.v2_webli` (checked bit-exact on real
  images in FP32 and bf16), so a trained tower is in the layout the deduper loads. The position embedding is
  resampled bicubically as timm does (the official code uses bilinear). Attention runs on flash-attn's
  variable-length kernels when the package is installed (`model.attention = "flash"`, the default then): the valid
  patches of a batch are packed into one sequence, so padding costs nothing and no mask is needed. Without it
  (`"sdpa"`), the padded batch is masked in PyTorch's `scaled_dot_product_attention`, which then cannot use its
  flash kernel.
* Preprocessing at training time must match inference: aspect ratio kept, bicubic, mean/std 0.5, the same patch
  budget (`--seq-len`, 1024 by default). A tower trained on 256-patch inputs and evaluated at 1024 is a different
  model.

## Step benchmark

`clip-train-bench` (`src/clip_training/bench_train.py`) runs full training steps (forward, sigmoid loss, backward,
fused AdamW) on synthetic data with the real architectures and random weights, finds the largest batch that fits and
times it; `--grad-cache MICRO` with `--batch-size` times a GradCache step instead. `--memory` caps PyTorch's
allocations to stand in for a smaller GPU; `--freeze-blocks`, `--train` and `--no-checkpointing` change what is
trained. Every image is a full 1024-patch sequence, the worst case. Trained parameters are FP32 master weights under
bf16 autocast; frozen ones are stored in bf16 (`--frozen-fp32` keeps them FP32), which changes their outputs by
less than 1e-4 in cosine.

Measured on an RTX 4080-class GPU, activation checkpointing on:

| Setup | Largest batch | Throughput |
|---|---|---|
| dedupe, full image tower, 24 GiB free | 83 pairs | 7.2 pairs/s (14 images/s) |
| dedupe, full image tower, 7 GiB | 1 pair | 6.3 pairs/s |
| dedupe, lower 14 of 27 blocks frozen, 7 GiB | 16 pairs | 8.0 pairs/s |
| the same with GradCache, micro-batches of 8 | 256 pairs | 6.2 pairs/s |
| search, both towers, 24 GiB free | 74 | 13.5 pairs/s |
| search, image tower only (14 blocks frozen), 7 GiB | 19 | 14.8 pairs/s |
| the same with GradCache, micro-batches of 2 | 256 | 11.3 pairs/s |

The GPU is saturated from a batch of about 4: a larger batch adds negatives to the loss, not speed. The full image
tower's weights, gradients and AdamW state alone take 6.4 GiB, so an 8 GiB GPU needs frozen blocks, LoRA or offloaded
optimizer state.

Gradient accumulation does not add negatives (each micro-batch's loss only compares its own samples); GradCache does:
it embeds the whole batch without gradients, takes the loss and its gradient with respect to the embeddings, then
embeds each micro-batch again and back-propagates its slice. Its gradients match the full batch's (relative
difference about 1e-6 in FP32) for one extra forward pass, 20-25% of the step. Its peak is higher than a single small
step's, because the second micro-batch's forward holds autocast's bf16 weight copies while the first one's gradients
exist (full image tower: 7.35 GiB against 6.51; in FP32, without those copies, 6.84 against 6.56), so the full tower
does not fit 7 GiB with it either.

## Image packs

Training reads images from a pack, not from the dataset directory: `clip-pack` decodes every image once, makes it
upright RGB on white, downscales it with Lanczos to at most 524,288 pixels (twice the 1024-patch budget, room for
augmentation) and stores it as lossless WebP, back to back in 16 GiB shards with an SQLite index (key, location,
size, BLAKE3, and each image's tags with their categories). One image is one `pread`, which suits a network share. A
truncated or undecodable source is reported and skipped; an interrupted import is run again and resumes. `verify`
checks every stored image against its BLAKE3 and decodes it; `info` prints the totals.

On Danbooru images a pack takes about 365 KiB and 5.9 bit/px per image (500 kpx on average), around 300 GiB for the
860k images of cagliostrolab-860k-ordered.

### cagliostrolab-860k-ordered

The dataset is image/caption pairs (`danbooru_<post>_<md5>.{jpg,png,webp}` and a `.txt`) imported in stages, the
top level and `continued/`; the stages are merged, a post being keyed by its file stem. Anime frames under
`continued/screencap/<series>/<character>/` are keyed by that path without `continued/`; the same frame filed under
two characters becomes two records with identical bytes. A caption reads

    1girl, <characters>, <copyrights>, <artists>, ||| <rating>, [nsfw,] <general tags>, <year>, <meta>, <quality>, <aesthetic>

and its tags are stored in that order, without the `|||`. Categories come from the post list JSON that comes with the
dataset, converted once into a tag database (`clip-pack cagliostro-tags JSON OUTPUT`; 861k posts and 204k tags,
each tag in one category without conflicts). The words the dataset adds for conditioning (the rating word, `nsfw`,
year ranges, quality and aesthetic words) get the category `training`; tags the database lacks are `name` before the
`|||` and `general` after it.

    clip-pack check-cagliostro DATASET [--no-decode] > problems.txt
    clip-pack cagliostro-tags DATASET-tags.json tags.sqlite
    clip-pack import-cagliostro PACK DATASET --categories tags.sqlite

The MD5 in a file name is the original post's, and about 60% of the PNGs (and a few hundred JPEG and WebP files) no
longer match it: the dataset's makers saved them again. A mismatch alone is therefore not damage; such an image is
decoded completely and accepted if that works, which for PNG's checksums is a real integrity check. Expect about 45
images that fail: JPEGs cut off at a whole number of MiB and a few originals that never decoded. `--settle` (300 s)
skips files modified in the last minutes, which tar is still writing while an archive is being extracted.

### deepghs konachan_full

A deepghs `*_full` booru mirror is one row per post in `table.parquet` (id, file name, MIME type, rating, a
space-separated tag string), the tag dictionary with Moebooru type numbers in `tags.parquet`, and the original files
in `images/<n>.tar`, each with a `.json` listing every member's offset, size and SHA-256. The importer reads a post's
bytes straight from the archive at that offset, checks the SHA-256 and decodes them; nothing is extracted. Posts are
keyed `konachan_<id>` (`--prefix`), so one pack can hold several boorus. Tags are stored names first, as the captions
expect: character, copyright, artist, circle, style, then general and tags of unknown type (no category), alphabetical
within a category as the dataset lists them. The rating is stored first as `safe`, `questionable` or `explicit` in
the category `training`, like the rating word of cagliostro. GIF posts (animations) and posts missing from the
archives are counted and skipped.

    clip-pack check-deepghs DATASET [--no-decode] > problems.txt
    clip-pack import-deepghs PACK DATASET [--prefix konachan] [--limit N]

`check-deepghs` reads every post's file, checks it against its SHA-256 and decodes it (the table's size must match),
and compares archive sizes with their listings, ids and tags with the tables, before anything is imported.

Written for konachan_full (319k posts, 875 GB of originals, 2 Mpx median); the sister datasets with the same table
columns should import the same way but have not been tried. Stored, a konachan image takes about 460 KiB.

## Training batches

`src/clip_training/data.py` turns a pack into batches for either route. The same seed and the same images give the
same batches bit for bit: from any pack built from the same dataset (images are numbered in key order, not in the
order parallel encoding stored them), at any number of loader workers, and when resuming mid-epoch. Each sample is
drawn from a generator seeded by the seed, the epoch and a hash of the image's key; outside the code, this rests on
the versions of Pillow, libwebp, libjpeg and numpy. Images go to the model as uint8 patches, preprocessed exactly as
the deduper does (aspect ratio kept, bicubic, at most `--seq-len` patches; `picurate.common.siglip_image.to_patches`) and
normalized on the GPU. The held-out split hashes the keys, the smallest key of a group of identical images deciding
for the group, so it depends neither on the seed nor on the stored bytes.

* **search**: the image and a caption made up from its tags at every step. It takes `L` of the image's tags, `L`
  log-uniform between one and all of them, up to 64 tokens. Tags are drawn with weight `(images with the tag + k) ^
  -alpha` (k = 100, alpha = 0.5), optionally mixed with a uniform share: common tags appear less often, and k keeps
  tags seen once or twice from crowding every caption they can be in. A tag on fewer than 10 images is never a
  caption's only tag. `training` tags are left out. The order is the source order (names first) or shuffled, half
  and half. On this dataset the median caption has 6 tags and 20 tokens; 19% have one tag and 15% fill 60 tokens or
  more.
* **dedupe**: two views per image, each altered with probability 0.8: scaled down to a factor between 0.3 and 1 of
  the model's input size with one of five filters, saved again as JPEG (quality 50-95, 4:2:0 or 4:4:4) or lossy WebP
  70% of the time, both in random order, and cropped by up to 4% per side 5% of the time.

The loss gets a matrix of positives instead of the identity: a caption also matches every other image in the batch
that has all its tags (a one-tag caption like `blue eyes` matches many), and identical images match each other. In
batches of 64 captions there are about 45 such extra positives.

`clip-data sample PACK [--route dedupe] [--out DIR]` prints a batch's captions and positives and can save what the
model sees as PNG; `clip-data bench PACK -j N` times the loader. Eight workers of a 16-thread desktop CPU deliver
about 500 search or 340 dedupe samples per second, against the 7-15 per second the GPU trains. Each worker keeps
`prefetch` batches ready in shared memory, and a dedupe batch of 256 pairs at 1024 patches is about 400 MB, so the
default is two.

## Training runs

    clip-train run RUN_DIR --config run.toml     # start
    clip-train run RUN_DIR                       # continue from the newest checkpoint
    clip-train export RUN_DIR OUTPUT --name NAME

The settings are a TOML file; [clip-train.example.toml](clip-train.example.toml) lists every key at its default.
Experiments train fully (with the lower image blocks frozen where memory is short); `[model.lora]` puts LoRA adapters
on the trained blocks instead, with the pooling head and final norms still trained fully. Defaults: schedule-free
AdamW at 1e-5 with 500 warm-up steps (`adamw` with cosine decay is there for comparison), clipping at norm 1, bf16
autocast with FP32 trained parameters, and frozen ones stored in bf16. `optim.lr` is the image tower's learning rate
and `optim.lr_text` the text tower's (0: the same), with one warm-up and decay curve for both: the text tower starts
from general English and has further to move to tag captions, while the image tower is better moved gently, so the
search route usually wants the text tower faster, which is smoother than freezing image blocks and combines with
it. `optim.micro_batch` turns on GradCache; it
must divide the batch size, and evaluation runs in batches of the same size, so every pass has the same shape.
`optim.offload` keeps the FP32 weights and the optimizer state in pinned host memory and runs the optimizer step on
the CPU, as ZeRO-Offload does: the GPU holds bf16 weights and FP32 gradient accumulators (the logit scale and bias
stay FP32), 6 bytes per trained parameter instead of 16. Both towers trained with a micro-batch of 8 peak at 6.2
GiB instead of 15.3, for about 2.5 s more per step (the schedule-free AdamW step over 840 M parameters on the
CPU; the copies are a quarter of that), a cost per step, not per sample: a batch of 256 pays about 12%, a batch of
64 a third. The loss follows the same trajectory (bf16 weights in the forward pass are what autocast computed with
anyway). Checkpoints hold the FP32 weights either way, so a run can switch `offload` when continued. It is meant
for 16 GiB GPUs, where the 16.9 GiB of weights, gradients and AdamW state of both towers do not fit at all, and for
leaving a larger card's memory to the desktop; an 8 GiB card is better served by frozen blocks or LoRA, since it has
neither the room for micro-batches beyond one or two nor the compute for an epoch in reasonable time.
`model.compile` compiles the towers' forward passes: on an RTX 4080-class GPU dedupe steps are 12-14% faster and
search steps 5%, for 0.7 GiB more memory and one to three minutes of compiling, once each for training, GradCache's
gradient-free pass and evaluation (the position-embedding resampling, done per distinct patch grid in the batch,
still splits the image tower into several graphs). The
logit scale and bias start from the checkpoint's (search) or SigLIP's initial log 10 and -10 (dedupe: image-image
similarities are distributed differently from image-text ones) and are learned at their own rate.

A run directory holds the settings it started with, the pack it is bound to (settings, image count and a hash of the
keys), the checkpoints and the TensorBoard logs (`tensorboard --logdir RUN_DIR`; TensorBoard 2.21 or later, since
earlier versions need the `pkg_resources` that current setuptools no longer ships). A checkpoint holds the trained
parameters only, the optimizer state and the position in the data, `(epoch, batch)`: since batches follow from the
seed and the images, nothing else is needed to continue. Checkpoints are written whole or not at all, every
`checkpoint.every_steps` steps or `every_minutes` minutes, on Ctrl-C (after the current step) and at the end; the
newest three are kept. A failure to load a batch (an unreachable pack, after a read has been retried for seven
seconds) also writes a checkpoint before it ends the run. A step with a non-finite loss or gradient is skipped; ten
in a row end the run without one. A continued run takes `--config` only for what does not change its results:
workers, intervals, the pack's path, the attention layout, `optim.offload`, and `run.steps` with the schedule-free optimizer, whose lack of a schedule makes
training longer just more steps. Different settings or a different pack are refused; a new run can start from
another one's weights with `model.init_from`, which must stay in place, since continuing and exporting rebuild the
frozen weights from it (as from the checkpoint at `model.revision`). One `run` at a
time trains in a run directory. Continuing is not bit-exact: GPU reductions in attention and matrix products are not deterministic,
so two uninterrupted runs already differ in the third or fourth digit of the loss after a step or two, and a
continued run differs from an uninterrupted one by as much.

Evaluation runs on the evaluation weights of schedule-free AdamW, over `eval.images` held-out images whose captions
and views are fixed across runs: made with the default caption and augmentation settings, whatever a run trains with,
so runs are compared on the same task. It logs the loss over the whole set, recall at 1 and 5 in both directions (a hit is
any positive: for search, any image with all of the caption's tags) and, for a dedupe tower's judgement, the cosine
of the positive pairs (mean and 5th percentile) against that of the hardest negative per image (mean and 95th
percentile). The real duplicate groups of a deduplicated library are the validation still to add.

`export` writes a checkpoint's evaluation weights in FP32 with LoRA merged, as safetensors: `image.*` in timm's
layout, for the search route `text.*` in `TextTower`'s, `logit_scale` and `logit_bias`, and a name in the metadata.
The name identifies the model the way a timm model ID does: a database is bound to it, so a test version needs a
name of its own. The deduper cannot load an export yet.

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

## Captions still to do

* **Mix in natural language**: for a fraction of steps use a VLM-generated sentence instead of the tag list.
  Re-captioning is a documented gain for retrieval and keeps the text tower from only understanding comma lists.
* Deduplicate the training set with this tool first; near-duplicate pairs with different tags remain false negatives.

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
2. Dedupe route: fine-tune on the pack, measure the variant/different gap on held-out duplicate groups.
3. Search route: train with the text tower frozen first.
