# picurate-toolkit

My own toolkit for image collections so large that `ls` in the directory takes more than 5 seconds, born from
dissatisfaction with off-the-shelf solutions (most of them are either slow or not suitable for my own image
directories). It started as a CLIP-based deduplicator (`clip-image-deduper`) and grew from there:

* `picurate-dedupe`: find duplicates by CLIP similarity, with `picurate-dedupe-review` as the GUI for the cases a
  single threshold should not decide.
* `picurate-tag`: Danbooru-tag captions for training datasets.
* A dataset editor (captions, crops, watermark removal), being ported over from my scripts repo.

Batch processing on the command line first, a GUI only where a human has to look. The three parts are separate
commands for now; how they should feed into each other is something the workflow has to show first.

## Installation

Requirements: Python 3.11+, a working PyTorch install (CUDA strongly recommended), and RAM/VRAM for your collection.
The models are downloaded from the Hugging Face Hub on first use (1.6 GiB for deduplication, 1.9 GiB for tagging).

```shell
git clone https://github.com/NeoChen1024/picurate-toolkit
cd picurate-toolkit
uv venv && source .venv/bin/activate
uv pip install -e .
```

A plain `python -m venv .venv` + `pip install -e .` works too. `pip install -e '.[training]'` adds the dependencies of
the CLIP fine-tuning scripts under `src/clip_training` (see `docs/training-clip.md`), which are not needed for
anything else.

## Deduplication

Everything is held in memory: embeddings are stored and kept on the GPU in FP16, about 2.3 KiB per image for the
default 1152-dimensional model, so a million images need about 2.3 GiB of RAM and VRAM.

### Key features

* Incremental embedding database: one SQLite file per image directory, updated by mtime, orphans cleaned up, bound to
  the model it was encoded with.
* Fast matching: 100k images against each other in about a second on an RTX 4080-class GPU. Queries are processed in
  blocks, so the work is a few hundred large kernels instead of one memory-bound pass per image.
* Exact distances: on CUDA a small Triton kernel reads the FP16 embeddings and computes `sum((a-b)^2)` in FP32
  directly, avoiding the `|a|^2+|b|^2-2ab` cancellation that costs `torch.cdist` about 1e-2 of precision on
  unnormalized CLIP embeddings.
* Pixel check: a matched pair only counts as a duplicate when its pixels differ by lossy compression alone, not by a
  signature, an edit or an artist revision that the embedding cannot see. On by default; it only ever blocks matches.
* Configurable keeping policies (which copy of a duplicate group survives), declared in TOML; four built in.
* Three-stage pipeline: decoder processes read and fully preprocess images, the main thread gathers batches, and
  the GPU runs one batch while the next is being decoded. Decoder count and batch size are independent knobs.

### Usage

One command with subcommands; `picurate-dedupe <subcommand> --help` lists every option.

Find duplicates within a directory and move the losers to a trash directory:

```shell
picurate-dedupe dedupe -i pictures -d pictures.sqlite --trash-dir trash -k highest-quality
```

Remove images from an "importing" directory that already exist in a "base" collection:

```shell
picurate-dedupe dedupe-import --base-image-dir pictures --base-db pictures.sqlite \
                              --import-image-dir incoming --import-db incoming.sqlite --trash-dir trash
```

Only (re)encode a directory into its database, or print the distance matrix of a few images to pick a threshold:

```shell
picurate-dedupe update-db -i pictures -d pictures.sqlite
picurate-dedupe encode-test a.jpg a_copy.png b.jpg
```

Review the band between the automatic threshold and a looser one in a GUI instead of trusting a single number:

```shell
uv pip install -e '.[gui]'
picurate-dedupe dedupe -i pictures -d pictures.sqlite --review-threshold 1.5   # stores groups, moves nothing
picurate-dedupe-review -d pictures.sqlite --trash-dir trash                    # decide, then Apply
picurate-dedupe review-status -d pictures.sqlite                               # or apply from the CLI:
picurate-dedupe review-apply -i pictures -d pictures.sqlite --trash-dir trash
```

Calibrate a threshold from the collection itself: sample images, synthesize lossy copies of them, and print
histograms of how far the copies land from the stored embedding next to how far the nearest *different* image is:

```shell
picurate-dedupe calibrate -i pictures -d pictures.sqlite -s 200 --seed 0 --variants jpeg90,half,quarter
```

#### Options worth knowing

* `-t, --threshold` (default `0.1`): Euclidean distance at or below which two images are duplicates. With the default
  model, bit-identical re-encodes are well under 0.1; a JPEG q90 re-save or a 50% downscale of the same picture lands
  around 0.2-0.5, a 25% downscale up to about 2.4, and different pictures at 5.5 and above. Raise the threshold if
  you want lossy copies caught too; `calibrate` measures these distributions on your own collection and model and
  suggests a value, `encode-test` does the same for a handful of files by hand.
* `calibrate` options: `-s, --samples` (default 200) images are drawn with `--seed` (default 0), so an unchanged
  directory gives the same sample again (the draw is over sorted paths, independent of database row order; it is not
  promised to survive version upgrades). `--variants` picks which lossy copies to synthesize (`jpeg90`, `jpeg75`,
  `webp80`, `half`, `quarter`), `-t` may be repeated to see what several thresholds would catch. The suggestion is
  the largest variant distance rounded up, provided it is below the nearest different image; the default 0.1 is
  intentionally far below it and is not what this command recommends.
* `--pixel-check/--no-pixel-check` (default on): for every pair within the threshold, compare the pixels and drop the
  pair unless the two differ by compression alone. `dedupe` then never merges an edited or revised copy, and
  `dedupe-import` never counts one as already present. See "How it works" for what is compared.
* `--pixel-buffer` (default 2048 MiB): decoded pixels the pixel check keeps across all its processes, so an image
  shared by several pairs is decoded once. The two images of each pair in progress are always held.
* `-k, --keeping-logic` (default `largest`) and `--keeping-config`: see below.
* `-n, --dry-run`: report what would be moved without moving. The database is still refreshed; add `--skip-update`
  for a run that writes nothing at all.
* `-m, --model-id`: a timm model name; any CLIP/SigLIP image tower timm ships works, NaFlex (`naflexvit_*`) or
  fixed-resolution (e.g. `vit_so400m_patch16_siglip_512.v2_webli`, `vit_pe_core_bigG_14_448.fb`). A database is bound
  to the model it was encoded with; switching models on an existing database requires `-f, --force-update`, which
  re-encodes everything.
* `--seq-len` (default `1024`): for NaFlex towers, the patch budget per image. The image keeps its aspect ratio and is
  resized to at most this many 16x16 patches; 1024 is the pixel budget of a 512px fixed model and costs the same
  time and VRAM. Lower it on small GPUs. It is part of the identity a database is bound to.
* `-b, --batch-size`: images per forward pass; raise it if VRAM allows. The default model is compute bound already at
  4, so this barely changes throughput.
* `-j, --workers`: decoder processes (default: one per CPU); twice as many reads are kept in flight. On network or
  spinning storage the reads are the bottleneck, so a value above the CPU count helps there.
* `--compile`: `torch.compile` the model with dynamic shapes, so the batch size stays free. About 17% faster for
  fixed-resolution towers, 20-30 s warm-up per run. No gain for the default NaFlex tower yet: timm's NaFlex
  position-embedding code breaks the graph.
* FP16 inference is not batch-invariant: the same image encoded in a different batch (size or neighbours) comes out
  up to ~0.04 away, with or without `--compile`. That is far below the default threshold and the ~0.4+ of a lossy
  copy, but it is the reason distances between near-identical files are not exactly zero.
* Files whose extension Pillow does not recognize are ignored. Files that fail to decode are skipped with a warning and
  retried on the next run. Unlike Pillow's defaults, very large images, truncated files and PNGs with bad checksums on
  metadata chunks (e.g. the `iCCP` chunk some Pixiv uploads carry) are accepted, like an image viewer would.

#### Keeping policies

A policy is an ordered list of criteria over the metadata stored in the database (size, mtime, width, height, pixels,
format, and regexes on the path). Duplicates are sorted by the criteria and the first one is kept. Built in, from
`src/picurate/dedupe/policies.toml`:

| name | keeps |
|---|---|
| `newest` | latest mtime, then larger file |
| `largest` | largest file, then newer |
| `highest-quality` | most pixels, then PNG > TIFF > BMP > WebP > AVIF > JPEG > GIF, then larger file, then newer |
| `pic-dir` | files in a `Wallpaper` folder, then by source site in the filename (Pixiv > yande.re > Danbooru > Konachan), then as `highest-quality` |

`pic-dir` encodes the author's own collection layout; it is there as an example of a site-specific policy. Add your
own or override a built-in with `--keeping-config my.toml`:

```toml
[policy.originals-first]
description = "Prefer the raw/ folder, then the biggest file"
criteria = [
    { attr = "dirname", match = '^raw/' },
    { attr = "size", prefer = "max" },
]
```

`policies.toml` documents the three criterion kinds (`prefer`, `match`, `order`).

### Reviewing duplicates

`dedupe --review-threshold T` matches at `T` and writes every group into `pictures.review.sqlite` (next to the
embedding database) instead of moving anything. Groups whose every pair passes the pixel check start *decided* with
the keeping policy's winner ticked, whatever their embedding distance; every other group starts *pending*, even one
within the normal `--threshold` (that is how artist revisions and pages of one pixiv set reach a human). With
`--no-pixel-check` the old rule applies: groups entirely within `--threshold` start decided. Groups are keyed by their
member paths, so re-running the match (after adding images, or with another threshold) keeps the decisions of groups
whose membership did not change and marks the others stale. Groups nobody has changed yet get their starting status
re-derived on every run, and pixel results are reused for files that did not change.

`picurate-dedupe-review` opens that file: group list on the left (surest first, status badge per row), two linked
image views in the middle (side by side, flip while holding Space, or an amplified difference), members with
thumbnails and keep checkboxes on the right. Z/X move between groups, 1-9 pick which member is shown as A
(Shift+1-9 for B), K toggles keep on A, Enter marks the group decided and jumps to the next pending one, U undoes,
F1 lists everything. Groups that passed the pixel check are marked ≈, and the Show box can list only those or only
the ones with pixel differences. A group can only be decided while it keeps a member; R discards a whole group on
purpose (every member is moved), and Show → "keeps nothing" finds groups with every keep unticked. The keeping policy is chosen in the toolbar (custom ones via `--keeping-config`); P resets the
keeps of every pending group to its choice, Shift+P only the current group's. Decisions are committed to the review database as they are made. **Apply** (Ctrl+Enter) lists
every move, then moves the losers of decided groups to the trash directory; groups whose files changed on disk since
matching are refused, and an applied group can be undone while the trash directory is untouched. Nothing is ever
deleted. `review-status` and `review-apply` do the same from the CLI.

### How it works

**Database.** Table `embeddings` has one row per image: `path` (relative to the image directory), `mtime`, `size`,
`width`, `height`, `format` as observed at encoding time, and the raw little-endian float16 embedding. Table `meta`
records `model_id`, `dim`, `dtype` and `schema_version`. FP16 storage is lossless: the model runs in FP16 on CUDA, so
every value it emits is already an FP16 number. On update, stored mtimes are compared with the files on disk; changed
or new images are (re)encoded and rows without a file are deleted. Writes are committed per batch, so an interrupted
run keeps its progress.

**Matching.** Embeddings are loaded as one `(N, D)` array and uploaded once. The self-dedupe searches the upper
triangle of the distance matrix in blocks of rows (block size chosen to keep a distance block under 256 MiB), collects
all pairs under the threshold and merges them into connected components, since A~B and B~C does not imply A~C.
On CUDA with Triton (bundled with PyTorch on Linux) the embeddings stay FP16 on the device and `l2_triton.py`
computes exact L2 distances with FP32 arithmetic. Elsewhere they are upcast to FP32 and `torch.cdist` is used; its
matmul form loses about 1e-2 near zero distance, which is why the kernel exists. Peak VRAM for 100k images is about
0.7 GiB on the Triton path and 1.1 GiB on the fallback.

**Pixel check.** Both images are reduced to the smaller one's size (at most 1024 px on the long side, box filter),
blurred with a Gaussian of sigma 1.5 px to remove JPEG block edges and ringing, and compared per pixel (largest
difference over the three channels). The *peak* is the largest mean over 16x16 windows, the *mean* the average over
the whole picture. A pair differs by compression only when the peak is at most 0.5, or when the peak is at most 12,
the mean at most 3 (a global colour change fails here) and the peak at most 20 times the mean (a small mark on an
otherwise identical copy fails here: compression noise is spread out, an edit is not). Pairs whose aspect ratios
differ by more than 1% (crops) and unreadable files never pass; transparency is composited on grey first. Measured on
the author's library with synthetic variants: JPEG q60-q90, WebP 80, double compression and a halved re-encode pass
94.5-100%, 2%-high text and a 6-level colour shift 0%, 1%-high text and 1.5-3% patches 1-2%. Pairs sharing images go
to the same process, which keeps decoded images within its share of `--pixel-buffer`; about 0.4 s per pair per core
on a network share.

**Encoding pipeline.** `--workers` processes each open a file, decode it, and run the model's full preprocessing
(resize, normalize, patchify, cast to the model dtype), so the main process only ever sees ready-to-stack arrays. It keeps
`2 * workers` decode jobs in flight, gathers results into batches as they complete (a slow file never blocks the
others), stages each batch in pinned memory and submits it to the GPU without waiting; the previous batch's result is
collected and written to the database only once the next one is queued. Memory is bounded by `2 * workers` arrays
plus two batches. Measured on 1000 cached 4 MB images (RTX 4080, 16 CPUs): the GPU alone does 65 img/s at any batch
size, 16 decoders alone 74 img/s, and the pipeline reaches 42 img/s against 24 img/s for the previous version at the
same batch size of 4.

**Model.** Only the image tower is needed, so it is loaded straight from timm (`naflexvit_so400m_patch16_siglip.v2_webli`,
the SigLIP2 so400m NaFlex tower, 1.6 GiB; the full checkpoint with the text tower is 4.3 GiB). NaFlex takes the image
at its own aspect ratio: the decoder resizes it to at most `--seq-len` 16x16 patches, normalizes and patchifies it,
and the encoder pads the patch sequences of a batch to `--seq-len` with a validity mask. Nothing is squashed or
cropped, which is what makes it tighter than the fixed-size towers on the variants a deduper cares about. Measured on
24 library images against `vit_so400m_patch16_siglip_512` (the previous default), as distance to the original /
smallest distance between different images: JPEG q90 0.5 / 5.6 vs 0.7 / 8.7, 50% downscale 0.4 / 5.6 vs 1.4 / 8.7,
25% downscale 2.4 / 5.6 vs 4.3 / 8.7. Same speed and VRAM at `--seq-len 1024`. Fixed-resolution towers still work:
for them the encoder forces `crop_pct=1.0, crop_mode="squash"` with bicubic resampling, which is what they were
trained with (timm's default eval transform center-crops 90% and shifts embeddings by several units).
`PE-Core-bigG-14-448` was the default before SigLIP2 and is noticeably more sensitive to compression artifacts: a
JPEG q90 re-save of a picture lands 5 to 15 away from the original, overlapping with different pictures.

## Tagging images (`picurate-tag`)

A separate command writes Danbooru-tag captions (`image.txt` next to `image.png`) for training, with
[pixai-labs/pixai-tagger-v1.0](https://huggingface.co/pixai-labs/pixai-tagger-v1.0) (1.9 GiB download, 30877 tags in
general/character/copyright/artist/meta/rating). It shares the decoder pipeline with the deduper.

```shell
picurate-tag run dataset --style anima --prefix "masterpiece, best quality"   # tag + write captions in one go
picurate-tag run dataset --style sdxl -r --existing skip                       # sd-scripts WD14-style captions
```

For anything large, tag into a score database once and render captions from it as often as needed:

```shell
picurate-tag update dataset -r -d dataset.tags.sqlite          # GPU; Ctrl-C any time, rerun to resume
picurate-tag export dataset -d dataset.tags.sqlite --style anima --override-artist-tag "nnn yryr"   # CPU, seconds
```

`update` commits every batch as it finishes and skips images whose mtime matches their row, so an interrupted run
(Ctrl-C, crash, power loss) loses at most the batch on the GPU and the next run carries on; it also drops rows of
deleted images. The database (`meta`, `tags`, `images`, `scores(image_id, tag_id, prob)`) keeps every tag scoring at
least `--score-floor` (0.05) plus all ratings, ~60 rows per image, and is bound to the model revision, dtype and floor
like the embedding database is to its model. `export` takes every caption option of `run`, refuses thresholds below
the floor and skips images changed since they were tagged. Captions match `run`'s, up to the order of near-tied tags
(fp16 results depend slightly on which images share a batch).

* `--style anima` follows [Anima](https://huggingface.co/circlestone-labs/Anima)'s order,
  `[prefix/meta/safety] [1girl/1boy/1other] [character] [series] [@artist] [general]`: spaces instead of underscores,
  `@` before artists, safety as `safe/sensitive/nsfw/explicit`. Quality and year tags are not something a tagger
  can know; put them in `--prefix`.
* `--style sdxl` reproduces sd-scripts' `tag_images_by_wd14_tagger.py --remove_underscore --character_tags_first
  --character_tag_expand --use_rating_tags_as_last_tag --always_first_tags "1girl,1boy"`: characters as
  `name, series`, general tags in Danbooru popularity order, `general/sensitive/questionable/explicit` last, and like sd-scripts the
  last always-first tag ends up first (`1boy, 1girl, ...`). Series,
  artists (no `@`) follow the characters and meta tags precede the rating, as WD14 has no such categories.
* Thresholds default to the checkpoint's per-category F1-optimal values (general 0.17, character 0.27, copyright
  0.24, meta 0.17) except artists, at `--artist-threshold 0.5`: the model names an artist for any image, including
  ones it never saw. `--no-artist` turns them off.
* Meta tags are filtered to the ones describing the image (`--meta all` keeps Danbooru bookkeeping such as
  `md5 mismatch`), and `lowres/highres/absurdres` come from the pixel size by Danbooru's rules (`--resolution`).
* `--override-artist-tag "name"` writes that artist (comma-separated for several) for every image instead of any
  prediction.
* `--existing overwrite|skip|append`, `--undesired-tags`, `--tag-replacement`, `-o captions.json(l)`, `--frequency`
  and `--debug` behave like their sd-scripts counterparts.

The network is reimplemented in `src/picurate/tagger/model.py` (no `trust_remote_code`; fp32 outputs match the
checkpoint's own code to 2e-6). fp16 is the default on CUDA, within ~1e-3 of fp32. `--compile` compiles one static
shape: 9.7 → 14.4 img/s on an RTX 4080 SUPER, about 1 GiB of VRAM at batch 1.

## Roadmap

* [ ] Port the dataset editor (PyQt6, project-based captions/crops/LaMa watermark repair) from my scripts repo to
  PySide6 as `picurate-dataset-editor`, then keep improving it there.
* [ ] Let the parts feed each other once the workflow settles: captions from the tag database in the editor,
  duplicate hints on import.

* [ ] Store a mean-centered copy or normalized embeddings to make thresholds model-independent? (`calibrate` now
  measures the scale per model instead.)
* [x] PySide6 review GUI for the band between the automatic and a looser threshold (`picurate-dedupe-review`;
  design notes in [docs/gui-review.md](docs/gui-review.md)).
* [ ] Train a custom model for anime image comparison, and later anime semantic search? Notes in
  [docs/training-clip.md](docs/training-clip.md), scripts under `src/clip_training`.
