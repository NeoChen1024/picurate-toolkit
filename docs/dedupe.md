# Deduplication (`picurate-dedupe`)

Finds duplicates by CLIP image-embedding distance, confirms them pixel by pixel, and moves the losers to a trash
directory. Embeddings live in one SQLite file per image directory and are updated incrementally, so a re-run only
encodes what changed. Everything is held in memory during matching: about 2.3 KiB per image in FP16 for the default
1152-dimensional model, so a million images need about 2.3 GiB of RAM and VRAM.

`picurate-dedupe <subcommand> --help` lists every option. Borderline groups can be handed to a human instead, see
[duplicate review](dedupe-review.md).

## Usage

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

Calibrate a threshold from the collection itself: sample images, synthesize lossy copies of them, and print
histograms of how far the copies land from the stored embedding next to how far the nearest *different* image is:

```shell
picurate-dedupe calibrate -i pictures -d pictures.sqlite -s 200 --seed 0 --variants jpeg90,half,quarter
```

## Options worth knowing

* `-t, --threshold` (default `0.1`): Euclidean distance at or below which two images are duplicates. With the default
  model, bit-identical re-encodes are well under 0.1; a JPEG q90 re-save or a 50% downscale of the same picture lands
  around 0.2-0.5, a 25% downscale up to about 2.4, and different pictures at 5.5 and above. The default is
  deliberately far below the lossy copies: raise it if you want those caught too, or send them to the review GUI.
  `calibrate` measures these distributions on your own collection and model, `encode-test` does the same for a
  handful of files by hand.
* `calibrate`: `-s, --samples` (default 200) images are drawn with `--seed` (default 0), so an unchanged directory
  gives the same sample again (not promised across versions). `--variants` picks the lossy copies to synthesize
  (`jpeg90`, `jpeg75`, `webp80`, `half`, `quarter`); `-t` may be repeated to see what several thresholds would catch.
  The suggestion is the largest variant distance rounded up, provided it is below the nearest different image.
* `--pixel-check/--no-pixel-check` (default on): for every pair within the threshold, compare the pixels and drop the
  pair unless the two differ by compression alone. `dedupe` then never merges an edited or revised copy, and
  `dedupe-import` never counts one as already present. It only ever blocks matches.
* `--pixel-buffer` (default 2048 MiB): decoded pixels the pixel check keeps across all its processes, so an image
  shared by several pairs is decoded once.
* `-k, --keeping-logic` (default `largest`) and `--keeping-config`: see [keeping policies](#keeping-policies).
* `-n, --dry-run`: report what would be moved without moving. The database is still refreshed; add `--skip-update`
  for a run that writes nothing at all.
* `-m, --model-id`: a timm model name; any CLIP/SigLIP image tower timm ships works, NaFlex (`naflexvit_*`) or
  fixed-resolution (e.g. `vit_so400m_patch16_siglip_512.v2_webli`). A database is bound to the model it was encoded
  with; switching models on an existing database requires `-f, --force-update`, which re-encodes everything.
* `--seq-len` (default `1024`): for NaFlex towers, the patch budget per image (at most this many 16x16 patches, aspect
  ratio kept). 1024 costs the same as a 512px fixed model; lower it on small GPUs. Part of the database binding.
* `-b, --batch-size`: images per forward pass. The default model is compute bound already at 4.
* `-j, --workers`: decoder processes (default: one per CPU). On network or spinning storage reads are the
  bottleneck, so more workers than CPUs helps there.
* `--compile`: `torch.compile` the model with dynamic shapes. About 17% faster for fixed-resolution towers, 20-30 s
  warm-up; no gain for the default NaFlex tower yet (timm's NaFlex position embedding breaks the graph).

FP16 inference is not batch-invariant: the same image encoded in a different batch comes out up to ~0.04 away. That
is far below the default threshold, but it is why near-identical files are not exactly zero apart.

Hidden (dot) files and directories are skipped unless `--include-hidden` is given; build output and caches live
there. Files whose extension Pillow does not recognize are ignored. Files that fail to decode are skipped with a warning and
retried on the next run. Very large images, truncated files and PNGs with bad checksums on metadata chunks (e.g. the
`iCCP` chunk some Pixiv uploads carry) are accepted, like an image viewer would.

## Keeping policies

A policy is an ordered list of criteria over the metadata stored in the database (size, mtime, width, height, pixels,
format, and regexes on the path). Duplicates are sorted by the criteria and the first one is kept. Built in, from
`src/picurate/dedupe/policies.toml`:

| name | keeps |
|---|---|
| `newest` | latest mtime, then larger file |
| `largest` | largest file, then newer |
| `highest-quality` | most pixels, then PNG > TIFF > BMP > WebP > AVIF > JPEG > GIF, then larger file, then newer |
| `pic-dir` | files in a `Wallpaper` folder, then by source site in the filename (Pixiv > yande.re > Danbooru > Konachan), then as `highest-quality` |

`pic-dir` encodes my own collection layout; it is there as an example of a site-specific policy. Add your own or
override a built-in with `--keeping-config my.toml`:

```toml
[policy.originals-first]
description = "Prefer the raw/ folder, then the biggest file"
criteria = [
    { attr = "dirname", match = '^raw/' },
    { attr = "size", prefer = "max" },
]
```

`policies.toml` documents the three criterion kinds (`prefer`, `match`, `order`).

## How it works

**Database.** Table `embeddings` has one row per image: `path` (relative to the image directory), `mtime`, `size`,
`width`, `height`, `format` as observed at encoding time, and the raw little-endian float16 embedding. Table `meta`
records `model_id`, `dim`, `dtype` and `schema_version`. FP16 storage is lossless: the model runs in FP16 on CUDA, so
every value it emits is already an FP16 number. On update, stored mtimes are compared with the files on disk; changed
or new images are (re)encoded and rows without a file are deleted. Writes are committed per batch, so an interrupted
run keeps its progress.

**Matching.** Embeddings are loaded as one `(N, D)` array and uploaded once. The self-dedupe searches the upper
triangle of the distance matrix in blocks of rows (each distance block under 256 MiB), collects all pairs under the
threshold and merges them into connected components, since A~B and B~C does not imply A~C. 100k images take about a
second on an RTX 4080-class GPU. On CUDA with Triton (bundled with PyTorch on Linux) a small kernel reads the FP16
embeddings and computes `sum((a-b)^2)` in FP32 directly. Elsewhere they are upcast to FP32 and `torch.cdist` is used,
whose `|a|^2+|b|^2-2ab` form loses about 1e-2 near zero distance on unnormalized CLIP embeddings.

**Pixel check.** Both images are reduced to the smaller one's size (at most 1024 px on the long side, box filter),
blurred with a Gaussian of sigma 1.5 px to remove JPEG block edges and ringing, and compared per pixel (largest
difference over the three channels). The *peak* is the largest mean over 16x16 windows, the *mean* the average over
the whole picture. A pair differs by compression only when the peak is at most 0.5, or when the peak is at most 12,
the mean at most 3 (a global colour change fails here) and the peak at most 20 times the mean (a small mark on an
otherwise identical copy fails here: compression noise is spread out, an edit is not). Pairs whose aspect ratios
differ by more than 1% (crops) and unreadable files never pass; transparency is composited on grey first. On
synthetic variants of real illustrations: JPEG q60-q90, WebP 80, double compression and a halved re-encode pass
94.5-100%; 2%-high text and a 6-level colour shift 0%; 1%-high text and 1.5-3% patches 1-2%. Pairs sharing images go
to the same process, which keeps decoded images within its share of `--pixel-buffer`.

**Encoding pipeline.** `--workers` processes each open a file, decode it and run the model's full preprocessing
(resize, normalize, patchify, cast to the model dtype), so the main process only stacks ready arrays. It keeps
`2 * workers` decode jobs in flight, gathers results into batches as they complete (a slow file never blocks the
others), and submits each batch to the GPU without waiting; a batch's result is written to the database once the
next one is queued. Memory is bounded by `2 * workers` arrays plus two batches.

**Model.** Only the image tower is needed, so it is loaded straight from timm:
`naflexvit_so400m_patch16_siglip.v2_webli`, the SigLIP2 so400m NaFlex tower (1.6 GiB). NaFlex takes the image at its
own aspect ratio, resized to at most `--seq-len` patches and padded with a validity mask, so nothing is squashed or
cropped. That is what keeps lossy copies close: distance to the original / to the nearest different image is 0.5 /
5.6 for a JPEG q90 re-save, 0.4 / 5.6 for a 50% downscale and 2.4 / 5.6 for a 25% one, against 0.7 / 8.7, 1.4 / 8.7
and 4.3 / 8.7 for the fixed 512px SigLIP tower. Fixed-resolution towers are fed the whole image squashed to their
input size with bicubic resampling, as they were trained (timm's default eval transform crops 90% and shifts
embeddings by several units). PE-Core towers are noticeably more sensitive to compression artifacts: a JPEG q90
re-save lands 5 to 15 away from the original, overlapping with different pictures.
