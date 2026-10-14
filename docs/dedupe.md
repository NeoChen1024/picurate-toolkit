# Deduplication (`picurate-dedupe`)

Finds duplicates among the images of a [project](project.md) by CLIP image-embedding distance and confirms them pixel
by pixel. A library is cleaned as a project over the library folders. Embeddings live in the project's
`embeddings.sqlite` and are updated incrementally, so a re-run only encodes what changed; duplicate groups go to its
`review.sqlite`. Nothing is moved: the images a decided group does not keep count as excluded, the editor does not
export them, and `picurate-project gc --trash-excluded` moves them out of the library when you are sure. Everything is held in memory during matching: about 2.3 KiB per image in FP16 for the default
1152-dimensional model, so a million images need about 2.3 GiB of RAM and VRAM.

`picurate-dedupe <subcommand> --help` lists every option. Groups the pixel check does not settle are decided by a
human in [picurate-curate](curate.md).

## Usage

Find the duplicate groups of a project (`picurate-project sync` first picks up new and changed files):

```shell
picurate-dedupe dedupe library -k highest-quality                  # only what the pixel check decides
picurate-dedupe dedupe library --review-threshold 1.5              # also store looser groups for review
picurate-dedupe review-status library
```

Move the images of an incoming directory that the project already has to a trash directory, before adding the rest
to the library:

```shell
picurate-dedupe dedupe-import -i incoming library --trash-dir trash --import-db incoming.sqlite
picurate-dedupe update-db library --reuse-from incoming.sqlite     # after moving incoming into the library and syncing
```

Only (re)encode a project into its database, or print the distance matrix of a few images to pick a threshold:

```shell
picurate-dedupe update-db library
picurate-dedupe encode-test a.jpg a_copy.png b.jpg
```

Calibrate a threshold from the collection itself: sample images, synthesize lossy copies of them, and print
histograms of how far the copies land from the stored embedding next to how far the nearest *different* image is:

```shell
picurate-dedupe calibrate library -s 200 --seed 0 --variants jpeg90,half,quarter
```

## What takes part

Every image of the project that is present (not marked missing by the last sync, not in the project's
[trash](project.md#trash)) and not excluded by a person (force exclude in the editor). The losers of decided groups do take part, so their groups stay the same from run to run.
Rows of missing and excluded images stay in `embeddings.sqlite`, so they are not encoded again if they come back;
rows of images no longer in the project are dropped. `dedupe` writes `review.sqlite`, so it refuses to start while
picurate-curate has the project open; encoding does not hold that lock, but matching needs it.

**Starting status.** `dedupe` stores every group found at `--review-threshold` (default: `--threshold`):

* Groups whose every pair passes the pixel check start *decided*, with the keeping policy's winner kept, whatever
  their embedding distance. Their other members are excluded.
* Every other group starts *pending*, even one within `--threshold`: that is how artist revisions and pages of one
  pixiv set reach a human.
* With `--no-pixel-check`, groups entirely within `--threshold` start decided instead.
* Groups holding several pages of one work (the policy's [page variants](#keeping-policies)) start pending either way,
  with a copy of each page kept.

A looser `--review-threshold` answers "worth a human look", not "safe to merge"; it may well be looser than anything
`calibrate` would suggest.

**Re-running.** Groups are identified by their members. A re-run (after adding images, or with another threshold)
keeps the decisions of groups whose membership did not change and marks the others stale; a stale group excludes
nothing. Groups nobody has touched get their starting status re-derived, and pixel results are reused for files that
did not change. A new group that shares members with a group the reviewer changed starts pending with that group's
keeps, and is never re-derived: a human decision is only replaced by another one.

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
* `--pixel-check/--no-pixel-check` (default on): compare the pixels of every pair within the threshold. A group is
  only decided automatically when every pair differs by compression alone, so an edited or revised copy goes to a
  human, and `dedupe-import` never counts one as already present.
* `--pixel-buffer` (default 2048 MiB): decoded pixels the pixel check keeps across all its processes, so an image
  shared by several pairs is decoded once.
* `-k, --keeping-logic` (default `largest`) and `--keeping-config`: see [keeping policies](#keeping-policies).
* `dedupe-import`: `--trash-dir` (omit it to only report) and `-n, --dry-run`. Images of the project that are
  excluded do not count as already present. `--import-db` keeps the import side's embeddings for the next run or for
  `update-db --reuse-from`; without it they go to a temporary file. Duplicates within the import directory are not
  looked for.
* `-m, --model-id`: the image tower, by default the SigLIP2 so400m NaFlex checkpoint
  (`naflexvit_so400m_patch16_siglip.v2_webli`); or the path of a `clip-train export` file, whose name then
  identifies the model. A database is bound to the model it was encoded with; switching models on an existing
  database requires `-f, --force-update`, which re-encodes everything.
* `--seq-len` (default `1024`): the patch budget per image (at most this many 16x16 patches, aspect ratio kept).
  1024 costs the same as a 512px fixed model; lower it on small GPUs. Part of the database binding.
* `-b, --batch-size`: images per forward pass. The default model is compute bound already at 4.
* `-j, --workers`: decoder processes (default: one per CPU). On network or spinning storage reads are the
  bottleneck, so more workers than CPUs helps there.
* `--compile`: `torch.compile` the model with dynamic shapes, 20-30 s warm-up. No measurable gain today: the
  position embedding is resampled per patch grid in the batch, which splits the graph.

FP16 inference is not batch-invariant: the same image encoded in a different batch comes out up to ~0.04 away. That
is far below the default threshold, but it is why near-identical files are not exactly zero apart.

Which files a project has is decided by its [import](project.md#importing); `dedupe-import` skips hidden (dot) files
and directories of the import directory unless `--include-hidden` is given. Files whose extension Pillow does not
recognize are ignored. Files that fail to decode are skipped with a warning and
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
| `pic-dir` | files in a `Wallpaper` folder, then by source site in the filename (Pixiv > yande.re > Danbooru > Konachan), then as `highest-quality`; one copy of every page of a Pixiv work |

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

**Page variants.** A policy can also name the pages of one work by filename, as `pic-dir` does for Pixiv
(`12345678_p0.jpg`, `12345678_p1.jpg`): `variants = '<regex>'`, searched in the basename, with the named groups
`work` and `page`. A pixiv set that changes one expression is usually close enough to group, but every page is meant
to stay. When a group holds at least two different pages of one work, each page keeps its own best copy. The other
members (another work, a copy from another site, a file in a wallpaper folder) go with the page they are closest to
along the group's pair distances, and the criteria rank each page's copies separately. Only the work ID and page
number are compared: different works are never treated as pages of each other. Such a group always starts pending,
since attaching the other copies to a page is a guess.

## How it works

**Database.** Table `embeddings` has one row per image: the [item ID](project.md#sources-and-identity) with the source
ID and path (relative to the source folder) it is derived from, `mtime`, `size`, `width`, `height`, `format` as
observed at encoding time, `blake3` of the file contents (hashed by the decode worker from the bytes it reads anyway),
and the raw little-endian float16 embedding. Table `meta` records `model_id`, `dim`, `dtype`
and `schema_version`; a database of another model or schema version is refused unless `--force-update` rebuilds it. FP16 storage is lossless: the model runs in FP16 on CUDA, so
every value it emits is already an FP16 number. On update, stored mtimes are compared with the files on disk; changed
or new images are (re)encoded. Writes are committed per batch, so an interrupted run keeps its progress. `update-db --reuse-from OTHER.sqlite` first hashes the images that need encoding and copies
the embedding of every one whose contents another database of the same model already holds; this is how images
deduplicated with `dedupe-import --import-db` and then moved into the library avoid a second encode.

The review database is described under [picurate-curate](curate.md#review-database).

**Matching.** Embeddings are loaded as one `(N, D)` array and uploaded once. The self-dedupe searches the upper
triangle of the distance matrix in blocks of rows (each distance block under 256 MiB), collects all pairs under the
threshold and merges them into connected components, since A~B and B~C does not imply A~C. 100k images take about a
second on an RTX 4080-class GPU. On CUDA with Triton (bundled with PyTorch on Linux) a small kernel reads the FP16
embeddings and computes `sum((a-b)^2)` in FP32 directly. Elsewhere they are upcast to FP32 and `torch.cdist` is used,
whose `|a|^2+|b|^2-2ab` form cancels near zero distance: on SigLIP2 embeddings (norm about 11) a distance of 0.1 is
off by up to about 5e-4, and the error grows with the square of the norm.

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

**Model.** Only the image tower is needed: SigLIP2 so400m NaFlex, our own implementation
(`picurate.common.siglip_image`, the `vision_model` of `google/siglip2-so400m-patch16-naflex`; it computes exactly
what timm's `naflexvit_so400m_patch16_siglip.v2_webli` does, whose name still identifies the model). NaFlex takes
the image at its own aspect ratio, resized to at most `--seq-len` patches, so nothing is squashed or cropped; with
flash-attn installed the valid patches of a batch are packed and attended without padding, otherwise padding is
masked. That is what keeps lossy copies close: distance to the original / to the nearest different image is 0.5 /
5.6 for a JPEG q90 re-save, 0.4 / 5.6 for a 50% downscale and 2.4 / 5.6 for a 25% one, against 0.7 / 8.7, 1.4 / 8.7
and 4.3 / 8.7 for the fixed 512px SigLIP tower. The position embedding is resampled to the patch grid bicubically as timm does, where the
official SigLIP2 code (and transformers) does it bilinearly: the stored embeddings are about 0.9 apart from the
official ones (cosine ~0.98). Kept on purpose: lossy copies land slightly closer this way, and text queries against
the stored embeddings still rank images the same. PE-Core towers are noticeably more sensitive to compression artifacts: a JPEG q90
re-save lands 5 to 15 away from the original, overlapping with different pictures.
