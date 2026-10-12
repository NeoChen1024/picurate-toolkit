# Tagging (`picurate-tag`)

Writes Danbooru-tag captions (`image.txt` next to `image.png`) for training, with
[pixai-labs/pixai-tagger-v1.0](https://huggingface.co/pixai-labs/pixai-tagger-v1.0) (1.9 GiB download, 30877 tags in
general/character/copyright/artist/meta/rating). It shares the decoder pipeline with the deduper.
`picurate-tag <subcommand> --help` lists every option.

## Usage

`run` tags a plain directory and writes the captions next to its images in one go:

```shell
picurate-tag run dataset --style anima --prefix "masterpiece, best quality"   # tag + write captions in one go
picurate-tag run dataset --style sdxl -r --existing skip                       # sd-scripts WD14-style captions
```

Each caption file is written under a temporary name and renamed into place, so an interrupted run never leaves a
half-written one; `--existing skip` carries on after an interruption.

## Tagging a project

For a [project](project.md), the scores go into the project's `tags.sqlite` once, and the captions are rendered from
them whenever the project is read: the editor, its export and `export` below all see the same caption, and changing
the options needs no GPU.

```shell
picurate-tag update library                                          # GPU; Ctrl-C any time, rerun to resume
picurate-tag captions library --style anima --combine merge --prefix "masterpiece"   # CPU, shows a few captions
picurate-tag export library -n                                       # .txt next to the source images
```

`update` tags the images that take part: present, and not excluded (a duplicate a decided group does not keep, an
image a person excluded, anything in the trash), so cull before tagging unless the scores should inform the culling.
It commits every batch as it finishes and skips images whose mtime matches their row, so an interrupted run
(Ctrl-C, crash, power loss) loses at most the batch on the GPU. Rows of excluded and missing images are kept; rows of
images no longer in the project are dropped.

`captions` records the caption options (every option of `run` that shapes the caption) and how the result is
combined with an image's imported caption:

| `--combine` | with an imported caption | without one |
|---|---|---|
| `skip` (default) | the imported caption | the tags |
| `overwrite` | the tags | the tags |
| `append` | the imported caption, then the tags it lacks, compared exactly (sd-scripts' `--append_tags`) | the tags |
| `merge` | the same, compared like the editor's duplicate highlighting (case, underscores, escapes ignored) | the tags |

The options given replace the recorded ones (left out: the defaults); without any, `captions` only shows them.
Thresholds below the database's score floor are refused. A caption a person edited in the editor always wins; an
image without scores keeps its imported caption. When an image changed after it was tagged (as `picurate-project
sync` records), its old scores are still used, and the editor and `export` warn until `update` re-tags it. Captions
match `run`'s up to the order of near-tied tags (fp16 results depend slightly on which images share a batch).

`export` writes each included image's caption as a same-stem `.txt` next to it, for training on the source folders
directly; `-o captions.json(l)` collects them in one file instead. An existing caption file is replaced only when it
holds the caption imported from it or the one the last export wrote; one edited or created outside the project is
skipped and listed unless `--force`. `-n` lists what would be written.

## Caption styles

* `--style anima` follows [Anima](https://huggingface.co/circlestone-labs/Anima)'s order,
  `[prefix/meta/safety] [1girl/1boy/1other] [character] [series] [@artist] [general]`: spaces instead of underscores,
  `@` before artists, safety as `safe/sensitive/nsfw/explicit`. Quality and year tags are not something a tagger
  can know; put them in `--prefix`. The prefix goes first on purpose, before meta: that is where the training target
  usually belongs.
* `--style sdxl` reproduces sd-scripts' `tag_images_by_wd14_tagger.py --remove_underscore --character_tags_first
  --character_tag_expand --use_rating_tags_as_last_tag --always_first_tags "1girl,1boy"`: characters as
  `name, series`, general tags in Danbooru popularity order, `general/sensitive/questionable/explicit` last, and like
  sd-scripts the last always-first tag ends up first (`1boy, 1girl, ...`). Series and artists (no `@`) follow the
  characters, and meta tags precede the rating, as WD14 has no such categories.

## Choosing tags

* Thresholds default to the checkpoint's per-category F1-optimal values (general 0.17, character 0.27, copyright
  0.24, artist 0.15, meta 0.17). `--threshold` sets all of them but the artist one, which has its own
  `--artist-threshold`. `--no-artist` turns artists off; `--override-artist-tag "name"` writes that artist
  (comma-separated for several) for every image instead.
* Meta tags are filtered to the ones describing the image (`--meta all` keeps Danbooru bookkeeping such as
  `md5 mismatch` and drawing-software tags), and `lowres/highres/absurdres` come from the pixel size by Danbooru's
  rules (`--resolution`).
* `run` skips hidden (dot) files and directories unless `--include-hidden` is given.
* `--existing overwrite|skip|append` (`run`), `--undesired-tags`, `--tag-replacement`, `-o captions.json(l)`,
  `--frequency` and `--debug` behave like their sd-scripts counterparts.

## Score database (`tags.sqlite`)

Tables `meta`, `tags(id, name, category)`, `images(id, item, source_id, path, mtime, size, width, height, format)`
and `scores(image_id, tag_id, prob)`; `item` is the [item ID](project.md#sources-and-identity), and the small integer
`id` keeps the score rows short. Every tag scoring at least `--score-floor` (default 0.05) is kept, plus all
ratings: about 60 rows per image. `sidecars(item, digest)` holds the BLAKE3 of the caption file `export` last wrote
for each image, and `meta` the options `captions` recorded. The database is bound to the model revision, dtype and
floor, like the embedding database is to its model; a mismatch, or a database of another schema version, is refused
unless `update -f` rebuilds it (keeping the caption options).

## Model

The network is reimplemented in `src/picurate/tagger/model.py` without `trust_remote_code`, pinned to a verified
checkpoint revision; fp32 outputs match the checkpoint's own code to 2e-6. Rotary position embeddings are kept in
fp32, so fp16 (the default on CUDA) stays within ~1e-3 of fp32. `--compile` compiles one static shape, padding the
last batch: about 50% faster, roughly 14 img/s on an RTX 4080-class GPU, and about 1 GiB of VRAM at batch 1.
