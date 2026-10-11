# Tagging (`picurate-tag`)

Writes Danbooru-tag captions (`image.txt` next to `image.png`) for training, with
[pixai-labs/pixai-tagger-v1.0](https://huggingface.co/pixai-labs/pixai-tagger-v1.0) (1.9 GiB download, 30877 tags in
general/character/copyright/artist/meta/rating). It shares the decoder pipeline with the deduper.
`picurate-tag <subcommand> --help` lists every option.

## Usage

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
(Ctrl-C, crash, power loss) loses at most the batch on the GPU and the next run carries on. It also drops rows of
deleted images. `export` takes every caption option of `run`, refuses thresholds below the database's score floor and
skips images changed since they were tagged. Its captions match `run`'s up to the order of near-tied tags (fp16
results depend slightly on which images share a batch).

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
* Hidden (dot) files and directories are skipped unless `--include-hidden` is given.
* `--existing overwrite|skip|append`, `--undesired-tags`, `--tag-replacement`, `-o captions.json(l)`, `--frequency`
  and `--debug` behave like their sd-scripts counterparts.

## Score database

Tables `meta`, `tags(id, name, category)`, `images(id, path, mtime, size, width, height, format)` and
`scores(image_id, tag_id, prob)`. Every tag scoring at least `--score-floor` (default 0.05) is kept, plus all
ratings: about 60 rows per image. The database is bound to the model revision, dtype and floor, like the embedding
database is to its model; a mismatch is refused unless `-f` rebuilds it.

## Model

The network is reimplemented in `src/picurate/tagger/model.py` without `trust_remote_code`, pinned to a verified
checkpoint revision; fp32 outputs match the checkpoint's own code to 2e-6. Rotary position embeddings are kept in
fp32, so fp16 (the default on CUDA) stays within ~1e-3 of fp32. `--compile` compiles one static shape, padding the
last batch: about 50% faster, roughly 14 img/s on an RTX 4080-class GPU, and about 1 GiB of VRAM at batch 1.
