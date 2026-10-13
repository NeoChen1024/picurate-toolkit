# Dataset editor (`picurate-dataset-editor`)

The last stage of a [project](project.md): going through a training dataset by hand. Captions with Danbooru/e621
tag completion, one crop per image, watermark fills (mean colour or LaMa repair) and a review status per image.
Original images stay where they are and are never written to; **Export** renders the dataset into the project's
`output/`. It needs the `gui` extra.

```shell
picurate-dataset-editor my-project   # a project directory, not the image directory
```

1. Create and fill the project with `picurate-project`, or from the editor: **New Project** (Ctrl+N, an empty
   directory) and **Import Images** (a folder or individual files), which follow the same
   [import rules](project.md#importing).
2. Edit and review. Changes are saved 500 ms after they are made, before moving to another image and on close.
3. **Export** writes PNG or lossless WebP plus a same-stem `.txt` per image, for every image except skipped ones,
   approved ones only, or the current one.

An image's caption starts as its imported caption, combined with the tagger's when the project has been
[tagged](tagger.md#tagging-a-project); editing it makes it the person's. The status bar says when a caption was
rendered from the tag scores of an earlier version of the image, and Export counts such captions.

F1 lists the keys. The ones used most: Z / X previous / next image (or the wheel), A / S crop / fill mode, D
preview, C cycle Unreviewed → Approved → Skipped, E cycle the exclusion (automatic → force keep → force exclude),
F focus the caption. In the caption box Tab accepts a completion (or opens the list),
Enter always inserts a newline, Ctrl+R searches, Esc returns to the canvas. Single-letter keys never fire in a text
or number field.

## Captions

Completion searches DraconicDragon's merged [Danbooru/e621 tag list](https://github.com/DraconicDragon/dbr-e621-lists-archive)
(about 322k tags with aliases) from any of the last five words, so free-form modifiers in front of a tag are kept.
Accepted tags get spaces instead of underscores by default; parenthesis escaping is off by default; both are project
settings. Only accepted candidates are formatted: unknown tags and natural-language text stay as typed. Putting the
caret in an entry highlights its duplicates elsewhere in the caption (compared ignoring case, underscores and
escapes).

## Image edits

Crop and fill rectangles are in the pixels of the EXIF-oriented original. Mean fill paints a rectangle with the
average of a ring around it (4 px by default), not counting rectangles already painted with a mean colour (a LaMa
repair's pixels count as picture). Image undo/redo is separate from
caption undo. Images are rendered as 8-bit RGB: alpha, higher bit depths and ICC profiles are not kept, and animated
files use their first frame.

**LaMa repair** (Fill method → LaMa, then draw a rectangle) opens a mask editor on that region. A mask is generated
there (Hi-SAM by default; SAM-TS, or SAM 2.1 with a tight box or points), adjusted with threshold, expand/contract and
a brush, then [Big-LaMa](https://github.com/advimman/lama) fills it with up to 128 px of surrounding context. Only
masked pixels in the region are changed. On a handful of signatures, text segmentation on a user-drawn region found
the strokes reliably, SAM 2 needed a tight box, and expanding the mask by 3-6 px was what removed the antialiased
edges; flat backgrounds are often fine with mean fill, while LaMa on texture synthesizes something plausible rather
than restoring it. Inspect the preview before accepting.

An accepted repair is stored under the project (`masks/`, `patches/`) with the hash of the image it was computed
from; export reuses it and never runs a model. If an earlier edit of the same image changes, the repair is stale and
refuses to render until it is removed and redone, or the earlier change is undone.

The models need the `repair` extra plus Hi-SAM and SAM 2 from source, and download weights from the Hugging Face Hub
on first use, at pinned revisions (the Hi-SAM and Big-LaMa weights come from third-party mirrors, `GoGiants1/Hi-SAM`
and `gaspardbruno/BigLama`, whose file is checked against a fixed SHA-256):

```shell
uv pip install -e '.[gui,repair]'
python -m picurate.dataset_editor.install_repair
```

Mask and LaMa devices, the default backend, expansion, context and offline mode are project settings (**Settings**).
An explicit `cuda:N` that is not available is an error; only Auto falls back to the CPU.

## Export

Export replaces the managed contents of `output/` as a whole; it is not incremental. It is built in a staging
directory from a snapshot taken when it starts and swapped in at the end, so a cancelled or failed export leaves the
previous output intact (an interrupted swap is recovered when the project is next opened). Files in `output/` the
editor did not write are kept, and an export that would overwrite one is refused. A source image whose size no longer
matches its crop and fills fails the export ([sources](project.md#sources-and-identity)); excluded images are left
out.

Moved or missing source images are covered under [sources](project.md#sources-and-identity).

## Exclusion

Thumbnails of excluded images are marked ✕, with what excluded them in the tooltip: a duplicate group (and the copy
it keeps), a person, or the [trash](project.md#trash) (images there can still be viewed, but their exclusion only
changes with a rollback). The Exclusion box decides it by hand: *automatic* follows the duplicate groups
([curation](curate.md)), *force keep* and *force exclude* override them, with an optional reason. It is a decision
like the others, saved with the image's history.
