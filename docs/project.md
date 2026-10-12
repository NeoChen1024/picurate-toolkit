# Projects (`picurate-project`)

A project is a directory that references images where they already are and owns everything derived from them:
captions, review status, edits and the exported dataset. Source images and their caption files are only ever read.
For curating a training dataset the stages are meant to run in order on one project: import, deduplication and
culling, optional batch tagging, then the [dataset editor](dataset-editor.md). Only import and the editor work on
projects so far; see [TODO](TODO.md).

```shell
picurate-project init my-dataset
picurate-project import my-dataset ~/pictures/set-a ~/pictures/extra.png
picurate-dataset-editor my-dataset
```

## Importing

* A directory is imported recursively, skipping hidden (dot) files and directories unless `--include-hidden`; single
  files can be given too. PNG, JPEG, WebP and BMP are imported; other files are ignored.
* A same-stem UTF-8 `.txt` next to an image becomes its starting caption. The caption file is never changed, and
  later changes to it are not picked up.
* An image the project already references (same resolved path) is skipped and keeps its edits, so importing a
  directory again only adds what is new; nothing appears without an explicit import.
* Every new image is hashed (SHA-256), so an import reads each file once. Progress is written every 1000 images; an
  interrupted import is resumed by running it again.
* Each image gets an output name from its path relative to the imported directory; names that collide (two sources
  with an `a/1.png`) get a suffix from the image's ID.

## Sources

Every imported directory, or the directory of an imported file, is a source folder. **Relink Sources** in the editor
points one somewhere else after checking the SHA-256 of every image in it. An image that is missing or changed keeps
its caption and review status editable, but cannot be edited or exported; its thumbnail is marked `[!]`.

## Project directory

```
project.json        schema version, source folders, export and completion settings
items.jsonl         one line per image: stable ID, source folder and relative path, size/mtime/SHA-256,
                    the caption it was imported with, output name
edits.jsonl         append-only: the complete state of an image (caption, crop, fills, review status) per save
masks/, patches/    the editor's accepted LaMa repairs
output/             the last export and its .manifest.json
```

The last saved state of an image wins. A torn last line of `edits.jsonl` (from a crash mid-write) is moved to an
`edits-*.recovery` file when the project is opened. The project does not contain the images: backing it up alone is
not a backup of the dataset.
