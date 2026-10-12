# Projects (`picurate-project`)

A project is a directory that references images where they already are and owns everything derived from them:
captions, review status, edits and the exported dataset. Source images and their caption files are only ever read.
For curating a training dataset the stages are meant to run in order on one project: import, deduplication and
culling, optional batch tagging, then the [dataset editor](dataset-editor.md). Only import and the editor work on
projects so far; see [TODO](TODO.md).

```shell
picurate-project init my-dataset
picurate-project import my-dataset -r ~/pictures/set-a ~/pictures/extra.png
picurate-project sync my-dataset             # later: pick up new, changed and vanished images
picurate-dataset-editor my-dataset
```

## Importing

* A directory contributes the images directly in it; with `-r` (the editor: *Include subfolders*) also those in its
  subdirectories. Hidden (dot) files and directories are skipped unless `--include-hidden`; single files can be given
  too. PNG, JPEG, WebP and BMP are imported; other files are ignored.
* `--include` / `--exclude` (the editor: rules typed one per line as `+ PATTERN` / `- PATTERN`) filter what is
  below a directory like rsync's: the first rule that matches decides, unmatched paths are imported, and an excluded
  directory is not descended into. A pattern without `/` matches a file or directory name; one with `/` or `**`
  matches the path relative to the imported directory, from its root when it starts with `/`. `*` stays within a
  name, `**` crosses directories, a trailing `/` matches directories only and `/***` a directory with everything in
  it. Only PNG files, in every subdirectory: `-r --include '*/' --include '*.png' --exclude '*'`. Single files given
  by name are not filtered.
* A same-stem UTF-8 `.txt` next to an image becomes its imported caption, unless `--no-captions`. The caption file
  is never changed, and later changes to it are not picked up.
* An image already in the project, through any source folder that contains it, is skipped and keeps its edits, so
  importing a directory again only adds what is new; nothing appears without an import or a sync.
* Images are only statted (size, mtime), never read, so importing a large library over the network is quick. Progress
  is committed every 1000 images; an interrupted import is resumed by running it again.

**Sync** brings a project up to date with its source folders, as they were imported: new images of an imported
directory are added (recursive, hidden and filtered as the last import of that directory was; a folder from which only single files were imported
gets nothing new), changed ones get their new size and mtime recorded, and vanished ones are marked missing until they
come back. A source folder that is not there at all is skipped and listed rather than marked missing. Captions and
decisions are kept.
* Each image gets an output name from its path relative to the imported directory; names that collide (two sources
  with an `a/1.png`) get a suffix from the image's ID.

## Sources and identity

Every imported directory, or the directory of an imported file, is a source folder, recorded once an image is
imported from it. It gets a random 256-bit ID; an image is identified by its source ID and its path relative to the
source folder, and its item ID is derived from both (BLAKE3 in key-derivation mode, context `picurate item_id v1`,
over the source ID, a NUL byte and the path). The embedding, review and tag databases key everything by the item ID;
a standalone image directory has the all-zero source ID, so its databases use the same IDs.

**Relink Sources** in the editor points a source folder somewhere else after checking that every image is there with
its recorded file size and, for images in the project's `embeddings.sqlite`, the same content hash (images marked
missing may stay missing, but at least one must be found); the IDs do not change.

The project keeps no content hash of its own; once the images are encoded for deduplication, the BLAKE3 in
`embeddings.sqlite` is used. Export refuses a source whose contents differ while its mtime is still the one recorded
at encoding (damage or a swapped file) and only warns when the mtime changed too. A crop and fills record the image size they were made on: a source image of
another size keeps its caption and review status editable, but its image edits cannot be changed or exported, and its
thumbnail is marked `[!]`. Another file size or mtime alone only warns. LaMa repairs additionally refuse when the
pixels they were computed from changed ([dataset editor](dataset-editor.md#image-edits)).

## Decisions and history

What a person decides about an image is stored per field: the caption, the crop, the fills, the review status and
an exclusion (none, force keep or force exclude, with a reason). A field without a stored value has its default; for
the caption that is the imported one, so an unedited caption keeps following it.

The other stages write only their own files in the project directory, and reading the project combines them with
these decisions. An image is **excluded** when a person forced it out, or, with no human exclusion decision, when a
decided duplicate group in `review.sqlite` does not keep it ([deduplication](dedupe.md#what-takes-part)), unless a
person excluded every copy the group keeps; force keep overrides the group. Excluded images are not exported, and the editor marks them with what excluded them.

Every change belongs to an operation (in the editor: one visit to an image). The history keeps one entry per field
and operation with the value before and after it; a field that ends an operation where it started gets no entry.
Only one editor can have a project open at a time, and the commands below refuse to run while it is open.

```shell
picurate-project history my-dataset            # recent operations: ID, time, kind, changes, image
picurate-project rollback my-dataset 42        # undo operation 42
picurate-project gc my-dataset --keep-revisions 3 --older-than 30d -n
```

**Rollback** gives every field the operation changed its earlier value back, as a new operation (so a rollback can be
rolled back). A field changed again since is left alone and listed.

**GC** prunes history entries beyond the newest `--keep-revisions` of their field and/or of operations older than
`--older-than` (with both, an entry must match both), then the operations left without entries, then the LaMa masks
and patches that neither the current decisions nor the remaining history refer to. Current decisions are never
touched; pruned operations can no longer be rolled back. Nothing is pruned automatically; `-n` lists first.

## Project directory

```
project.json        schema version, source folders (ID → path and how it was imported), export and completion settings
project.sqlite      items     one row per image: ID, source ID and relative path, size, mtime, the imported caption
                              and where it came from, output name, missing (as of the last sync)
                    state     the current decisions, one row per image and changed field (JSON)
                    history   every change: operation, image, field, previous and new value (NULL: the default)
                    ops       when each operation happened and what it was
project.lock        held by the editor that has the project open
masks/, patches/    the editor's accepted LaMa repairs
output/             the last export and its .manifest.json
```

The project does not contain the images: backing it up alone is not a backup of the dataset.
