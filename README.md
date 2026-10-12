# picurate-toolkit

My own toolkit for image collections so large that `ls` in the directory takes more than 5 seconds, born from
dissatisfaction with off-the-shelf solutions (most of them are either slow or not suitable for my own image
directories). It started as a CLIP-based deduplicator (`clip-image-deduper`) and grew from there.

Batch processing on the command line first, a GUI only where a human has to look. The parts are separate commands for
now; how they should feed into each other is something the workflow has to show first.

| Command                                           | What it does                                                                                           |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| [picurate-dedupe](docs/dedupe.md)                 | Find duplicates by CLIP similarity, confirm them pixel by pixel, move the losers to a trash directory. |
| [picurate-dedupe-review](docs/dedupe-review.md)   | GUI for the duplicate groups a single threshold should not decide.                                     |
| [picurate-tag](docs/tagger.md)                    | Danbooru-tag captions for training datasets, in Anima or sd-scripts (SDXL) layout.                     |
| [picurate-project](docs/project.md)               | Create a dataset project and import existing image/caption pairs into it.                              |
| [picurate-dataset-editor](docs/dataset-editor.md) | GUI for going through a dataset by hand: captions with tag completion, crops, watermark removal.       |

## Installation

Requirements: Python 3.11+, a working PyTorch install (CUDA strongly recommended), and RAM/VRAM for your collection.
The models are downloaded from the Hugging Face Hub on first use (1.6 GiB for deduplication, 1.9 GiB for tagging).

```shell
git clone https://github.com/NeoChen1024/picurate-toolkit
cd picurate-toolkit
uv venv && source .venv/bin/activate
uv pip install -e '.[gui]'
```

A plain `python -m venv .venv` + `pip install -e .` works too; leave out `[gui]` if you only want the command-line
tools. `[training]` adds the dependencies of the CLIP fine-tuning scripts under `src/clip_training`
([notes](docs/training-clip.md)), which nothing else needs.

## Deduplication

```shell
picurate-dedupe dedupe -i pictures -d pictures.sqlite --trash-dir trash -k highest-quality
```

The first run encodes every image into `pictures.sqlite`; later runs only encode what changed, so re-running on a
growing collection is cheap. Matching is fast: 100k images against each other take about a second on an RTX
4080-class GPU, and a million images need about 2.3 GiB of RAM and VRAM.

Things I'd want to know before trusting it with my own pictures:

* Nothing is ever deleted. Losers are moved to `--trash-dir`, keeping their relative paths, and `-n` shows what
  would happen.
* The default threshold (0.1) is deliberately conservative: it catches re-encodes, not lossy copies or downscales.
  `picurate-dedupe calibrate` shows where those land on your collection if you want to go looser.
* Every match is confirmed by a pixel check: a pair only counts when its pixels differ by compression alone, so a
  signed, edited or revised copy is never merged, however close its embedding.
* Which copy survives is a keeping policy (`largest` by default); you can write your own in TOML.

For the cases in between, match with a looser `--review-threshold` and decide in the GUI instead:

```shell
picurate-dedupe dedupe -i pictures -d pictures.sqlite --review-threshold 1.5   # stores groups, moves nothing
picurate-dedupe-review -d pictures.sqlite --trash-dir trash                    # decide, then Apply
```

Details: [deduplication](docs/dedupe.md), [duplicate review](docs/dedupe-review.md).

## Tagging

```shell
picurate-tag run dataset --style anima --prefix "masterpiece, best quality"   # tag + write captions in one go
picurate-tag update dataset -r -d dataset.tags.sqlite                         # or: scores into a database...
picurate-tag export dataset -d dataset.tags.sqlite --style sdxl              # ...and captions from it, no GPU
```

Uses [pixai-tagger-v1.0](https://huggingface.co/pixai-labs/pixai-tagger-v1.0), reimplemented so it needs no
`trust_remote_code` and works in fp16. `update` can be interrupted at any time and resumes where it stopped, which
is the thing I missed most in kohya's tagging script. Details: [tagging](docs/tagger.md).

## Datasets

```shell
picurate-project init my-dataset
picurate-project import my-dataset ~/pictures/set-a   # same-stem .txt files come along as captions
picurate-dataset-editor my-dataset
```

How I want dataset curation to go: import into a project, dedupe and cull what does not belong, batch-tag if there
are no captions yet, then go through everything by hand in the editor. Every stage should still work on its own, and
only import and the editor know about projects so far. The images are only referenced, never touched; captions and
edits live in the project and are rendered into a fresh dataset on export.

The editor is for the part no tagger gets right: fixing captions with Danbooru/e621 completion, adding the
natural-language bits, cropping, and painting out watermarks and signatures (mean colour, or a Hi-SAM mask plus LaMa
when the background is not flat). Details: [projects](docs/project.md), [dataset editor](docs/dataset-editor.md).

## License

GNU GPL version 3 or later; see [LICENSE](LICENSE).

## Credits

* [timm](https://github.com/huggingface/pytorch-image-models) and Google's SigLIP2 for the image embeddings.
* [PixAI](https://huggingface.co/pixai-labs/pixai-tagger-v1.0) for the tagger.
* [Anima](https://huggingface.co/circlestone-labs/Anima)'s prompt format and kohya's
  [sd-scripts](https://github.com/kohya-ss/sd-scripts) WD14 tagger, whose caption layouts `picurate-tag` follows.
* DraconicDragon's [Danbooru/e621 tag lists](https://github.com/DraconicDragon/dbr-e621-lists-archive) for the
  dataset editor's tag completion.
* [Hi-SAM](https://github.com/ymy-k/Hi-SAM), [SAM 2](https://github.com/facebookresearch/sam2) and
  [LaMa](https://github.com/advimman/lama) for watermark repair.
