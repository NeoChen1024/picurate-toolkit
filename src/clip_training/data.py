"""Training batches from a pack (see :mod:`clip_training.pack`) for the two routes of docs/training-clip.md.

The same seed and the same images give the same batches, bit for bit: in every run, from every pack built from the
same dataset (images are numbered in key order, not in the order parallel encoding stored them), at any number of
loader workers, and when resuming mid-epoch. Each sample is drawn from a generator seeded by ``(seed, epoch, a hash
of the image's key)``, an epoch's order by ``(seed, epoch)``. What it rests on outside this module: libwebp decoding
the same pixels, Pillow resizing and libjpeg/libwebp compressing alike (their versions), and numpy's generators.
Images reach the model as SigLIP2 NaFlex input, exactly as the deduper preprocesses them (aspect ratio kept, bicubic resize to at most ``seq_len`` 16x16 patches), but as uint8
patches: :func:`to_device` normalizes them on the GPU, a quarter of the bytes of float32 over the bus.

* ``search``: one image and a caption made up for it on the spot from its stored tags (:class:`CaptionMaker`).
* ``dedupe``: two views of one image, each altered the way duplicates found in practice differ (:class:`Augment`).

Both carry a ``(B, B)`` boolean ``labels`` matrix of positives for the sigmoid loss instead of the identity: a caption
also matches every other image of the batch that has all of its tags, and images with identical stored bytes (one
anime frame filed under two characters) match each other. :func:`holdout` splits on the stored bytes for the same
reason.
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
import PIL.Image
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from picurate.common.siglip_image import DEFAULT_SEQ_LEN, PATCH, image_size_for_seq, to_patches

from .pack import Pack

ROUTES = ("search", "dedupe")
_FILTERS = (
    PIL.Image.Resampling.LANCZOS,
    PIL.Image.Resampling.BICUBIC,
    PIL.Image.Resampling.BILINEAR,
    PIL.Image.Resampling.HAMMING,
    PIL.Image.Resampling.BOX,
)


# -- images ---------------------------------------------------------------------------------------------------------


def from_patches(patches: np.ndarray, coords: np.ndarray) -> PIL.Image.Image:
    """The image of one sample's valid uint8 patches (for looking at what the model gets)."""
    rows, columns = coords.max(0) + 1
    grid = np.zeros((rows, columns, PATCH, PATCH, 3), dtype=np.uint8)
    grid[coords[:, 0], coords[:, 1]] = patches.reshape(-1, PATCH, PATCH, 3)
    return PIL.Image.fromarray(grid.transpose(0, 2, 1, 3, 4).reshape(rows * PATCH, columns * PATCH, 3))


@dataclass(frozen=True)
class Augment:
    """How a dedupe view differs from the stored image: duplicates found in practice are mostly the same picture at
    another, not integer-related resolution and compressed again; cropped copies are rarer.

    Each view is altered with probability ``probability`` (so a pair is sometimes the image and an altered copy, and
    sometimes two altered copies). An altered view is, in random order, scaled down to a linear factor drawn
    log-uniformly from [``min_scale``, 1] of the model's input size (with one of five resampling filters) and, with
    probability ``recompress``, saved as JPEG (quality ``quality``, 4:2:0 or 4:4:4) or, ``webp`` of the time, as lossy
    WebP. With probability ``crop`` each side loses up to ``max_crop`` of its length first."""

    probability: float = 0.8
    min_scale: float = 0.3
    recompress: float = 0.7
    quality: tuple[int, int] = (50, 95)
    webp: float = 0.3
    crop: float = 0.05
    max_crop: float = 0.04

    def __call__(self, image: PIL.Image.Image, rng: np.random.Generator, seq_len: int) -> PIL.Image.Image:
        if rng.random() >= self.probability:
            return image
        if rng.random() < self.crop:
            w, h = image.size
            left, top, right, bottom = rng.uniform(0, self.max_crop, 4)
            image = image.crop((round(w * left), round(h * top), w - round(w * right), h - round(h * bottom)))
        steps = [self._rescale, self._recompress]
        if rng.random() < 0.5:
            steps.reverse()
        for step in steps:
            image = step(image, rng, seq_len)
        return image

    def _rescale(self, image: PIL.Image.Image, rng: np.random.Generator, seq_len: int) -> PIL.Image.Image:
        height, width = image_size_for_seq(image.height, image.width, seq_len)
        scale = math.exp(rng.uniform(math.log(self.min_scale), 0.0)) * min(height / image.height, width / image.width)
        if scale >= 1.0:
            return image
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        return image.resize(size, _FILTERS[rng.integers(len(_FILTERS))])

    def _recompress(self, image: PIL.Image.Image, rng: np.random.Generator, seq_len: int) -> PIL.Image.Image:
        if rng.random() >= self.recompress:
            return image
        quality = int(rng.integers(self.quality[0], self.quality[1] + 1))
        buffer = io.BytesIO()
        if rng.random() < self.webp:
            image.save(buffer, format="WEBP", quality=quality, method=4)
        else:
            image.save(buffer, format="JPEG", quality=quality, subsampling=int(rng.choice([0, 2])))
        buffer.seek(0)
        with PIL.Image.open(buffer) as decoded:
            return decoded.convert("RGB")


# -- tags and captions ----------------------------------------------------------------------------------------------


class TagTable:
    """Every image's tags (pack tag IDs in source order) and what the samplers need about them, read once from the
    pack index into arrays (31 M rows in about 15 s; forked loader workers share them)."""

    def __init__(self, pack: Pack):
        conn = pack.connect()
        try:
            rows = conn.execute("SELECT id, name, category FROM tags").fetchall()
            parts, cursor = [], conn.execute("SELECT image, tag FROM image_tags")
            while chunk := cursor.fetchmany(1 << 20):
                parts.append(np.array(chunk, dtype=np.int64))
            digests = dict(conn.execute("SELECT id, substr(blake3, 1, 8) FROM images"))
        finally:
            conn.close()
        size = max((r[0] for r in rows), default=0) + 1
        self.names: list[str] = [""] * size
        self.categories: list[str | None] = [None] * size
        for tag, name, category in rows:
            self.names[tag], self.categories[tag] = name, category
        pairs = np.concatenate(parts) if parts else np.zeros((0, 2), dtype=np.int64)
        # Pack IDs -> image indices. The pack may still be growing: drop the rows of images Pack did not list.
        by_id = np.argsort(pack.ids)
        found = np.minimum(np.searchsorted(pack.ids[by_id], pairs[:, 0]), max(len(pack.ids) - 1, 0))
        known = pack.ids[by_id[found]] == pairs[:, 0] if len(pack.ids) else np.zeros(len(pairs), dtype=bool)
        image = by_id[found[known]]
        order = np.argsort(image, kind="stable")  # keeps each image's rows in their (image, position) order
        self.tags = pairs[known, 1][order].astype(np.int32)
        self.offsets = np.searchsorted(image[order], np.arange(len(pack.ids) + 1))
        self.counts = np.bincount(self.tags, minlength=size)  # images per tag
        # The first 8 bytes of each image's BLAKE3: equal for identical images.
        self.digests = np.array([int.from_bytes(digests[int(i)], "little", signed=True) for i in pack.ids], dtype=np.int64)
        # Per-image seeds from the key, so an image's samples do not depend on anything else in the pack.
        self.seeds = np.array(
            [int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "little") for key in pack.keys],
            dtype=np.uint64,
        )

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def of(self, i: int) -> np.ndarray:
        return self.tags[self.offsets[i] : self.offsets[i + 1]]


def holdout(table: TagTable, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """``(training, held out)`` image indices: about ``fraction`` of the images, chosen by a hash of their keys (the
    smallest of a group of identical images, so the group lands on one side). The split depends neither on the seed
    nor on the encoder that made the stored bytes."""
    group = np.unique(table.digests, return_inverse=True)[1]
    smallest = np.full(group.max(initial=-1) + 1, np.iinfo(np.uint64).max, dtype=np.uint64)
    np.minimum.at(smallest, group, table.seeds)
    held = (smallest[group] >> np.uint64(11)).astype(np.float64) / 2.0**53 < fraction
    return np.flatnonzero(~held), np.flatnonzero(held)


@dataclass(frozen=True)
class CaptionConfig:
    """How :class:`CaptionMaker` builds a caption.

    A caption takes ``L`` of the image's tags, ``L`` drawn log-uniformly from 1 to all of them (so one-tag captions and
    full ones are both common), then cut at the text tower's ``context`` tokens. Tags are drawn without replacement
    with probability ``uniform / n + (1 - uniform) * w / sum(w)``, ``w = (images with the tag + k) ** -alpha``: common
    tags (``1girl``) are drawn less, and ``k`` keeps a tag seen on one or two images from dominating every caption it
    can appear in. A tag on fewer than ``min_count`` images is never a caption's only tag. Categories in ``exclude``
    (the dataset's conditioning words, ``training``) are left out. Tags keep their source order (characters,
    copyrights and artists first), or are shuffled with probability ``shuffle``."""

    exclude: frozenset[str] = frozenset({"training"})
    alpha: float = 0.5
    k: float = 100.0
    uniform: float = 0.0
    min_count: int = 10
    shuffle: float = 0.5
    context: int = 64


class CaptionMaker:
    """Captions for the images of a :class:`TagTable` (see :class:`CaptionConfig`). The tokenizer is loaded in each
    process on first use."""

    def __init__(self, table: TagTable, config: CaptionConfig, tokenizer_json: str):
        self.table, self.config, self.tokenizer_json = table, config, tokenizer_json
        self.weights = (table.counts + config.k) ** -config.alpha
        self.allowed = np.array([category not in config.exclude for category in table.categories])
        self.rare = table.counts < config.min_count
        self._tokenizer = None
        self._lengths: dict[int, int] = {}

    def __getstate__(self) -> dict:
        return self.__dict__ | {"_tokenizer": None, "_lengths": {}}

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from picurate.common.siglip_text import Tokenizer

            self._tokenizer = Tokenizer(self.tokenizer_json, self.config.context)
        return self._tokenizer

    def _length(self, tag: int) -> int:
        """Tokens of the tag after ``, `` in a caption."""
        if tag not in self._lengths:
            self._lengths[tag] = (
                len(self.tokenizer.tokenizer.encode(" " + self.table.names[tag].lower(), add_special_tokens=False).ids) + 1
            )
        return self._lengths[tag]

    def __call__(self, i: int, rng: np.random.Generator) -> tuple[str, np.ndarray]:
        """The caption of image ``i`` and its tags."""
        config = self.config
        tags = self.table.of(i)
        if len(tags) == 0:
            raise ValueError(f"image {i} has no tags to caption it with")
        tags = tags[self.allowed[tags]]
        if len(tags) == 0:
            tags = self.table.of(i)
        n = len(tags)
        count = min(n, int((n + 1) ** rng.random()))
        pool = np.arange(n)
        if count == 1 and not self.rare[tags].all():
            pool = pool[~self.rare[tags]]
        weights = self.weights[tags[pool]]
        p = config.uniform / len(pool) + (1 - config.uniform) * weights / weights.sum()
        # Efraimidis-Spirakis: the largest u ** (1 / p) are a weighted sample without replacement, in drawing order.
        drawn = pool[np.argsort(np.log(rng.random(len(pool))) / p)[::-1][:count]]
        budget, kept = config.context - 1, 0  # one position for <eos>
        for position in drawn:
            budget -= self._length(int(tags[position]))
            if budget < 0 and kept:
                break
            kept += 1
        while True:
            chosen = drawn[:kept] if rng.random() < config.shuffle else np.sort(drawn[:kept])
            text = ", ".join(self.table.names[t] for t in tags[chosen])
            if kept == 1 or len(self.tokenizer.tokenizer.encode(text.lower(), add_special_tokens=False).ids) < config.context:
                return text, tags[chosen]
            kept -= 1  # the per-tag estimate was short (tokens merged across a separator)


# -- datasets -------------------------------------------------------------------------------------------------------


def default_tokenizer_json() -> str:
    from huggingface_hub import hf_hub_download

    from picurate.common.siglip_text import DEFAULT_REPO, DEFAULT_REVISION

    return hf_hub_download(DEFAULT_REPO, "tokenizer.json", revision=DEFAULT_REVISION)


class PackDataset(Dataset):
    """Samples of one route, indexed by ``(epoch, image index)`` as :class:`EpochBatchSampler` yields them; batch them
    with :func:`collate`."""

    def __init__(
        self,
        pack: Pack,
        table: TagTable,
        route: str,
        *,
        seed: int = 0,
        seq_len: int = DEFAULT_SEQ_LEN,
        captions: CaptionConfig = CaptionConfig(),
        augment: Augment = Augment(),
        tokenizer_json: str | None = None,
    ):
        if route not in ROUTES:
            raise ValueError(f"route must be one of {ROUTES}")
        self.pack, self.table, self.route, self.seed, self.seq_len, self.augment = pack, table, route, seed, seq_len, augment
        self.captions = CaptionMaker(table, captions, tokenizer_json or default_tokenizer_json()) if route == "search" else None

    def __len__(self) -> int:
        return len(self.pack)

    def __getitem__(self, item: tuple[int, int]) -> dict:
        epoch, i = item
        rng = np.random.default_rng([self.seed, epoch, int(self.table.seeds[i])])
        image = self.pack.image(i)
        sample = {"index": i, "digest": self.table.digests[i]}
        if self.route == "dedupe":
            sample["views"] = [to_patches(self.augment(image, rng, self.seq_len), self.seq_len) for _ in range(2)]
        else:
            sample["views"] = [to_patches(image, self.seq_len)]
            sample["caption"], sample["caption_tags"] = self.captions(i, rng)
            sample["tags"] = self.table.of(i)
            sample["text"] = self.captions.tokenizer([sample["caption"]])[0]
        return sample


@dataclass
class Collate:
    """Stacks samples into a batch: ``patches`` (uint8), ``patch_coord`` and ``patch_valid`` padded to ``seq_len``
    (dedupe: all first views, then all second views, as ``clip-train-bench`` takes them), ``labels``, ``index`` and,
    for search, ``text`` (token IDs), ``captions`` and ``caption_tags``."""

    seq_len: int = DEFAULT_SEQ_LEN

    def __call__(self, samples: list[dict]) -> dict:
        views = [sample["views"][k] for k in range(len(samples[0]["views"])) for sample in samples]
        patches = torch.zeros((len(views), self.seq_len, PATCH * PATCH * 3), dtype=torch.uint8)
        coords = torch.zeros((len(views), self.seq_len, 2), dtype=torch.int16)  # int64 on the GPU
        valid = torch.zeros((len(views), self.seq_len), dtype=torch.bool)
        for row, (p, c) in enumerate(views):
            patches[row, : len(p)] = torch.from_numpy(p)
            coords[row, : len(c)] = torch.from_numpy(c)
            valid[row, : len(p)] = True
        digests = np.array([sample["digest"] for sample in samples])
        labels = digests[:, None] == digests[None, :]
        batch = {
            "patches": patches,
            "patch_coord": coords,
            "patch_valid": valid,
            "index": torch.tensor([s["index"] for s in samples]),
        }
        if "caption" in samples[0]:
            # labels[i, j]: image i has every tag of caption j.
            image_tags = [set(sample["tags"].tolist()) for sample in samples]
            for j, sample in enumerate(samples):
                wanted = sample["caption_tags"].tolist()
                labels[:, j] |= [all(tag in tags for tag in wanted) for tags in image_tags]
            batch["text"] = torch.stack([sample["text"] for sample in samples])
            batch["caption_tags"] = [sample["caption_tags"] for sample in samples]
            batch["captions"] = [sample["caption"] for sample in samples]
        batch["labels"] = torch.from_numpy(labels)
        return batch


class EpochBatchSampler(Sampler):
    """Batches of ``(epoch, image index)`` over ``indices``, shuffled per epoch by ``(seed, epoch)``; the last
    incomplete batch is dropped. ``set_epoch(epoch, start)`` resumes at batch ``start`` of an epoch."""

    def __init__(self, indices: np.ndarray, batch_size: int, seed: int = 0):
        self.indices, self.batch_size, self.seed = np.asarray(indices), batch_size, seed
        self.epoch, self.start = 0, 0

    def set_epoch(self, epoch: int, start: int = 0) -> None:
        self.epoch, self.start = epoch, start

    def __len__(self) -> int:
        return len(self.indices) // self.batch_size - self.start

    def __iter__(self):
        order = self.indices[np.random.default_rng([self.seed, self.epoch]).permutation(len(self.indices))]
        for b in range(self.start, len(self.indices) // self.batch_size):
            yield [(self.epoch, int(i)) for i in order[b * self.batch_size : (b + 1) * self.batch_size]]


def make_loader(dataset: PackDataset, sampler: EpochBatchSampler, workers: int, prefetch: int = 2) -> DataLoader:
    """A loader over forked workers (they share the tag table instead of unpickling a copy each), pinned for
    asynchronous copies to the GPU. Workers ignore Ctrl-C, which is the training loop's to handle. Up to ``workers *
    prefetch`` batches wait in shared memory (a dedupe batch of 256 at 1024 patches is about 400 MB)."""
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=Collate(dataset.seq_len),
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        prefetch_factor=prefetch if workers else None,
        multiprocessing_context="fork" if workers else None,
        worker_init_fn=ignore_interrupt,
    )


def ignore_interrupt(_worker: int) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def to_device(batch: dict, device, dtype: torch.dtype = torch.bfloat16) -> dict:
    """The image tower's input on ``device``: patches normalized to [-1, 1] (SigLIP's mean and std of 0.5). The
    subtraction is exact in bf16 and fp16, so the division is the only rounding, as when normalizing in FP32 first."""
    pin = batch["patches"].is_pinned()
    patches = batch["patches"].to(device, non_blocking=pin).to(dtype).sub_(127.5).div_(127.5)
    return {
        "patches": patches,
        "patch_coord": batch["patch_coord"].to(device, non_blocking=pin).long(),
        "patch_valid": batch["patch_valid"].to(device, non_blocking=pin),
    }


# -- command line ---------------------------------------------------------------------------------------------------


def _open(pack_dir: str, route: str, seed: int, seq_len: int, alpha: float, k: float, uniform: float, min_count: int):
    started = time.perf_counter()
    pack = Pack(pack_dir)
    table = TagTable(pack)
    click.echo(f"{len(pack)} images, {len(table.tags)} image tags, read in {time.perf_counter() - started:.1f} s", err=True)
    captions = CaptionConfig(alpha=alpha, k=k, uniform=uniform, min_count=min_count)
    return PackDataset(pack, table, route, seed=seed, seq_len=seq_len, captions=captions)


def _options(function):
    for option in reversed(
        [
            click.argument("pack_dir", type=click.Path(exists=True, file_okay=False)),
            click.option("--route", type=click.Choice(ROUTES), default="search", show_default=True),
            click.option("--seed", type=int, default=0, show_default=True),
            click.option(
                "--seq-len", type=click.IntRange(16), default=DEFAULT_SEQ_LEN, show_default=True, help="Patches per image."
            ),
            click.option("--alpha", type=float, default=CaptionConfig.alpha, show_default=True, help="Tag weight exponent."),
            click.option("-k", type=float, default=CaptionConfig.k, show_default=True, help="Tag weight count offset."),
            click.option("--uniform", type=click.FloatRange(0, 1), default=CaptionConfig.uniform, show_default=True),
            click.option("--min-count", type=click.IntRange(1), default=CaptionConfig.min_count, show_default=True),
            click.option("--batch-size", "-b", type=click.IntRange(1), default=16, show_default=True),
        ]
    ):
        function = option(function)
    return function


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main():
    """Look at and time the training batches made from a pack."""


@main.command()
@_options
@click.option("--epoch", type=int, default=0, show_default=True)
@click.option("--out", type=click.Path(file_okay=False), default=None, help="Also save every view as PNG here.")
def sample(pack_dir, route, seed, seq_len, alpha, k, uniform, min_count, batch_size, epoch, out):
    """Print one batch: keys, captions and positives besides the diagonal."""
    dataset = _open(pack_dir, route, seed, seq_len, alpha, k, uniform, min_count)
    sampler = EpochBatchSampler(np.arange(len(dataset)), batch_size, seed)
    sampler.set_epoch(epoch)
    batch = Collate(seq_len)([dataset[item] for item in next(iter(sampler))])
    labels = batch["labels"].numpy()
    n = len(batch["index"])
    if out:
        os.makedirs(out, exist_ok=True)
    for row, i in enumerate(batch["index"].tolist()):
        key = dataset.pack.keys[i]
        others = [c for c in np.flatnonzero(labels[:, row]) if c != row]
        line = f"{row:3d} {key}"
        if "captions" in batch:
            tokens = int((batch["text"][row] != dataset.captions.tokenizer.pad).sum())
            line += f"\n    [{tokens} tokens] {batch['captions'][row]}"
            line += f"\n    also matches images {others}" if others else ""
        elif others:
            line += f"\n    identical to {others}"
        click.echo(line)
        if out:
            for view in range(len(batch["patches"]) // n):
                r = view * n + row
                count = int(batch["patch_valid"][r].sum())
                image = from_patches(batch["patches"][r, :count].numpy(), batch["patch_coord"][r, :count].numpy())
                image.save(Path(out) / f"{row:03d}-{view}.png")
    click.echo(f"{int(labels.sum()) - n} positives off the diagonal of {n}x{n}", err=True)


@main.command()
@_options
@click.option("-j", "--workers", type=click.IntRange(0), default=8, show_default=True)
@click.option("--batches", type=click.IntRange(1), default=50, show_default=True)
def bench(pack_dir, route, seed, seq_len, alpha, k, uniform, min_count, batch_size, workers, batches):
    """Time the loader: samples per second with WORKERS processes."""
    dataset = _open(pack_dir, route, seed, seq_len, alpha, k, uniform, min_count)
    loader = make_loader(dataset, EpochBatchSampler(np.arange(len(dataset)), batch_size, seed), workers)
    iterator = iter(loader)
    next(iterator)  # workers start and fill their queues
    started = time.perf_counter()
    positives = 0
    for _ in range(batches):
        batch = next(iterator)
        positives += int(batch["labels"].sum()) - len(batch["index"])
    seconds = time.perf_counter() - started
    click.echo(
        f"{batches * batch_size / seconds:.1f} samples/s ({1000 * seconds / batches:.0f} ms per batch of {batch_size}),"
        f" {positives / batches:.1f} positives off the diagonal per batch"
    )


if __name__ == "__main__":
    main()
