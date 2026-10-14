"""Fine-tuning SigLIP2 so400m on a pack (see docs/training-clip.md): ``clip-train run RUN_DIR --config FILE``.

A run directory holds everything a run needs to continue:

* ``config.toml``: the configuration it was started with (:class:`Config`).
* ``binding.json``: the pack it trains on (settings, image count and a hash of the keys). Batches are a function of
  the seed and the pack's images (:mod:`clip_training.data`), so a run continues only on the same images.
* ``checkpoints/step-NNNNNNN/``: the trained parameters (``trainable.safetensors``; frozen ones come from the base
  checkpoint), the optimizer and loss-scaler state (``optimizer.pt``) and the position (``state.json``: step, epoch,
  batch within the epoch). Written to ``*.partial`` and renamed when complete; the newest ``keep_last`` are kept.
* ``tensorboard/``: the metrics.
* ``.lock``: held while a ``run`` trains, so a second one in the same directory is refused.

Running ``clip-train run RUN_DIR`` again continues from the newest checkpoint; the data position follows from
``(epoch, batch)``. The configuration must stay the same, except for what does not change the result: loader
workers, logging, evaluation and checkpoint intervals, the pack's path (the binding identifies the pack) and, with the
schedule-free optimizer, ``run.steps`` (training longer). Anything else is a new run, which can start from another
run's weights with ``model.init_from`` (an ``export``). Ctrl-C finishes the step, writes a checkpoint and stops; so
does a failure to load a batch (an unreachable pack), before the error ends the run. A step with a non-finite loss or
gradient is skipped, and too many in a row end the run without a checkpoint. The GPU's own nondeterminism (attention
and matmul reductions) makes a continued run differ from an uninterrupted one by rounding, not by its data.

Schedule-free AdamW trains on one weight sequence and evaluates on another: evaluation, checkpoints and exports all
use the evaluation weights (``optimizer.eval()``), and a checkpoint stores the optimizer in that mode.
"""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import json
import logging
import math
import os
import shutil
import signal
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import click
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from picurate.common.log import setup_logging
from picurate.common.siglip_image import ATTENTIONS, DEFAULT_REPO, DEFAULT_REVISION, IMAGE_TOWER, ImageTower, load_image_tower
from picurate.common.siglip_text import TextTower, load_text_encoder

from .data import (
    DEFAULT_SEQ_LEN,
    ROUTES,
    Augment,
    CaptionConfig,
    Collate,
    EpochBatchSampler,
    PackDataset,
    TagTable,
    holdout,
    ignore_interrupt,
    make_loader,
    to_device,
)
from .pack import Pack

logger = logging.getLogger(__name__)

PRECISIONS = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
OPTIMIZERS = ("schedulefree_adamw", "adamw")
EVAL_SEED = 0x5EED  # captions and views of the evaluation set: the same in every run
CHECKPOINTS = "checkpoints"
MAX_SKIPPED = 10  # consecutive steps with non-finite gradients that end a run


# -- configuration --------------------------------------------------------------------------------------------------


@dataclass
class RunConfig:
    route: str = "dedupe"  # dedupe | search
    seed: int = 0
    steps: int = 10000  # optimizer steps
    log_every: int = 10


@dataclass
class DataConfig:
    pack: str = ""
    holdout: float = 0.01  # share of the pack kept out of training (see data.holdout)
    seq_len: int = DEFAULT_SEQ_LEN
    batch_size: int = 256  # samples the loss sees per step
    workers: int = 8
    prefetch: int = 2  # batches per worker waiting in shared memory
    captions: CaptionConfig = field(default_factory=CaptionConfig)
    augment: Augment = field(default_factory=Augment)


@dataclass
class LoraConfig:
    rank: int = 16
    alpha: float = 16.0


@dataclass
class ModelConfig:
    repo: str = DEFAULT_REPO  # the Hugging Face checkpoint both towers start from
    revision: str = DEFAULT_REVISION  # "": the newest
    attention: str = ""  # flash | sdpa; "": flash with flash-attn installed, else sdpa (same results, speed differs)
    train: str = "image"  # search: image | text | both (dedupe always trains the image tower)
    freeze_blocks: int = 0  # the patch embedding and the lowest N of the image tower's 27 blocks
    freeze_text_layers: int = 0  # the embeddings and the lowest N of the text tower's 27 layers
    train_token_embedding: bool = False  # the text tower's 256k x 1152 token table (295 M parameters)
    grad_checkpointing: bool = True
    compile: bool = False  # torch.compile the towers' forward: 5-14% faster steps after minutes of compiling
    precision: str = "bf16"  # bf16 | fp16 | fp32 autocast; trained parameters stay FP32
    init_from: str = ""  # an export to start from instead of the base checkpoints
    lora: LoraConfig | None = None  # LoRA on the trained blocks instead of training them fully


@dataclass
class LossConfig:
    scale: float | None = None  # log of the logit scale; default: the checkpoint's (search) or log(10) (dedupe)
    bias: float | None = None  # default: the checkpoint's (search) or -10 (dedupe)
    learn_scale_bias: bool = True


@dataclass
class OptimConfig:
    name: str = "schedulefree_adamw"  # schedulefree_adamw | adamw (with warm-up and cosine decay over run.steps)
    lr: float = 1e-5  # image tower
    lr_text: float = 0.0  # text tower; 0: the same as lr
    lr_scale_bias: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0  # on matrices only
    warmup_steps: int = 500
    max_grad_norm: float = 1.0  # 0: no clipping
    micro_batch: int = 0  # GradCache micro-batch, a divisor of data.batch_size; 0: the whole batch in one pass


@dataclass
class EvalConfig:
    every_steps: int = 1000  # 0: never
    images: int = 1024  # held-out images evaluated, in batches of optim.micro_batch (or data.batch_size)
    at_start: bool = True


@dataclass
class CheckpointConfig:
    every_steps: int = 1000  # 0: never
    every_minutes: float = 30.0  # 0: never
    keep_last: int = 3  # 0: keep all


@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    @classmethod
    def load(cls, path: Path) -> Config:
        with open(path, "rb") as stream:
            config = _build(cls, tomllib.load(stream), "")
        config.check()
        return config

    def check(self) -> None:
        """Refuses settings that cannot work, before minutes of loading."""
        r, d, m, o, a, c = self.run, self.data, self.model, self.optim, self.data.augment, self.data.captions
        rules = [
            (r.route in ROUTES, f"run.route must be one of {ROUTES}"),
            (m.train in ("image", "text", "both"), "model.train must be image, text or both"),
            (r.route == "search" or m.train == "image", "the dedupe route trains the image tower only (model.train = 'image')"),
            (m.precision in PRECISIONS, f"model.precision must be one of {tuple(PRECISIONS)}"),
            (o.name in OPTIMIZERS, f"optim.name must be one of {OPTIMIZERS}"),
            (bool(d.pack), "data.pack is not set"),
            (not m.init_from or os.path.isfile(m.init_from), f"model.init_from {m.init_from} does not exist"),
            (r.seed >= 0, "run.seed must not be negative"),
            (r.steps >= 1 and r.log_every >= 1, "run.steps and run.log_every must be at least 1"),
            (0 < d.holdout < 1, "data.holdout must be between 0 and 1"),
            (d.seq_len >= 1 and d.batch_size >= 1, "data.seq_len and data.batch_size must be at least 1"),
            (d.workers >= 0 and d.prefetch >= 1, "data.workers must not be negative, data.prefetch at least 1"),
            (all(0 <= v <= 1 for v in (a.probability, a.recompress, a.webp, a.crop)), "data.augment shares must be in [0, 1]"),
            (0 < a.min_scale <= 1 and 0 <= a.max_crop < 0.5, "data.augment.min_scale must be in (0, 1], max_crop in [0, 0.5)"),
            (1 <= a.quality[0] <= a.quality[1] <= 100, "data.augment.quality must be [low, high] within 1-100"),
            (0 <= c.uniform <= 1 and 0 <= c.shuffle <= 1, "data.captions.uniform and shuffle must be in [0, 1]"),
            (c.alpha >= 0 and c.k >= 0 and c.min_count >= 1 and c.context >= 2, "data.captions values out of range"),
            (m.freeze_blocks >= 0 and m.freeze_text_layers >= 0, "model.freeze_* must not be negative"),
            (m.attention in ("",) + ATTENTIONS, f"model.attention must be one of {ATTENTIONS}"),
            (m.lora is None or (m.lora.rank >= 1 and m.lora.alpha > 0), "model.lora.rank and alpha must be positive"),
            (
                o.lr > 0 and o.lr_text >= 0 and o.lr_scale_bias >= 0 and o.weight_decay >= 0,
                "optim learning rates and weight decay out of range",
            ),
            (all(0 <= b < 1 for b in o.betas) and len(o.betas) == 2, "optim.betas must be two values in [0, 1)"),
            (o.warmup_steps >= 0 and o.max_grad_norm >= 0 and o.micro_batch >= 0, "optim values must not be negative"),
            (not o.micro_batch or d.batch_size % o.micro_batch == 0, "data.batch_size must be a multiple of optim.micro_batch"),
            (self.eval.every_steps >= 0 and self.eval.images >= 1, "eval.every_steps must not be negative, eval.images >= 1"),
            (
                min(self.checkpoint.every_steps, self.checkpoint.every_minutes, self.checkpoint.keep_last) >= 0,
                "checkpoint values must not be negative",
            ),
        ]
        for ok, message in rules:
            if not ok:
                raise ValueError(message)

    def flat(self) -> dict[str, object]:
        """``{"section.key": value}`` of every setting, defaults included."""
        out = {}

        def walk(value, prefix):
            if dataclasses.is_dataclass(value):
                for f in dataclasses.fields(value):
                    walk(getattr(value, f.name), f"{prefix}{f.name}.")
            else:
                out[prefix[:-1]] = sorted(value) if isinstance(value, frozenset) else value

        walk(self, "")
        return out

    def mutable(self) -> set[str]:
        """Settings a continued run may change: they do not change what it computes."""
        keys = {"run.log_every", "data.pack", "data.workers", "data.prefetch", "eval.every_steps", "model.compile"}
        keys.add("model.attention")
        keys |= {"checkpoint.every_steps", "checkpoint.every_minutes", "checkpoint.keep_last"}
        if self.optim.name == "schedulefree_adamw":
            keys.add("run.steps")
        return keys


def _build(cls, data: dict, path: str):
    """``cls`` (a dataclass) from a TOML table, refusing unknown keys and converting lists to tuples or sets where
    the default is one."""
    names = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(names)
    if unknown:
        raise ValueError(f"unknown setting(s) in [{path or 'top level'}]: {', '.join(sorted(unknown))}")
    values = {}
    for name, value in data.items():
        f = names[name]
        default = f.default_factory() if f.default_factory is not dataclasses.MISSING else f.default
        where = f"{path}.{name}" if path else name
        if name == "lora" and cls is ModelConfig:
            values[name] = _build(LoraConfig, value, where)
        elif dataclasses.is_dataclass(default):
            values[name] = _build(type(default), value, where)
        elif isinstance(default, tuple):
            values[name] = tuple(value)
        elif isinstance(default, frozenset):
            values[name] = frozenset(value)
        elif isinstance(default, float) and isinstance(value, int):
            values[name] = float(value)
        else:
            values[name] = value
    return cls(**values)


# -- model ----------------------------------------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trained low-rank update ``up(down(x)) * alpha / rank`` (zero at the start)."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.down = nn.Linear(base.in_features, rank, bias=False, device=base.weight.device)
        self.up = nn.Linear(rank, base.out_features, bias=False, device=base.weight.device)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.up(self.down(x)) * self.scale

    def merged(self) -> nn.Linear:
        linear = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        weight = self.base.weight.float() + (self.up.weight.float() @ self.down.weight.float()) * self.scale
        linear.weight.data.copy_(weight)
        if self.base.bias is not None:
            linear.bias.data.copy_(self.base.bias.float())
        return linear


def add_lora(module: nn.Module, config: LoraConfig) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, config.rank, config.alpha))
        else:
            add_lora(child, config)


def merge_lora(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            setattr(module, name, child.merged())
        else:
            merge_lora(child)


class Towers(nn.Module):
    """The image tower, the text tower (search route) and the loss's learned logit scale and bias."""

    def __init__(self, image: nn.Module, text: TextTower | None, scale: float, bias: float):
        super().__init__()
        self.image, self.text = image, text
        self.scale = nn.Parameter(torch.tensor(float(scale)))
        self.bias = nn.Parameter(torch.tensor(float(bias)))


def build_model(config: Config) -> Towers:
    """The towers as the configuration trains them, on the CPU in FP32: base checkpoints (or ``init_from``), frozen
    parts, LoRA adapters (seeded by ``run.seed``) and gradient checkpointing."""
    m, route = config.model, config.run.route
    exported = load_file(m.init_from) if m.init_from else None
    attention = m.attention or None
    if exported is None:
        image = load_image_tower(m.repo, m.revision or None, attention=attention)
    else:
        image = ImageTower(attention=attention)
    if m.freeze_blocks > len(image.blocks):
        raise ValueError(f"model.freeze_blocks is {m.freeze_blocks}; the image tower has {len(image.blocks)} blocks")
    text, scale, bias = None, math.log(10.0), -10.0
    if route == "search":
        encoder = load_text_encoder(m.repo, m.revision or None)
        text, scale, bias = encoder.tower, encoder.logit_scale, encoder.logit_bias
        if m.freeze_text_layers > len(text.layers):
            raise ValueError(f"model.freeze_text_layers is {m.freeze_text_layers}; the text tower has {len(text.layers)}")
    if exported is not None:
        image.load_state_dict({k[6:]: v for k, v in exported.items() if k.startswith("image.")})
        if text is not None and any(k.startswith("text.") for k in exported):
            text.load_state_dict({k[5:]: v for k, v in exported.items() if k.startswith("text.")})
        if "logit_scale" in exported:
            scale, bias = float(exported["logit_scale"]), float(exported["logit_bias"])
    scale = config.loss.scale if config.loss.scale is not None else scale
    bias = config.loss.bias if config.loss.bias is not None else bias
    towers = Towers(image, text, scale, bias)
    towers.requires_grad_(False)
    towers.scale.requires_grad_(config.loss.learn_scale_bias)
    towers.bias.requires_grad_(config.loss.learn_scale_bias)
    torch.manual_seed(config.run.seed)
    if route == "dedupe" or m.train in ("image", "both"):
        blocks = image.blocks[m.freeze_blocks :]
        heads = [image.norm, image.attn_pool]
        if m.freeze_blocks == 0 and m.lora is None:
            image.embeds.requires_grad_(True)
        _train(blocks, heads, m.lora)
    if text is not None and m.train in ("text", "both"):
        layers = text.layers[m.freeze_text_layers :]
        _train(layers, [text.final_layer_norm, text.head], m.lora)
        if m.freeze_text_layers == 0 and m.lora is None:
            text.position_embedding.requires_grad_(True)
            text.token_embedding.requires_grad_(m.train_token_embedding)
    if m.grad_checkpointing:
        image.set_grad_checkpointing(True)
        if text is not None:
            text.grad_checkpointing = True
    return towers


def _train(blocks, heads: list[nn.Module], lora: LoraConfig | None) -> None:
    for head in heads:
        head.requires_grad_(True)
    for block in blocks:
        if lora is None:
            block.requires_grad_(True)
        else:
            add_lora(block, lora)


def sigmoid_loss(a: torch.Tensor, b: torch.Tensor, labels: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor):
    """SigLIP's pairwise sigmoid loss with a matrix of positives: ``labels[i, j]`` says whether ``a[i]`` and ``b[j]``
    match."""
    a, b = F.normalize(a.float(), dim=-1), F.normalize(b.float(), dim=-1)
    logits = a @ b.T * scale.exp() + bias
    return -F.logsigmoid((2 * labels.float() - 1) * logits).sum() / len(a)


# -- training -------------------------------------------------------------------------------------------------------


def pack_binding(pack: Pack) -> dict:
    keys = hashlib.blake2b("\n".join(pack.keys).encode(), digest_size=16).hexdigest()
    return {"meta": pack.meta, "images": len(pack), "keys_blake2b": keys}


def checkpoints(run_dir: Path) -> list[Path]:
    """Complete checkpoints, oldest first."""
    directory = run_dir / CHECKPOINTS
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.is_dir() and p.name.startswith("step-") and not p.suffix)


def _fsync(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Trainer:
    def __init__(self, run_dir: Path, config: Config, pack: Pack, resume: Path | None):
        self.run_dir, self.config = run_dir, config
        if not torch.cuda.is_available():
            raise click.ClickException("CUDA is not available")
        self.device = "cuda"
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"
        self.dtype = PRECISIONS[config.model.precision]

        started = time.perf_counter()
        self.pack = pack
        self.table = TagTable(self.pack)
        self.train_indices, held = holdout(self.table, config.data.holdout)
        if config.run.route == "search":
            tagged = np.diff(self.table.offsets) > 0
            if not tagged.all():
                logger.warning("%d images without tags are left out of the search route", int((~tagged).sum()))
                self.train_indices, held = self.train_indices[tagged[self.train_indices]], held[tagged[held]]
        if len(held) == 0:
            raise click.ClickException("no held-out images to evaluate on")
        order = np.random.default_rng(EVAL_SEED).permutation(len(held))[: config.eval.images]
        self.eval_indices = np.sort(held[order])
        # Evaluation passes take the shape of the training passes (the last one padded).
        self.eval_batch = config.optim.micro_batch or config.data.batch_size
        logger.info(
            "Pack: %d images (%d trained, %d held out, %d evaluated), read in %.1f s",
            len(self.pack), len(self.train_indices), len(held), len(self.eval_indices), time.perf_counter() - started,
        )  # fmt: skip
        data = config.data
        common = dict(seq_len=data.seq_len, captions=data.captions, augment=data.augment)
        self.dataset = PackDataset(self.pack, self.table, config.run.route, seed=config.run.seed, **common)
        tokenizer = self.dataset.captions.tokenizer_json if self.dataset.captions else None
        # The evaluation set is the same task in every run: default captions and views, whatever this run trains on.
        self.eval_dataset = PackDataset(
            self.pack, self.table, config.run.route, seed=EVAL_SEED, seq_len=data.seq_len, tokenizer_json=tokenizer
        )
        self.sampler = EpochBatchSampler(self.train_indices, data.batch_size, config.run.seed)
        self.batches_per_epoch = len(self.train_indices) // data.batch_size
        if self.batches_per_epoch == 0:
            raise click.ClickException("the pack holds fewer training images than one batch")
        self.loader = make_loader(self.dataset, self.sampler, data.workers, data.prefetch)

        self.model = build_model(config).to(self.device)
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        if self.dtype != torch.float32:
            for p in self.model.parameters():
                if not p.requires_grad and p is not self.model.scale and p is not self.model.bias:
                    p.data = p.data.to(self.dtype)  # frozen: nothing updates them, store them as autocast computes
        # Compiled forwards are kept apart from the model, so parameter names (and checkpoints) stay the same.
        self.image_forward, self.text_forward = self.model.image, self.model.text
        if config.model.compile:
            self.image_forward = torch.compile(self.model.image)
            if self.model.text is not None:
                self.text_forward = torch.compile(self.model.text)
        self.image_trained = any(p.requires_grad for p in self.model.image.parameters())
        self.text_trained = self.model.text is not None and any(p.requires_grad for p in self.model.text.parameters())
        self.names = {p: n for n, p in self.model.named_parameters() if p.requires_grad}
        self.optimizer, self.scheduler = self._optimizer()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.dtype == torch.float16)
        self.step, self.epoch, self.batch = 0, 0, 0
        if resume is not None:
            self._load(resume)
        self.schedule_free = config.optim.name == "schedulefree_adamw"
        if self.schedule_free:
            self.optimizer.train()
        self.model.train()
        # Drops what an earlier attempt logged after the checkpoint continued from (all of it without one).
        self.writer = SummaryWriter(run_dir / "tensorboard", purge_step=self.step + 1 if resume else 0)
        trained = sum(p.numel() for p in self.params)
        logger.info(
            "Route %s: %.1f M of %.0f M parameters trained%s, %s autocast",
            config.run.route, trained / 1e6, sum(p.numel() for p in self.model.parameters()) / 1e6,
            " (LoRA)" if config.model.lora else "", config.model.precision,
        )  # fmt: skip
        self._eval_labels = None
        self._eval_loader: DataLoader | None = None
        self.stop = False
        self.skipped = 0  # consecutive steps with non-finite gradients

    def _optimizer(self):
        """Parameter groups: each tower at its own learning rate (``lr`` for the image tower, ``lr_text`` for the text
        tower), each split into matrices (weight decay) and the rest (no decay: vectors, position and token
        embeddings, the pooling query), and the logit scale and bias at ``lr_scale_bias``."""
        o = self.config.optim
        exempt = {f"image.{name}" for name in self.model.image.no_weight_decay()} | {"image.attn_pool.latent"}
        groups = []
        for tower, lr in (("image", o.lr), ("text", o.lr_text or o.lr)):
            decay, flat = [], []
            for p in self.params:
                name = self.names[p]
                if p is self.model.scale or p is self.model.bias or not name.startswith(f"{tower}."):
                    continue
                if p.ndim < 2 or name in exempt or name.endswith("_embedding.weight"):
                    flat.append(p)
                else:
                    decay.append(p)
            groups.append({"params": decay, "lr": lr, "weight_decay": o.weight_decay, "name": tower})
            groups.append({"params": flat, "lr": lr, "weight_decay": 0.0, "name": f"{tower}_flat"})
        scale_bias = [p for p in (self.model.scale, self.model.bias) if p.requires_grad]
        groups.append({"params": scale_bias, "lr": o.lr_scale_bias, "weight_decay": 0.0, "name": "scale_bias"})
        groups = [g for g in groups if g["params"]]
        assert sum(len(g["params"]) for g in groups) == len(self.params), "every trained parameter in exactly one group"
        if o.name == "schedulefree_adamw":
            import schedulefree

            return schedulefree.AdamWScheduleFree(groups, betas=o.betas, warmup_steps=o.warmup_steps), None
        optimizer = torch.optim.AdamW(groups, betas=o.betas, fused=True)
        steps, warmup = self.config.run.steps, o.warmup_steps

        def factor(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, steps - warmup))))

        return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, factor)

    # -- one step

    def _embed(self, batch: dict, start: int, stop: int, sides: tuple[bool, bool]):
        """The two sides the loss pairs up for samples ``start:stop``: both views (dedupe) or images and texts
        (search); a side not in ``sides`` is None."""
        n = len(batch["index"])
        if self.config.run.route == "dedupe":
            views = [to_device(_rows(batch, s, e), self.device, self.dtype) for s, e in ((start, stop), (n + start, n + stop))]
            embeddings = self.image_forward({k: torch.cat([v[k] for v in views]) for k in views[0]})
            return embeddings.chunk(2)
        a = self.image_forward(to_device(_rows(batch, start, stop), self.device, self.dtype)) if sides[0] else None
        b = self.text_forward(batch["text"][start:stop].to(self.device, non_blocking=True)) if sides[1] else None
        return a, b

    def _loss(self, a, b, labels):
        return sigmoid_loss(a, b, labels, self.model.scale, self.model.bias)

    def train_step(self, batch: dict) -> tuple[float, float]:
        """One optimizer step; ``(loss, gradient norm)``. A step whose loss or gradient is not finite changes no
        weights (with fp16, the loss scaler skips it and lowers the scale)."""
        n = len(batch["index"])
        labels = batch["labels"].to(self.device, non_blocking=True)
        micro = self.config.optim.micro_batch or n
        autocast = torch.autocast("cuda", dtype=self.dtype, enabled=self.dtype != torch.float32)
        if micro >= n:
            with autocast:
                a, b = self._embed(batch, 0, n, (True, True))
            loss = self._loss(a, b, labels)
            self.scaler.scale(loss).backward()
        else:
            # GradCache: the loss of the whole batch with the activations of one micro-batch.
            spans = [(s, min(n, s + micro)) for s in range(0, n, micro)]
            with torch.no_grad(), autocast:
                parts = [self._embed(batch, s, e, (True, True)) for s, e in spans]
            a = torch.cat([p[0] for p in parts]).float().requires_grad_()
            b = torch.cat([p[1] for p in parts]).float().requires_grad_()
            del parts
            loss = self._loss(a, b, labels)
            self.scaler.scale(loss).backward()
            dedupe = self.config.run.route == "dedupe"
            sides = (True, True) if dedupe else (self.image_trained, self.text_trained)
            for s, e in spans:
                with autocast:
                    pa, pb = self._embed(batch, s, e, sides)
                pairs = [(t.float(), g[s:e]) for t, g in ((pa, a.grad), (pb, b.grad)) if t is not None]
                torch.autograd.backward([t for t, _ in pairs], [g for _, g in pairs])
        self.scaler.unscale_(self.optimizer)
        limit = self.config.optim.max_grad_norm
        norm = torch.nn.utils.clip_grad_norm_(self.params, limit if limit > 0 else float("inf"))
        loss, norm = loss.item(), norm.item()
        finite = math.isfinite(loss) and math.isfinite(norm)
        if finite or self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler is not None:
            self.scheduler.step()
        if finite or self.scaler.is_enabled():
            self.skipped = 0
        else:
            self.skipped += 1
            logger.warning("Step %d skipped: loss %s, gradient norm %s", self.step + 1, loss, norm)
            if self.skipped >= MAX_SKIPPED:
                raise click.ClickException(
                    f"{MAX_SKIPPED} steps in a row with non-finite gradients; the last checkpoint is the last good state"
                )
        return loss, norm

    # -- evaluation

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Metrics on the held-out images, with the evaluation weights."""
        if self.schedule_free:
            self.optimizer.eval()
        self.model.eval()
        size, count = self.eval_batch, len(self.eval_indices)
        if self._eval_loader is None:
            # Built once: its workers are forked before the first evaluation (not again from a process with live
            # loader threads every time) and kept, like the training loader's.
            padded = np.resize(self.eval_indices, -(-count // size) * size)  # the last batch filled up from the start
            batches = [[(0, int(i)) for i in padded[s : s + size]] for s in range(0, len(padded), size)]
            self._eval_loader = DataLoader(
                self.eval_dataset,
                batch_sampler=batches,
                collate_fn=Collate(self.config.data.seq_len),
                num_workers=self.config.data.workers,
                persistent_workers=self.config.data.workers > 0,
                multiprocessing_context="fork" if self.config.data.workers else None,
                worker_init_fn=ignore_interrupt,
            )
        loader = self._eval_loader
        a, b, caption_tags = [], [], []
        autocast = torch.autocast("cuda", dtype=self.dtype, enabled=self.dtype != torch.float32)
        for batch in loader:
            with autocast:
                pa, pb = self._embed(batch, 0, len(batch["index"]), (True, True))
            a.append(F.normalize(pa.float(), dim=-1))
            b.append(F.normalize(pb.float(), dim=-1))
            caption_tags.extend(batch.get("caption_tags", []))
        a, b, caption_tags = torch.cat(a)[:count], torch.cat(b)[:count], caption_tags[:count]
        if self._eval_labels is None:
            self._eval_labels = self._labels(caption_tags).to(self.device)
        labels = self._eval_labels
        similarity = a @ b.T
        logits = similarity * self.model.scale.float().exp() + self.model.bias.float()
        metrics = {"loss": float(-F.logsigmoid((2 * labels.float() - 1) * logits).sum() / len(a))}
        for k in (1, 5):
            metrics[f"a_to_b_r{k}"] = _recall(similarity, labels, k)
            metrics[f"b_to_a_r{k}"] = _recall(similarity.T, labels.T, k)
        positive = similarity.diagonal()
        hardest = similarity.masked_fill(labels, -2.0).max(1).values
        metrics |= {
            "positive_cos_mean": float(positive.mean()),
            "positive_cos_p05": float(positive.quantile(0.05)),
            "hardest_negative_cos_mean": float(hardest.mean()),
            "hardest_negative_cos_p95": float(hardest.quantile(0.95)),
        }
        self.model.train()
        if self.schedule_free:
            self.optimizer.train()
        return metrics

    def _labels(self, caption_tags: list[np.ndarray]) -> torch.Tensor:
        digests = self.table.digests[self.eval_indices]
        labels = digests[:, None] == digests[None, :]
        if caption_tags:
            image_tags = [set(self.table.of(int(i)).tolist()) for i in self.eval_indices]
            for j, tags in enumerate(caption_tags):
                wanted = tags.tolist()
                labels[:, j] |= [all(t in have for t in wanted) for have in image_tags]
        return torch.from_numpy(labels)

    # -- checkpoints

    def save(self) -> None:
        final = self.run_dir / CHECKPOINTS / f"step-{self.step:07d}"
        if final.exists():
            return  # continued from it and stopped before another step
        if self.schedule_free:
            self.optimizer.eval()
        if not all(bool(torch.isfinite(p).all()) for p in self.params):
            raise click.ClickException(f"non-finite weights at step {self.step}; no checkpoint written")
        epoch, batch = (self.epoch + 1, 0) if self.batch >= self.batches_per_epoch else (self.epoch, self.batch)
        partial = final.with_suffix(".partial")
        shutil.rmtree(partial, ignore_errors=True)
        partial.mkdir(parents=True)
        save_file({name: p.detach().contiguous() for p, name in self.names.items()}, partial / "trainable.safetensors")
        state = {"optimizer": self.optimizer.state_dict(), "scaler": self.scaler.state_dict()}
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        torch.save(state, partial / "optimizer.pt")
        position = {"step": self.step, "epoch": epoch, "batch": batch}
        (partial / "state.json").write_text(json.dumps(position) + "\n")
        for path in (*partial.iterdir(), partial):
            _fsync(path)
        os.replace(partial, final)
        _fsync(final.parent)
        if self.schedule_free:
            self.optimizer.train()
        keep = self.config.checkpoint.keep_last
        if keep:
            for old in checkpoints(self.run_dir)[:-keep]:
                shutil.rmtree(old)
        logger.info("Checkpoint %s", final.name)

    def _load(self, path: Path) -> None:
        tensors = load_file(path / "trainable.safetensors")
        by_name = {name: p for p, name in self.names.items()}
        if set(tensors) != set(by_name):
            raise click.ClickException(f"{path} holds other trained parameters than this configuration trains")
        for name, tensor in tensors.items():
            by_name[name].data.copy_(tensor)
        state = torch.load(path / "optimizer.pt", map_location=self.device, weights_only=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state["scaler"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        position = json.loads((path / "state.json").read_text())
        self.step, self.epoch, self.batch = position["step"], position["epoch"], position["batch"]
        logger.info("Continuing from %s (epoch %d, batch %d)", path.name, self.epoch, self.batch)

    # -- loop

    def log(self, values: dict[str, float], prefix: str) -> None:
        for key, value in values.items():
            self.writer.add_scalar(f"{prefix}/{key}", value, self.step)

    def run(self) -> None:
        config = self.config
        if self.step == 0 and config.eval.at_start:
            self.log(self.evaluate(), "eval")
        last_checkpoint = time.monotonic()
        losses, norms, waited, paused, samples = [], [], 0.0, 0.0, 0
        window = time.perf_counter()
        progress = tqdm(total=config.run.steps, initial=self.step, unit="step", desc="Training", dynamic_ncols=True)
        previous = signal.signal(signal.SIGINT, self._interrupt)
        try:
            while self.step < config.run.steps and not self.stop:
                self.sampler.set_epoch(self.epoch, self.batch)
                batches = iter(self.loader)
                while True:
                    fetched = time.perf_counter()
                    try:
                        batch = next(batches)
                    except StopIteration:
                        self.epoch, self.batch = self.epoch + 1, 0
                        break
                    except Exception:
                        # Between two steps the state is whole: keep it, then let the error end the run.
                        logger.error("Loading batch %d of epoch %d failed; writing a checkpoint", self.batch, self.epoch)
                        self.save()
                        raise
                    waited += time.perf_counter() - fetched
                    loss, norm = self.train_step(batch)
                    self.step += 1
                    self.batch += 1
                    losses.append(loss)
                    norms.append(norm)
                    samples += len(batch["index"])
                    progress.update()
                    progress.set_postfix(loss=f"{loss:.4f}", epoch=self.epoch)
                    if self.step % config.run.log_every == 0:
                        elapsed = time.perf_counter() - window - paused  # training only, not evaluation or saving
                        rates = {
                            f"lr_{g['name']}": g.get("scheduled_lr", g["lr"])
                            for g in self.optimizer.param_groups
                            if g["name"] in ("image", "text")
                        }
                        self.log(
                            {
                                "loss": float(np.mean(losses)),
                                "grad_norm": float(np.mean(norms)),
                                **rates,
                                "logit_scale": float(self.model.scale.detach().exp()),
                                "logit_bias": float(self.model.bias.detach()),
                                "samples_per_s": samples / elapsed,
                                "data_wait": waited / elapsed,
                                "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                            },
                            "train",
                        )
                        torch.cuda.reset_peak_memory_stats()
                        losses, norms, waited, paused, samples = [], [], 0.0, 0.0, 0
                        window = time.perf_counter()
                    began = time.perf_counter()
                    if config.eval.every_steps and self.step % config.eval.every_steps == 0:
                        metrics = self.evaluate()
                        self.log(metrics, "eval")
                        logger.info("Step %d: %s", self.step, ", ".join(f"{k} {v:.4f}" for k, v in metrics.items()))
                    every = config.checkpoint
                    due = bool(every.every_steps) and self.step % every.every_steps == 0
                    due = due or (every.every_minutes > 0 and time.monotonic() - last_checkpoint >= 60 * every.every_minutes)
                    if due and self.step < config.run.steps and not self.stop:
                        self.save()
                        last_checkpoint = time.monotonic()
                    paused += time.perf_counter() - began
                    if self.step >= config.run.steps or self.stop:
                        break
            self.save()
        finally:
            signal.signal(signal.SIGINT, previous)
            progress.close()
            self.writer.close()

    def _interrupt(self, *_):
        logger.warning("Stopping after this step (Ctrl-C again to abort without a checkpoint)")
        self.stop = True
        signal.signal(signal.SIGINT, signal.default_int_handler)


def _rows(batch: dict, start: int, stop: int) -> dict:
    return {k: batch[k][start:stop] for k in ("patches", "patch_coord", "patch_valid")}


def _recall(similarity: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """Share of rows with a positive among their ``k`` most similar columns."""
    top = similarity.topk(min(k, similarity.shape[1]), dim=1).indices
    return float(labels.gather(1, top).any(1).float().mean())


# -- command line ---------------------------------------------------------------------------------------------------


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--verbose", "-v", is_flag=True, help="Debug logging.")
def main(verbose: bool) -> None:
    """Fine-tune SigLIP2 so400m on an image pack."""
    setup_logging(verbose, ("picurate", "clip_training"))


@main.command()
@click.argument("run_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Settings (TOML) for a new run; for a continued run, settings that may change (workers, intervals, steps).",
)
def run(run_dir: Path, config_path: Path | None):
    """Train in RUN_DIR: start a run with --config, or continue the one there from its newest checkpoint."""
    stored = run_dir / "config.toml"
    if not stored.exists() and config_path is None:
        raise click.ClickException(f"{run_dir} holds no run; start one with --config")
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = open(run_dir / ".lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise click.ClickException(f"another clip-train run is training in {run_dir}") from None
    if stored.exists():
        config = Config.load(stored)
        if config_path is not None:
            new = Config.load(config_path)
            old_flat, new_flat = config.flat(), new.flat()
            missing = object()  # a key only one side has (model.lora set on one side only)
            changed = {k for k in old_flat.keys() | new_flat.keys() if old_flat.get(k, missing) != new_flat.get(k, missing)}
            refused = changed - config.mutable()
            if refused:
                raise click.ClickException(
                    "A continued run cannot change " + ", ".join(sorted(refused)) + "; start a new run (model.init_from)"
                )
            if changed:
                logger.info("Changed for this run: %s", ", ".join(f"{k} = {new_flat[k]!r}" for k in sorted(changed)))
                config = new
    else:
        config = Config.load(config_path)
    for partial in (run_dir / CHECKPOINTS).glob("*.partial"):
        logger.info("Removing %s, left by an interrupted checkpoint", partial.name)
        shutil.rmtree(partial)
    pack = Pack(config.data.pack)
    binding, binding_path = pack_binding(pack), run_dir / "binding.json"
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise click.ClickException(f"{config.data.pack} is not the pack this run was started on")
    found = checkpoints(run_dir)
    trainer = Trainer(run_dir, config, pack, found[-1] if found else None)
    # The run's files are written once it has started.
    if config_path is not None:
        shutil.copyfile(config_path, stored)
    if not binding_path.exists():
        binding_path.write_text(json.dumps(binding, indent=2) + "\n")
        trainer.writer.add_text("config", "```\n" + stored.read_text() + "\n```")
    trainer.run()
    click.echo(f"Stopped at step {trainer.step} of {config.run.steps}", err=True)


@main.command()
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--name", required=True, help="The model's name, which databases made with it are bound to.")
@click.option("--step", type=int, default=None, help="Checkpoint to export (default: the newest).")
def export(run_dir: Path, output: Path, name: str, step: int | None):
    """Write the evaluation weights of a run's checkpoint to OUTPUT (safetensors, FP32, LoRA merged): ``image.*``,
    for the search route ``text.*``, and ``logit_scale``/``logit_bias``."""
    config = Config.load(run_dir / "config.toml")
    found = {int(p.name.split("-")[1]): p for p in checkpoints(run_dir)}
    if not found:
        raise click.ClickException(f"{run_dir} has no checkpoint")
    path = found[max(found)] if step is None else found.get(step)
    if path is None:
        raise click.ClickException(f"{run_dir} has no checkpoint of step {step}")
    model = build_model(config)
    tensors = load_file(path / "trainable.safetensors")
    model.load_state_dict(tensors, strict=False)
    missing = set(tensors) - {n for n, _ in model.named_parameters()}
    if missing:
        raise click.ClickException(f"{path} does not match the run's configuration")
    merge_lora(model)
    out = {f"image.{k}": v for k, v in model.image.state_dict().items()}
    if model.text is not None:
        out |= {f"text.{k}": v for k, v in model.text.state_dict().items()}
    out["logit_scale"], out["logit_bias"] = model.scale.detach(), model.bias.detach()
    metadata = {
        "name": name,
        "image_tower": IMAGE_TOWER,
        "seq_len": str(config.data.seq_len),
        "route": config.run.route,
        "step": path.name.split("-")[1],
    }
    partial = output.with_name(output.name + ".partial")
    save_file({k: v.float().contiguous() for k, v in out.items()}, partial, metadata=metadata)
    _fsync(partial)
    os.replace(partial, output)
    click.echo(f"{output}: {name}, {path.name}", err=True)


if __name__ == "__main__":
    main()
