#!/usr/bin/env python3
"""Training-step benchmark for the two SigLIP2 fine-tuning routes (see docs/training-clip.md), on synthetic data.

It finds the largest batch whose full step (forward, sigmoid loss, backward, AdamW step) fits the GPU, or a memory
limit standing in for a smaller GPU, and measures steps per second there:

* ``dedupe``: the image tower alone. Each sample is two views of one image, encoded in one pass; the loss is SigLIP's
  sigmoid loss between the two views (positives on the diagonal, every other image a negative).
* ``search``: the image tower and the text tower of :mod:`picurate.common.siglip_text`, with the sigmoid image-text
  loss and a learned scale and bias.

The towers are the real architectures (``picurate.common.siglip_image`` and ``siglip_text``) with
random weights: speed and memory do not depend on the values, and nothing is downloaded. Every image is a full
sequence of ``--seq-len`` patches, the worst case of the NaFlex budget; texts are 64 tokens, as SigLIP2 always pads.
Trained parameters stay FP32 (the master weights) under autocast, so the optimizer state is what AdamW keeps in the
real run; frozen ones are stored in the autocast precision, since nothing updates them.
"""

from __future__ import annotations

import gc
import math
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import click
import torch
import torch.nn.functional as F
from torch import nn

from picurate.common.siglip_image import ATTENTIONS, ImageTower
from picurate.common.siglip_text import TextConfig, TextTower

PRECISIONS = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def sigmoid_loss(a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """SigLIP's pairwise sigmoid loss: row i of ``a`` and ``b`` match, every other pair does not."""
    a, b = F.normalize(a.float(), dim=-1), F.normalize(b.float(), dim=-1)
    logits = a @ b.T * scale.exp() + bias
    labels = 2 * torch.eye(len(a), device=a.device) - 1
    return -F.logsigmoid(labels * logits).sum() / len(a)


class Bench(nn.Module):
    def __init__(self, route: str, train: str, freeze_blocks: int, checkpointing: bool, attention: str | None = None):
        super().__init__()
        self.route = route
        self.image = ImageTower(attention=attention)
        self.text = TextTower(TextConfig()) if route == "search" else None
        # SigLIP2's initial values; learned in both routes.
        self.scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.bias = nn.Parameter(torch.tensor(-10.0))
        if self.text is not None and train == "text":
            self.image.requires_grad_(False)
        if self.text is not None and train == "image":
            self.text.requires_grad_(False)
        for block in self.image.blocks[:freeze_blocks]:
            block.requires_grad_(False)
        if checkpointing:
            self.image.set_grad_checkpointing(True)
            if self.text is not None:
                self.text.grad_checkpointing = True

    def embed(self, images: dict, texts: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """The two sides the loss pairs up: both views (dedupe), or images and texts (search)."""
        embeddings = self.image(images)
        if self.route == "dedupe":
            a, b = embeddings.chunk(2)
            return a, b
        return embeddings, self.text(texts)

    def loss(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return sigmoid_loss(a, b, self.scale, self.bias)

    def forward(self, images: dict, texts: torch.Tensor | None) -> torch.Tensor:
        return self.loss(*self.embed(images, texts))


def synthetic_batch(route: str, batch: int, seq_len: int, device: str) -> tuple[dict, torch.Tensor | None]:
    """A batch of full patch sequences (two views per sample for ``dedupe``) and, for ``search``, token IDs. On the
    CPU it is pinned, as a data loader would deliver it."""
    n = 2 * batch if route == "dedupe" else batch
    side = math.isqrt(seq_len)
    grid = torch.stack(torch.meshgrid(torch.arange(side), torch.arange(seq_len // side + 1), indexing="ij"), -1)
    coords = grid.reshape(-1, 2)[:seq_len].to(device).expand(n, -1, -1)
    images = {
        "patches": torch.randn(n, seq_len, 16 * 16 * 3, device=device),
        "patch_coord": coords,
        "patch_valid": torch.ones(n, seq_len, dtype=torch.bool, device=device),
    }
    texts = torch.randint(0, TextConfig().vocab_size, (batch, TextConfig().context), device=device) if route == "search" else None
    if device == "cpu":
        images = {key: value.contiguous().pin_memory() for key, value in images.items()}
        texts = None if texts is None else texts.pin_memory()
    return images, texts


def sub_batch(
    route: str, images: dict, texts: torch.Tensor | None, start: int, stop: int, device: str
) -> tuple[dict, torch.Tensor | None]:
    """Samples ``start:stop`` of a batch, in the same layout (for ``dedupe``: both views of each), on ``device``."""
    if route == "dedupe":
        n = len(images["patches"]) // 2
        rows = torch.cat([torch.arange(start, stop), torch.arange(n + start, n + stop)]).to(images["patches"].device)
        return {key: value[rows].to(device, non_blocking=True) for key, value in images.items()}, None
    images = {key: value[start:stop].to(device, non_blocking=True) for key, value in images.items()}
    return images, texts[start:stop].to(device, non_blocking=True)


class Runner:
    def __init__(self, model: Bench, precision: str, seq_len: int, compile: bool, device: str, micro: int | None):
        self.model = model
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self.params, lr=1e-5, weight_decay=0.0, fused=True)
        self.dtype = PRECISIONS[precision]
        self.scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
        self.seq_len = seq_len
        self.device = device
        self.micro = micro
        if compile:
            model.image = torch.compile(model.image, dynamic=True)
            if model.text is not None:
                model.text = torch.compile(model.text, dynamic=True)

    def autocast(self):
        return torch.autocast("cuda", dtype=self.dtype, enabled=self.dtype != torch.float32)

    def step(self, images, texts) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        if self.micro is None:
            with self.autocast():
                loss = self.model(images, texts)
            self.scaler.scale(loss).backward()
        else:
            self.grad_cache_backward(images, texts)
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def grad_cache_backward(self, images, texts) -> None:
        """GradCache: the loss of the whole batch with the memory of one micro-batch. Embed every micro-batch without
        gradients, take the loss and its gradient with respect to the embeddings, then embed each micro-batch again
        with gradients and back-propagate its slice of that gradient. Exact as long as the towers are deterministic
        (no dropout here; a real run with dropout must replay each micro-batch's RNG state)."""
        route = self.model.route
        batch = len(images["patches"]) // (2 if route == "dedupe" else 1)
        spans = [(start, min(batch, start + self.micro)) for start in range(0, batch, self.micro)]
        with torch.no_grad(), self.autocast():
            parts = [self.model.embed(*sub_batch(route, images, texts, *span, self.device)) for span in spans]
        a = torch.cat([part[0] for part in parts]).float().requires_grad_()
        b = torch.cat([part[1] for part in parts]).float().requires_grad_()
        del parts
        self.scaler.scale(self.model.loss(a, b)).backward()
        for start, stop in spans:
            with self.autocast():
                pa, pb = self.model.embed(*sub_batch(route, images, texts, start, stop, self.device))
            pairs = [(t.float(), g[start:stop]) for t, g in ((pa, a.grad), (pb, b.grad)) if t.requires_grad]
            torch.autograd.backward([t for t, _ in pairs], [g for _, g in pairs])

    def measure(self, batch: int, warmup: int, steps: int) -> dict | None:
        """Time ``steps`` steps at ``batch`` after ``warmup``; None if it runs out of memory."""
        torch.cuda.reset_peak_memory_stats()
        result = None
        try:
            # GradCache keeps the batch on the host and moves one micro-batch at a time, like a real run.
            images, texts = synthetic_batch(self.model.route, batch, self.seq_len, "cpu" if self.micro else self.device)
            for _ in range(warmup):
                self.step(images, texts)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(steps):
                self.step(images, texts)
            torch.cuda.synchronize()
            seconds = (time.perf_counter() - start) / steps
            result = {"batch": batch, "ms": seconds * 1000, "it/s": 1 / seconds, "samples/s": batch / seconds}
            result["peak GiB"] = torch.cuda.max_memory_allocated() / 2**30
        except torch.OutOfMemoryError:
            pass
        # Outside the except block, so the traceback no longer holds the step's tensors.
        images = texts = None
        self.optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        return result


def report(result: dict | None, batch: int) -> None:
    if result is None:
        click.echo(f"  batch {batch:4d}: out of memory")
    else:
        click.echo(
            f"  batch {batch:4d}: {result['ms']:7.0f} ms/step  {result['it/s']:6.2f} it/s  {result['samples/s']:7.1f} samples/s"
            f"  peak {result['peak GiB']:5.2f} GiB"
        )


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--route", type=click.Choice(["dedupe", "search"]), default="dedupe", show_default=True, help="What is trained.")
@click.option(
    "--train",
    type=click.Choice(["both", "image", "text"]),
    default="both",
    show_default=True,
    help="search: which towers are trained (the other is frozen).",
)
@click.option(
    "--freeze-blocks",
    type=click.IntRange(0, 27),
    default=0,
    show_default=True,
    help="Freeze the lowest N of the image tower's 27 blocks.",
)
@click.option("--checkpointing/--no-checkpointing", default=True, show_default=True, help="Activation checkpointing per block.")
@click.option("--precision", type=click.Choice(list(PRECISIONS)), default="bf16", show_default=True, help="Autocast precision.")
@click.option("--frozen-fp32", is_flag=True, help="Keep frozen parameters in FP32 instead of the autocast precision.")
@click.option(
    "--seq-len", type=click.IntRange(16), default=1024, show_default=True, help="Image patches per image (the NaFlex budget)."
)
@click.option("--compile", is_flag=True, help="torch.compile the step's forward (dynamic shapes).")
@click.option(
    "--attention",
    type=click.Choice(ATTENTIONS),
    default=None,
    help="Attention layout of the image tower; default: flash with flash-attn installed, else sdpa.",
)
@click.option(
    "--memory",
    type=float,
    default=None,
    help="Limit PyTorch's allocations to this many GiB, to stand in for a smaller GPU (its CUDA context and other programs come on top: use ~7 for an 8 GiB card).",
)
@click.option(
    "--grad-cache",
    "micro",
    type=click.IntRange(1),
    default=None,
    help="GradCache with micro-batches of this many samples: the loss sees the whole --batch-size, memory follows the micro-batch (one extra forward pass).",
)
@click.option(
    "--batch-size",
    "-b",
    type=click.IntRange(1),
    default=None,
    help="Measure this batch only, instead of searching the largest one.",
)
@click.option("--max-batch", type=click.IntRange(1), default=4096, show_default=True, help="Stop the search here.")
@click.option("--steps", type=click.IntRange(1), default=10, show_default=True, help="Timed steps at the final batch.")
@click.option("--warmup", type=click.IntRange(1), default=3, show_default=True, help="Untimed steps before timing.")
def main(
    route,
    train,
    freeze_blocks,
    checkpointing,
    precision,
    frozen_fp32,
    seq_len,
    compile,
    attention,
    memory,
    micro,
    batch_size,
    max_batch,
    steps,
    warmup,
):
    """Find how large a batch a fine-tuning step of SigLIP2 so400m fits on this GPU, and how fast it runs."""
    if micro is not None and batch_size is None:
        raise click.ClickException("--grad-cache needs --batch-size: its memory hardly grows with the batch, only its time")
    if not torch.cuda.is_available():
        raise click.ClickException("CUDA is not available")
    device = "cuda"
    total = torch.cuda.get_device_properties(0).total_memory
    if memory is not None:
        torch.cuda.set_per_process_memory_fraction(min(1.0, memory * 2**30 / total))
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    model = Bench(route, train, freeze_blocks, checkpointing, attention).to(device).train()
    if not frozen_fp32:
        for p in model.parameters():
            if not p.requires_grad:
                p.data = p.data.to(PRECISIONS[precision])
    runner = Runner(model, precision, seq_len, compile, device, micro)
    trainable = sum(p.numel() for p in runner.params)
    click.echo(
        f"GPU: {torch.cuda.get_device_name(0)}, {total / 2**30:.1f} GiB" + (f", limited to {memory:g} GiB" if memory else "")
    )
    click.echo(
        f"Route {route}"
        + (f" (training {train})" if route == "search" else "")
        + f": {trainable / 1e6:.0f}M of {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters trained,"
        f" {freeze_blocks} image blocks frozen, checkpointing {'on' if checkpointing else 'off'}, {precision} autocast,"
        f" frozen ones in {'fp32' if frozen_fp32 else precision}, {seq_len} patches per image, {model.image.attention} attention"
        f"{', compiled' if compile else ''}" + (f", GradCache with micro-batches of {micro}" if micro else "")
    )
    click.echo(
        f"Weights, gradients and AdamW state: {(sum(p.numel() * p.element_size() for p in model.parameters()) + trainable * 12) / 2**30:.2f} GiB"
    )

    if batch_size is not None:
        report(runner.measure(batch_size, warmup, steps), batch_size)
        return
    # Double until a batch does not fit, then bisect; each probe runs one step.
    fits, fails, batch = 0, None, 1
    while batch <= max_batch:
        result = runner.measure(batch, 1, 1)
        report(result, batch)
        if result is None:
            fails = batch
            break
        fits, batch = batch, batch * 2
    if fits == 0:
        raise click.ClickException("Not even a batch of 1 fits")
    while fails is not None and fails - fits > 1:
        batch = (fits + fails) // 2
        result = runner.measure(batch, 1, 1)
        report(result, batch)
        fits, fails = (batch, fails) if result is not None else (fits, batch)
    click.echo(f"Largest batch: {fits}; timing {steps} steps")
    # A probe that just fit can fail over several steps (fragmentation); step down until it holds.
    while (result := runner.measure(fits, warmup, steps)) is None and fits > 1:
        report(result, fits)
        fits -= 1
    report(result, fits)


if __name__ == "__main__":
    main()
