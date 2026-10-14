"""The SigLIP2 so400m NaFlex image tower, without timm or transformers.

The network is timm's ``naflexvit_so400m_patch16_siglip.v2_webli`` (the ``vision_model`` of
``google/siglip2-so400m-patch16-naflex``) with the same parameter names, so checkpoints trained on either load into
the other: a linear patch embedding over 16x16 RGB patches, a learned 16x16 position embedding resampled to each
image's patch grid (bicubically with antialiasing, as timm does; the official code resamples bilinearly), 27
pre-LayerNorm blocks and a one-query attention pooling head. Images arrive as the deduper preprocesses them: a
sequence of at most ``seq_len`` patches with their grid coordinates and a validity mask, padded to the same length.

Attention runs in one of two layouts. ``flash`` (flash-attn's variable-length kernels, when the package is installed)
packs the valid patches of the batch into one sequence and attends within each image, so padding costs nothing and
no mask is built. ``sdpa`` keeps the padded batch and masks padding in ``F.scaled_dot_product_attention``, which then
cannot use its flash kernel. Both compute the same function; the two differ by floating-point noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

DEFAULT_REPO = "google/siglip2-so400m-patch16-naflex"
DEFAULT_REVISION = "cc24074f717b612951c2dead130904ab9b65a81e"
IMAGE_TOWER = "naflexvit_so400m_patch16_siglip.v2_webli"  # the model's name: databases are bound to it
PATCH = 16
DEFAULT_SEQ_LEN = 1024

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:  # pragma: no cover - depends on the environment
    flash_attn_varlen_func = None

ATTENTIONS = ("flash", "sdpa")


def default_attention() -> str:
    """The fastest attention layout available: ``flash`` with flash-attn installed, else ``sdpa``."""
    return "flash" if flash_attn_varlen_func is not None else "sdpa"


def image_size_for_seq(height: int, width: int, seq_len: int = DEFAULT_SEQ_LEN, patch: int = PATCH) -> tuple[int, int]:
    """``(height, width)`` to resize an image to for at most ``seq_len`` patches at its aspect ratio: the largest scale
    whose dimensions, rounded up to whole patches, fit. The same binary search as timm's
    ``get_image_size_for_seq`` (Apache-2.0), so packs and databases made with it stay identical."""

    def target(ratio: float) -> tuple[int, int]:
        h = int(max(patch * math.ceil(height * ratio / patch), patch))
        w = int(max(patch * math.ceil(width * ratio / patch), patch))
        return h, w

    low, high, eps = 1e-6, 100.0, 1e-5
    while high - low >= eps:
        mid = (low + high) / 2.0
        h, w = target(mid)
        if (h // patch) * (w // patch) <= seq_len:
            low = mid
        else:
            high = mid
    if low <= eps:
        raise ValueError("image too large")
    if low >= 100.0:
        raise ValueError("image too small")
    return target(low)


def to_patches(image: PIL.Image.Image, seq_len: int = DEFAULT_SEQ_LEN) -> tuple[np.ndarray, np.ndarray]:
    """``(N, 16*16*3)`` uint8 patches and ``(N, 2)`` (row, column) coordinates of ``image``: resized to
    :func:`image_size_for_seq` bicubically, patches row by row, each patch's pixels row by row, RGB. This is the
    deduper's preprocessing (before normalization), the same as timm's NaFlex evaluation transform."""
    height, width = image_size_for_seq(image.height, image.width, seq_len)
    pixels = np.asarray(image.convert("RGB").resize((width, height), PIL.Image.Resampling.BICUBIC))
    rows, columns = height // PATCH, width // PATCH
    patches = pixels.reshape(rows, PATCH, columns, PATCH, 3).transpose(0, 2, 1, 3, 4).reshape(rows * columns, -1)
    coords = np.stack(np.meshgrid(np.arange(rows), np.arange(columns), indexing="ij"), -1).reshape(-1, 2)
    return patches, coords


@dataclass(frozen=True)
class ImageConfig:
    width: int = 1152
    layers: int = 27
    heads: int = 16
    mlp: int = 4304
    patch: int = PATCH
    pos_grid: int = 16  # the learned position embedding is pos_grid x pos_grid
    eps: float = 1e-6

    @classmethod
    def from_json(cls, config: dict) -> "ImageConfig":
        v = config["vision_config"]
        return cls(
            width=v["hidden_size"],
            layers=v["num_hidden_layers"],
            heads=v["num_attention_heads"],
            mlp=v["intermediate_size"],
            patch=v.get("patch_size", PATCH),
            pos_grid=int(math.isqrt(v.get("num_patches", 256))),
            eps=v.get("layer_norm_eps", 1e-6),
        )


class _Layout:
    """Where the tokens of a batch are and how they attend. ``sdpa``: ``x`` is ``(B, N, C)`` with a ``(B, N)``
    validity mask. ``flash``: ``x`` is ``(T, C)``, the valid tokens of all images in order, with flash-attn's
    cumulative sequence lengths."""

    def __init__(self, kind: str, valid: torch.Tensor):
        self.kind = kind
        self.valid = valid
        self.batch = valid.shape[0]
        if kind == "flash":
            lengths = valid.sum(1, dtype=torch.int32)
            self.cu_seqlens = F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))
            self.max_seqlen = int(lengths.max())
            self.cu_one = torch.arange(self.batch + 1, device=valid.device, dtype=torch.int32)
        else:
            self.mask = valid[:, None, None, :]  # (B, 1, 1, N): keys that may be attended

    def pack(self, x: torch.Tensor) -> torch.Tensor:
        return x[self.valid] if self.kind == "flash" else x

    def attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Self-attention within each image. ``flash``: ``(T, H, D)`` each; ``sdpa``: ``(B, N, H, D)``."""
        if self.kind == "flash":
            return flash_attn_varlen_func(q, k, v, self.cu_seqlens, self.cu_seqlens, self.max_seqlen, self.max_seqlen)
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=self.mask)
        return out.transpose(1, 2)

    def pool(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """One query per image over that image's tokens: ``q`` is ``(B, H, D)``, the result ``(B, H, D)``."""
        if self.kind == "flash":
            return flash_attn_varlen_func(q, k, v, self.cu_one, self.cu_seqlens, 1, self.max_seqlen)
        out = F.scaled_dot_product_attention(q[:, :, None], k.transpose(1, 2), v.transpose(1, 2), attn_mask=self.mask)
        return out[:, :, 0]


class Attention(nn.Module):
    def __init__(self, cfg: ImageConfig):
        super().__init__()
        self.heads, self.head_dim = cfg.heads, cfg.width // cfg.heads
        self.qkv = nn.Linear(cfg.width, 3 * cfg.width)
        self.proj = nn.Linear(cfg.width, cfg.width)

    def forward(self, x: torch.Tensor, layout: _Layout) -> torch.Tensor:
        q, k, v = self.qkv(x).unflatten(-1, (3, self.heads, self.head_dim)).unbind(-3)
        return self.proj(layout.attend(q, k, v).flatten(-2))


class Mlp(nn.Module):
    def __init__(self, cfg: ImageConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.width, cfg.mlp)
        self.fc2 = nn.Linear(cfg.mlp, cfg.width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, cfg: ImageConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.width, eps=cfg.eps)
        self.attn = Attention(cfg)
        self.norm2 = nn.LayerNorm(cfg.width, eps=cfg.eps)
        self.mlp = Mlp(cfg)

    def forward(self, x: torch.Tensor, layout: _Layout) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), layout)
        return x + self.mlp(self.norm2(x))


class Embeds(nn.Module):
    """Patch projection plus the learned position embedding, resampled once per distinct patch grid in the batch."""

    def __init__(self, cfg: ImageConfig):
        super().__init__()
        self.proj = nn.Linear(cfg.patch * cfg.patch * 3, cfg.width)
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.pos_grid, cfg.pos_grid, cfg.width))

    def forward(self, patches: torch.Tensor, coord: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        x = self.proj(patches)
        grids = (coord.masked_fill(~valid[..., None], 0).amax(1) + 1).tolist()  # (rows, columns) per image
        groups: dict[tuple[int, int], list[int]] = {}
        for i, grid in enumerate(grids):
            groups.setdefault(tuple(grid), []).append(i)
        table = self.pos_embed.permute(0, 3, 1, 2)  # (1, C, rows, columns)
        for (rows, columns), images in groups.items():
            if (rows, columns) == tuple(table.shape[2:]):
                resampled = table
            else:
                resampled = F.interpolate(table.float(), size=(rows, columns), mode="bicubic", align_corners=False, antialias=True)
            flat = resampled.flatten(2).transpose(1, 2)[0].to(x.dtype)  # (rows * columns, C)
            index = torch.as_tensor(images, device=x.device)
            keep = valid[index]
            where = (coord[index, :, 0] * columns + coord[index, :, 1]).masked_fill(~keep, 0)
            x = x.index_add(0, index, flat[where] * keep[:, :, None].to(x.dtype))
        return x


class AttentionPool(nn.Module):
    def __init__(self, cfg: ImageConfig):
        super().__init__()
        self.heads, self.head_dim = cfg.heads, cfg.width // cfg.heads
        self.latent = nn.Parameter(torch.zeros(1, 1, cfg.width))
        self.q = nn.Linear(cfg.width, cfg.width)
        self.kv = nn.Linear(cfg.width, 2 * cfg.width)
        self.proj = nn.Linear(cfg.width, cfg.width)
        self.norm = nn.LayerNorm(cfg.width, eps=cfg.eps)
        self.mlp = Mlp(cfg)

    def forward(self, x: torch.Tensor, layout: _Layout) -> torch.Tensor:
        q = self.q(self.latent[0].expand(layout.batch, -1)).unflatten(-1, (self.heads, self.head_dim))
        k, v = self.kv(x).unflatten(-1, (2, self.heads, self.head_dim)).unbind(-3)
        x = self.proj(layout.pool(q, k, v).flatten(-2))
        return x + self.mlp(self.norm(x))


class ImageTower(nn.Module):
    """``forward`` takes the deduper's batch: ``patches`` ``(B, N, 768)`` normalized to [-1, 1], ``patch_coord``
    ``(B, N, 2)`` (row, column) and ``patch_valid`` ``(B, N)``; it returns ``(B, width)`` unnormalized embeddings."""

    def __init__(self, cfg: ImageConfig = ImageConfig(), attention: str | None = None):
        super().__init__()
        self.cfg = cfg
        self.attention = attention or default_attention()
        if self.attention not in ATTENTIONS:
            raise ValueError(f"attention must be one of {ATTENTIONS}")
        if self.attention == "flash" and flash_attn_varlen_func is None:
            raise ValueError("flash attention needs the flash-attn package")
        self.embeds = Embeds(cfg)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.norm = nn.LayerNorm(cfg.width, eps=cfg.eps)
        self.attn_pool = AttentionPool(cfg)
        self.grad_checkpointing = False

    def no_weight_decay(self) -> set[str]:
        return {"embeds.pos_embed", "attn_pool.latent"}

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        valid = batch["patch_valid"]
        layout = _Layout(self.attention, valid)
        x = layout.pack(self.embeds(batch["patches"], batch["patch_coord"], valid))
        for block in self.blocks:
            if self.grad_checkpointing and self.training:
                x = checkpoint(block, x, layout, use_reentrant=False)
            else:
                x = block(x, layout)
        return self.attn_pool(self.norm(x), layout)


def _convert(state: dict[str, torch.Tensor], cfg: ImageConfig) -> dict[str, torch.Tensor]:
    """Hugging Face ``vision_model.*`` tensors in this module's (timm's) names."""
    w = cfg.width
    out = {
        "embeds.proj.weight": state["embeddings.patch_embedding.weight"],
        "embeds.proj.bias": state["embeddings.patch_embedding.bias"],
        "embeds.pos_embed": state["embeddings.position_embedding.weight"].reshape(1, cfg.pos_grid, cfg.pos_grid, w),
        "norm.weight": state["post_layernorm.weight"],
        "norm.bias": state["post_layernorm.bias"],
        "attn_pool.latent": state["head.probe"],
        "attn_pool.q.weight": state["head.attention.in_proj_weight"][:w],
        "attn_pool.q.bias": state["head.attention.in_proj_bias"][:w],
        "attn_pool.kv.weight": state["head.attention.in_proj_weight"][w:],
        "attn_pool.kv.bias": state["head.attention.in_proj_bias"][w:],
        "attn_pool.proj.weight": state["head.attention.out_proj.weight"],
        "attn_pool.proj.bias": state["head.attention.out_proj.bias"],
        "attn_pool.norm.weight": state["head.layernorm.weight"],
        "attn_pool.norm.bias": state["head.layernorm.bias"],
        "attn_pool.mlp.fc1.weight": state["head.mlp.fc1.weight"],
        "attn_pool.mlp.fc1.bias": state["head.mlp.fc1.bias"],
        "attn_pool.mlp.fc2.weight": state["head.mlp.fc2.weight"],
        "attn_pool.mlp.fc2.bias": state["head.mlp.fc2.bias"],
    }
    for i in range(cfg.layers):
        src, dst = f"encoder.layers.{i}.", f"blocks.{i}."
        for kind in ("weight", "bias"):
            out[f"{dst}attn.qkv.{kind}"] = torch.cat([state[f"{src}self_attn.{p}_proj.{kind}"] for p in ("q", "k", "v")])
            out[f"{dst}attn.proj.{kind}"] = state[f"{src}self_attn.out_proj.{kind}"]
            out[f"{dst}norm1.{kind}"] = state[f"{src}layer_norm1.{kind}"]
            out[f"{dst}norm2.{kind}"] = state[f"{src}layer_norm2.{kind}"]
            out[f"{dst}mlp.fc1.{kind}"] = state[f"{src}mlp.fc1.{kind}"]
            out[f"{dst}mlp.fc2.{kind}"] = state[f"{src}mlp.fc2.{kind}"]
    return out


def load_image_tower(
    repo: str = DEFAULT_REPO,
    revision: str | None = None,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    attention: str | None = None,
) -> ImageTower:
    """Download (or find in the Hugging Face cache) the checkpoint and build the image tower from its
    ``vision_model``, in evaluation mode."""
    import json

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    if revision is None and repo == DEFAULT_REPO:
        revision = DEFAULT_REVISION
    config_path = hf_hub_download(repo, "config.json", revision=revision)
    weights_path = hf_hub_download(repo, "model.safetensors", revision=revision)
    with open(config_path, encoding="utf-8") as stream:
        cfg = ImageConfig.from_json(json.load(stream))
    state = {}
    with safe_open(weights_path, "pt") as checkpoint_file:
        for key in checkpoint_file.keys():
            if key.startswith("vision_model."):
                state[key[len("vision_model.") :]] = checkpoint_file.get_tensor(key)
    with torch.device("meta"):
        tower = ImageTower(cfg, attention)
    tower.load_state_dict(_convert(state, cfg), strict=True, assign=True)
    return tower.to(device=device, dtype=dtype).eval()
