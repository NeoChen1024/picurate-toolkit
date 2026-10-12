"""The PixAI tagger network (a SAM3-style ViTDet backbone with an attention-pooling multi-label head), written for
inference only, plus its image preprocessing and checkpoint loading.

The checkpoint ships its own ``trust_remote_code`` module; this is an independent rewrite of the same architecture
restricted to what the published config uses, so no remote code runs and the model can be cast and compiled freely.
Two differences from the remote module are deliberate:

* RoPE is applied with real cos/sin tables kept in fp32 instead of a complex64 buffer. ``Module.to(torch.float16)``
  silently casts a complex buffer to a real one (dropping the imaginary part), which makes every fp16/bf16 output of
  the remote module garbage; the real form also compiles without graph breaks.
* The model is built on the meta device and the checkpoint assigned into it, skipping a pointless random init of
  486M parameters.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as vF
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

default_repo_id = "pixai-labs/pixai-tagger-v1.0"
# The architecture below is written against this snapshot's config; pinning it keeps a silent upstream change from
# loading weights into a network they no longer fit.
default_revision = "9fe10addf9326e292da8a85a98ea74cd91b41771"


# -- network ---------------------------------------------------------------------------------------------------------


def _axial_rope_angles(head_dim: int, side: int, scale: float, theta: float = 10000.0) -> torch.Tensor:
    """``(side*side, head_dim/2)`` rotation angles for 2D axial RoPE over a row-major ``side x side`` token grid: the
    first half of the channel pairs rotate with the column index, the second half with the row index."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 4, device="cpu")[: head_dim // 4].float() / head_dim))
    t = torch.arange(side * side, dtype=torch.float32, device="cpu")
    tx = (t % side) * scale
    ty = torch.div(t, side, rounding_mode="floor") * scale
    return torch.cat([torch.outer(tx, freqs), torch.outer(ty, freqs)], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent channel pairs of ``x`` (..., L, D) by the per-token angles; computed in fp32."""
    xf = x.float().unflatten(-1, (-1, 2))
    a, b = xf[..., 0], xf[..., 1]
    return torch.stack([a * cos - b * sin, a * sin + b * cos], dim=-1).flatten(-2).type_as(x)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, side: int, rope_scale: float):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        angles = _axial_rope_angles(dim // num_heads, side, rope_scale)
        # Not in the checkpoint; kept fp32 whatever the model dtype (see ``load_model``).
        self.register_buffer("rope_cos", angles.cos(), persistent=False)
        self.register_buffer("rope_sin", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        L = H * W
        q, k, v = self.qkv(x).reshape(B, L, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).unbind(0)
        q = _apply_rope(q, self.rope_cos, self.rope_sin)
        k = _apply_rope(k, self.rope_cos, self.rope_sin)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, H, W, C))


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    """Pre-norm transformer block; ``window > 0`` restricts attention to non-overlapping ``window x window`` tiles."""

    def __init__(self, dim: int, num_heads: int, mlp_hidden: int, grid: int, window: int, rope_pt: int):
        super().__init__()
        self.window = window
        side = window or grid
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = Attention(dim, num_heads, side, rope_pt / side)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.mlp = Mlp(dim, mlp_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        if self.window:
            B, H, W, C = x.shape
            w = self.window
            x = x.view(B, H // w, w, W // w, w, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, w, w, C)
            x = self.attn(x)
            x = x.view(B, H // w, W // w, w, w, C).permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)
        else:
            x = self.attn(x)
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class AttentionPool(nn.Module):
    """One learned query attending over all patch tokens."""

    def __init__(self, dim: int, num_heads: int, num_patches: int):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(dim)
        self.q = nn.Parameter(torch.empty(dim))
        self.kv = nn.Linear(dim, dim * 2)
        self.pos_embed = nn.Parameter(torch.empty(num_patches, dim))
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        x = self.norm(x) + self.pos_embed
        k, v = self.kv(x).view(B, N, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4).unbind(0)
        q = self.q.view(1, self.num_heads, 1, -1).expand(B, -1, -1, -1)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.reshape(B, D))


@dataclass(frozen=True, slots=True)
class TaggerConfig:
    img_size: int
    patch_size: int
    embed_dim: int
    depth: int
    num_heads: int
    mlp_hidden: int
    window_size: int
    global_blocks: tuple[int, ...]
    pretrain_grid: int
    rope_pt: int
    num_classes: int

    @classmethod
    def from_json(cls, cfg: dict) -> TaggerConfig:
        # Fail loudly on any switch this rewrite does not implement instead of computing something else.
        expected = {
            "use_rope": True,
            "use_interp_rope": True,
            "use_abs_pos": True,
            "tile_abs_pos": True,
            "pretrain_use_cls_token": True,
            "retain_cls_token": False,
            "rel_pos_blocks": [],
            "init_values": None,
            "ln_pre": True,
            "ln_post": False,
            "bias_patch_embed": False,
            "qkv_bias": True,
            "act_layer": "GELU",
            "norm_layer": "LayerNorm",
            "in_chans": 3,
        }
        wrong = {k: cfg.get(k) for k, v in expected.items() if cfg.get(k) != v}
        if wrong:
            raise ValueError(f"Unsupported tagger config (this implementation expects {expected}): {wrong}")
        grid = cfg["img_size"] // cfg["patch_size"]
        window = cfg["window_size"]
        if grid % window:
            raise ValueError(f"Token grid {grid} is not a multiple of the window size {window}")
        return cls(
            img_size=cfg["img_size"],
            patch_size=cfg["patch_size"],
            embed_dim=cfg["embed_dim"],
            depth=cfg["depth"],
            num_heads=cfg["num_heads"],
            mlp_hidden=int(cfg["embed_dim"] * cfg["mlp_ratio"]),
            window_size=window,
            global_blocks=tuple(cfg["global_att_blocks"]),
            pretrain_grid=cfg["pretrain_img_size"] // cfg["patch_size"],
            # The remote code passes the window size as the RoPE "pretrain" size when the config leaves it unset.
            rope_pt=cfg.get("rope_pt_size") or window,
            num_classes=cfg["num_classes"],
        )


class ViTDetTagger(nn.Module):
    def __init__(self, cfg: TaggerConfig):
        super().__init__()
        self.cfg = cfg
        dim = cfg.embed_dim
        grid = cfg.img_size // cfg.patch_size
        self.grid = grid
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, dim, cfg.patch_size, cfg.patch_size, bias=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.pretrain_grid**2 + 1, dim))  # + unused cls slot
        self.ln_pre = nn.LayerNorm(dim, eps=1e-5)
        self.blocks = nn.ModuleList(
            Block(dim, cfg.num_heads, cfg.mlp_hidden, grid, 0 if i in cfg.global_blocks else cfg.window_size, cfg.rope_pt)
            for i in range(cfg.depth)
        )
        self.head_pool = AttentionPool(dim, cfg.num_heads, grid * grid)
        self.head = nn.Linear(dim, cfg.num_classes)

    def _tiled_pos_embed(self) -> torch.Tensor:
        """The pretrain-size position grid repeated (not interpolated) over the token grid, ``(1, H, W, C)``."""
        p = self.cfg.pretrain_grid
        pos = self.pos_embed[:, 1:].reshape(1, p, p, -1)
        reps = self.grid // p + 1
        return pos.tile(1, reps, reps, 1)[:, : self.grid, : self.grid]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, 3, S, S)`` normalized images -> ``(B, num_classes)`` logits."""
        x = self.patch_embed.proj(x).permute(0, 2, 3, 1)
        x = self.ln_pre(x + self._tiled_pos_embed())
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.head_pool(x.flatten(1, 2)))


# -- preprocessing ---------------------------------------------------------------------------------------------------


class TaggerPreprocess:
    """PIL -> ``(3, S, S)`` array in ``dtype``: transparency flattened onto white, aspect-preserving resize so the long
    side is ``S``, centered on black padding, normalized to [-1, 1]. Matches the checkpoint's own image processor.
    Picklable for decoder workers."""

    def __init__(self, size: int, dtype: np.dtype):
        self.size = size
        self.dtype = dtype

    def __call__(self, img: PIL.Image.Image) -> np.ndarray:
        if img.mode != "RGB":
            rgba = img.convert("RGBA")
            canvas = PIL.Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            canvas.alpha_composite(rgba)
            img = canvas.convert("RGB")
        x = vF.to_tensor(img)
        s = self.size
        h, w = x.shape[-2:]
        if h != s or w != s:
            r = min(s / h, s / w)
            nh, nw = int(h * r), int(w * r)
            x = vF.resize(x, [nh, nw])
            left, top = (s - nw) // 2, (s - nh) // 2
            x = vF.pad(x, [left, top, s - nw - left, s - nh - top], 0)
        return x.sub_(0.5).div_(0.5).numpy().astype(self.dtype, copy=False)


# -- loading ---------------------------------------------------------------------------------------------------------


@dataclass(slots=True)
class LoadedTagger:
    model: ViTDetTagger
    config: dict  # the raw config.json: tag names, category split, per-category thresholds


def _effective_revision(model: str, revision: str | None) -> str | None:
    return default_revision if revision is None and model == default_repo_id else revision


def model_identity(model: str, revision: str | None = None) -> str:
    """What a tag database is bound to: ``repo@revision`` (``main`` when unpinned), or a local directory's path."""
    if os.path.isdir(model):
        return os.path.abspath(model)
    return f"{model}@{_effective_revision(model, revision) or 'main'}"


def _resolve_files(model: str, revision: str | None) -> tuple[str, str]:
    """``(config.json, model.safetensors)`` paths from a local directory or a Hub repo (downloaded/cached)."""
    if os.path.isdir(model):
        return os.path.join(model, "config.json"), os.path.join(model, "model.safetensors")
    from huggingface_hub import hf_hub_download

    revision = _effective_revision(model, revision)
    return (
        hf_hub_download(model, "config.json", revision=revision),
        hf_hub_download(model, "model.safetensors", revision=revision),
    )


def load_model(
    model: str = default_repo_id, *, revision: str | None = None, device: str = "cpu", dtype: torch.dtype = torch.float32
) -> LoadedTagger:
    config_path, weights_path = _resolve_files(model, revision)
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    cfg = TaggerConfig.from_json(config)
    with torch.device("meta"):
        net = ViTDetTagger(cfg)
    # The RoPE tables are computed on the CPU explicitly, so they are real tensors even under the meta context.
    net.load_state_dict(load_file(weights_path), strict=True, assign=True)
    net = net.to(device=device, dtype=dtype).eval()
    for m in net.modules():
        if isinstance(m, Attention):
            m.rope_cos = m.rope_cos.float()
            m.rope_sin = m.rope_sin.float()
    if len(config["tags"]) != cfg.num_classes or sum(n for _, n in config["tags_split"]) != cfg.num_classes:
        raise ValueError("Tag list / category split do not match the number of classes")
    return LoadedTagger(net, config)
