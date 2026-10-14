"""The SigLIP2 so400m text tower, without transformers.

It embeds text into the space of the image tower picurate-dedupe stores (timm's
``naflexvit_so400m_patch16_siglip.v2_webli``, the same checkpoint as ``google/siglip2-so400m-patch16-naflex``). The
network is a pre-LayerNorm transformer over a learned position embedding of 64, without an attention mask: texts are
lowercased (the checkpoint's ``tokenizer.json`` does not do it, and cased queries rank images badly), tokenized with
its Gemma tokenizer, ended with ``<eos>`` and padded with ``<pad>`` to exactly 64 tokens, and the hidden state of the
last position (usually padding) goes through a linear head, as the model was trained. Only the ``text_model.*``
tensors are read from the checkpoint (about 0.7 B parameters, 1.4 GiB in fp16).

The text tower was trained against image embeddings whose position embedding is resampled bilinearly; timm resamples
it bicubically, so cosine similarities against the stored embeddings differ from the official ones by up to about
0.014 (on SigLIP2's scale, where matches score 0.10-0.18 and unrelated images 0.02-0.06).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .siglip_image import DEFAULT_REPO, DEFAULT_REVISION, IMAGE_TOWER  # noqa: F401 - the checkpoint both towers share


@dataclass(frozen=True, slots=True)
class TextConfig:
    vocab_size: int = 256000
    width: int = 1152
    layers: int = 27
    heads: int = 16
    mlp: int = 4304
    context: int = 64
    projection: int = 1152
    eps: float = 1e-6

    @classmethod
    def from_json(cls, config: dict) -> TextConfig:
        text, default = config.get("text_config", {}), cls()
        return cls(
            vocab_size=text.get("vocab_size", default.vocab_size),
            width=text.get("hidden_size", default.width),
            layers=text.get("num_hidden_layers", default.layers),
            heads=text.get("num_attention_heads", default.heads),
            mlp=text.get("intermediate_size", default.mlp),
            context=text.get("max_position_embeddings", default.context),
            projection=text.get("projection_size", text.get("hidden_size", default.projection)),
            eps=text.get("layer_norm_eps", default.eps),
        )


class Attention(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        q, k, v = (
            proj(x).view(b, n, self.heads, c // self.heads).transpose(1, 2) for proj in (self.q_proj, self.k_proj, self.v_proj)
        )
        return self.out_proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, c))


class Mlp(nn.Module):
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(width, hidden)
        self.fc2 = nn.Linear(hidden, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, cfg: TextConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(cfg.width, eps=cfg.eps)
        self.self_attn = Attention(cfg.width, cfg.heads)
        self.layer_norm2 = nn.LayerNorm(cfg.width, eps=cfg.eps)
        self.mlp = Mlp(cfg.width, cfg.mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))


class TextTower(nn.Module):
    """``(B, context)`` token IDs -> ``(B, projection)`` embeddings (unnormalized). Parameter names follow the
    checkpoint's ``text_model.*`` tensors, so its state dict loads as is."""

    def __init__(self, cfg: TextConfig = TextConfig()):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.width)
        self.position_embedding = nn.Embedding(cfg.context, cfg.width)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.final_layer_norm = nn.LayerNorm(cfg.width, eps=cfg.eps)
        self.head = nn.Linear(cfg.width, cfg.projection)
        self.grad_checkpointing = False  # recompute each layer's activations in the backward pass (training)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(ids) + self.position_embedding.weight[: ids.shape[1]]
        for layer in self.layers:
            if self.grad_checkpointing and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return self.head(self.final_layer_norm(x)[:, -1])


# Checkpoint tensor names -> TextTower parameter names.
_RENAMES = (("text_model.embeddings.", ""), ("text_model.encoder.layers.", "layers."), ("text_model.", ""))


class Tokenizer:
    """The checkpoint's tokenizer as SigLIP2 was trained with it: lowercase, ``<eos>`` appended, truncated and padded
    with ``<pad>`` to exactly ``context`` tokens."""

    def __init__(self, tokenizer_json: str, context: int = 64):
        from tokenizers import Tokenizer as _Tokenizer

        self.tokenizer = _Tokenizer.from_file(tokenizer_json)
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        self.context = context
        self.eos = self.tokenizer.token_to_id("<eos>")
        self.pad = self.tokenizer.token_to_id("<pad>")

    def __call__(self, texts: list[str]) -> torch.Tensor:
        out = torch.full((len(texts), self.context), self.pad, dtype=torch.long)
        for i, text in enumerate(texts):
            ids = self.tokenizer.encode(text.lower(), add_special_tokens=False).ids[: self.context - 1] + [self.eos]
            out[i, : len(ids)] = torch.tensor(ids)
        return out


@dataclass
class TextEncoder:
    """A loaded text tower with its tokenizer and SigLIP's similarity scale: ``sigmoid(cos * exp(logit_scale) +
    logit_bias)`` is the model's match probability."""

    tower: TextTower
    tokenizer: Tokenizer
    logit_scale: float
    logit_bias: float

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        """``(N, D)`` float32 embeddings on the CPU, unnormalized like the stored image embeddings."""
        param = next(self.tower.parameters())
        return self.tower(self.tokenizer(texts).to(param.device)).float().cpu()


def load_text_encoder(
    repo: str = DEFAULT_REPO, revision: str | None = None, *, device: str = "cpu", dtype: torch.dtype = torch.float32
) -> TextEncoder:
    """Download (or find in the Hugging Face cache) the checkpoint and build the text tower from it."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    if revision is None and repo == DEFAULT_REPO:
        revision = DEFAULT_REVISION
    files = {
        name: hf_hub_download(repo, name, revision=revision) for name in ("config.json", "tokenizer.json", "model.safetensors")
    }
    with open(files["config.json"], encoding="utf-8") as stream:
        cfg = TextConfig.from_json(json.load(stream))
    state = {}
    with safe_open(files["model.safetensors"], "pt") as checkpoint:
        for key in checkpoint.keys():
            if key.startswith("text_model."):
                name = key
                for old, new in _RENAMES:
                    if name.startswith(old):
                        name = new + name[len(old) :]
                        break
                state[name] = checkpoint.get_tensor(key)
        scale, bias = (float(checkpoint.get_tensor(k).float().reshape(-1)[0]) for k in ("logit_scale", "logit_bias"))
    with torch.device("meta"):
        tower = TextTower(cfg)
    tower.load_state_dict(state, strict=True, assign=True)
    tower = tower.to(device=device, dtype=dtype).eval()
    return TextEncoder(tower, Tokenizer(files["tokenizer.json"], cfg.context), scale, bias)
