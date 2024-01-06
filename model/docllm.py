from typing import Sequence, TypeAlias

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from jaxtyping import Float
import einops

from .config import DocLLMConfig

Shape: TypeAlias = Sequence[int] | int


class RotaryEmbeddings(nn.Module):
    """Rotary embeddings for DocLLM."""

    def __init__(
        self, head_dim: int, max_seq_len: int, theta_base: int = 10000
    ) -> None:
        super().__init__()

        dim = head_dim
        dim_thetas = 1.0 / (
            theta_base ** (torch.arange(0, dim, 2, dtype=torch.float) / head_dim)
        )
        token_dim_thetas = torch.outer(torch.arange(1, max_seq_len + 1), dim_thetas)
        sin = torch.sin(token_dim_thetas)
        cos = torch.cos(token_dim_thetas)
        self.register_buffer("sin", sin)
        self.register_buffer("cos", cos)

    def _get_sin_cos_cache(
        self, x: Float[Tensor, "b h s d"]
    ) -> tuple[Float[Tensor, "s d"], Float[Tensor, "s d"]]:
        sin = self.sin[: x.shape[-2], :].to(x.device)
        cos = self.cos[: x.shape[-2], :].to(x.device)
        return sin, cos

    def _rotate(
        self,
        x: Float[Tensor, "b h s d"],
        y: Float[Tensor, "b h s d"],
        sin: Float[Tensor, "s d"],
        cos: Float[Tensor, "s d"],
    ) -> tuple[Float[Tensor, "b h s d"], Float[Tensor, "b h s d"]]:
        return x * cos - y * sin, x * sin + y * cos

    def forward(self, x: Float[Tensor, "b s h d"]) -> Float[Tensor, "b h s d"]:
        x1, x2 = x[..., 0::2], x[..., 1::2]
        sin, cos = self._get_sin_cos_cache(x)
        rx1, rx2 = self._rotate(x1, x2, sin, cos)
        rotated = torch.stack([rx1, rx2], dim=-1)
        rotated = einops.rearrange(rotated, "b h s d two -> b h s (d two)")
        return rotated


class RMSNorm(nn.Module):
    """Simple RMSNorm implementation."""

    def __init__(self, dim: Shape, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Float[Tensor, "b s hd"]):
        non_centered_variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(non_centered_variance + self.eps)
        return x * self.weight


class FFNLayer(nn.Module):
    """Feed forward network for DocLLM."""

    def __init__(self, hidden_size: int, expanded_size: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_size, expanded_size, bias=False)
        self.up_linear = nn.Linear(hidden_size, expanded_size, bias=False)
        self.down_linear = nn.Linear(expanded_size, hidden_size, bias=False)
        self.activation = nn.SiLU()

    def forward(self, x: Float[Tensor, "b s hd"]) -> Float[Tensor, "b s hd"]:
        return self.down_linear(self.activation(self.gate(x)) * self.up_linear(x))


class GroupedQueryAttention(nn.Module):  # For now, just vanilla attention
    def __init__(self, config: DocLLMConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.max_seq_len = config.max_seq_len
        self.dropout = config.dropout
        self.rotary_emb = RotaryEmbeddings(self.head_dim, self.max_seq_len)
        self.qw = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.kw = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.vw = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.out = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def _attention(
        self,
        q: Float[Tensor, "b s hd"],
        k: Float[Tensor, "b s hd"],
        v: Float[Tensor, "b s hd"],
        mask: Float[Tensor, "b s"] | None,
    ) -> Float[Tensor, "b s hd"]:
        q = einops.rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = einops.rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = einops.rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)
        q = self.rotary_emb(q)
        k = self.rotary_emb(k)
        attn_weights = torch.einsum("bhsd, bhSd -> bhsS", q, k)
        attn_weights = attn_weights * torch.rsqrt(torch.tensor(self.head_dim).float())
        if mask:
            attn_weights = attn_weights.masked_fill(mask, float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # TODO: revisit
        x = torch.einsum("bhsS,bhsd->bhsd", attn_weights, v)
        x = einops.rearrange(x, "b h s d -> b s (h d)")
        return x

    def forward(
        self, x: Float[Tensor, "b s hd"], mask: Float[Tensor, "b s"] | None
    ) -> Float[Tensor, "b s hd"]:
        q = self.qw(x)
        k = self.kw(x)
        v = self.vw(x)
        x = self._attention(q, k, v, mask=mask)
        return self.out(x)


class DecoderBlock(nn.Module):
    """Decoder block for DocLLM."""

    def __init__(self, config: DocLLMConfig) -> None:
        super().__init__()
        self.config = config
        self.attn = GroupedQueryAttention(config)
        self.ffn = FFNLayer(config.hidden_size, config.ffn_expanded_size)
        self.pre_attn_norm = RMSNorm(config.hidden_size)
        self.pre_ffn_norm = RMSNorm(config.hidden_size)

    def forward(
        self, x: Float[Tensor, "b s hd"], mask: Float[Tensor, "b s"] | None
    ) -> Float[Tensor, "b s hd"]:
        residual = x
        x = self.pre_attn_norm(x)
        x = self.attn(x, mask=mask) + residual
        residual = x
        x = self.pre_ffn_norm(x)
        x = self.ffn(x) + residual
        return x


class TransformerDecoder(nn.Module):
    """Decoder for DocLLM."""

    def __init__(self, config: DocLLMConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.num_layers)]
        )
        self.pre_out_norm = RMSNorm(config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, x: Float[Tensor, "b s hd"]) -> Float[Tensor, "b s hd"]:
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask=None)
        x = self.pre_out_norm(x)
        x = self.out_proj(x)
        x = F.softmax(x, dim=-1)
        return x


class DocLLM:
    """DocLLM model."""

    def __init__(self, config: DocLLMConfig) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = ...
        self.decoder = TransformerDecoder(config)

    def forward(self, x: Float[Tensor, "b s"]):
        return self.decoder(x)