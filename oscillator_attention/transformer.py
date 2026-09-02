"""
Transformer wrappers for Fixed-Query Oscillator Attention.

KWSTransformer      — keyword spotting (classification, mean-pool)
LoheLanguageTransformer — autoregressive LM using LoheAttention
SVATransformer      — subject-verb agreement (binary classification at verb)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import KuramotoAttention, LoheAttention, SoftmaxAttention


# ── Positional encoding ───────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


# ── KWS transformer ───────────────────────────────────────────────────────────

class _KWSTransformerLayer(nn.Module):
    def __init__(self, d_model: int, attn: nn.Module, ffn_mult: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = attn
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model),
            nn.Dropout(dropout),
        )
        self._last_attn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a_out, attn = self.attn(self.norm1(x))
        self._last_attn = attn.detach()
        x = x + a_out
        x = x + self.ffn(self.norm2(x))
        return x


class KWSTransformer(nn.Module):
    """
    Keyword spotting transformer (encoder + mean-pool classification).

    Architecture: Linear(n_feats→d_model) → n_layers×[PreNorm Attn + FFN] → mean-pool → Linear

    Args:
        n_feats:   input feature dimension (40 for log-mel)
        d_model:   model dimension (default 32)
        n_heads:   attention heads (default 2)
        n_layers:  transformer layers (default 1)
        n_classes: number of output classes (default 10)
        T:         sequence length (default 49 for 1-second 20ms-shift log-mel)
        attn_type: 'kuramoto' or 'softmax'
        p:         Kuramoto readout exponent (default 1); ignored for softmax
        N:         ODE steps (default 500); ignored for softmax
        dt:        ODE step size (default 0.05); ignored for softmax
        dropout:   dropout rate (default 0.1)
    """

    def __init__(self, n_feats: int = 40, d_model: int = 32, n_heads: int = 2,
                 n_layers: int = 1, n_classes: int = 10, T: int = 49,
                 attn_type: str = 'lohe', p: int = 1, d_osc: int = 2,
                 N: int = 500, dt: float = 0.05, dropout: float = 0.1):
        super().__init__()
        self.proj    = nn.Linear(n_feats, d_model)
        self.dropout = nn.Dropout(dropout)

        def _make_attn():
            if attn_type == 'lohe':
                return LoheAttention(d_model, n_heads, d_model // n_heads,
                                     d_osc, T, p=p, causal=False)
            elif attn_type == 'kuramoto':
                return KuramotoAttention(d_model, n_heads, T, N, dt, p)
            elif attn_type == 'softmax':
                return SoftmaxAttention(d_model, n_heads)
            else:
                raise ValueError(f"Unknown attn_type: {attn_type!r}")

        self.layers = nn.ModuleList([
            _KWSTransformerLayer(d_model, _make_attn(), dropout=dropout)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, n_feats) → logits (B, n_classes)"""
        h = self.dropout(self.proj(x))
        for layer in self.layers:
            h = layer(h)
        return self.head(h.mean(dim=1))

    def get_attention_weights(self) -> torch.Tensor:
        return self.layers[0]._last_attn


# ── Lohe language transformer ─────────────────────────────────────────────────

class _LoheTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, d_ff: int,
                 dropout: float, d_osc: int, max_seq_len: int, p: int = 1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = LoheAttention(d_model, n_heads, d_head, d_osc,
                                   max_seq_len, p=p, causal=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out, _ = self.attn(self.norm1(x), padding_mask=padding_mask)
        x = x + out
        x = x + self.ffn(self.norm2(x))
        return x


class LoheLanguageTransformer(nn.Module):
    """
    Autoregressive language model using LoheAttention.

    Architecture matches scaling-law experiments:
      d_model=128, n_heads=4, n_layers=2, d_ff=512.
    Only d_osc varies (2, 8, 32).

    Args:
        vocab_size:  vocabulary size
        d_model:     model dimension (default 128)
        n_heads:     attention heads (default 4)
        n_layers:    transformer layers (default 2)
        d_ff:        FFN width (default 512)
        max_seq_len: maximum sequence length (default 50)
        dropout:     dropout rate (default 0.1)
        d_osc:       oscillator sphere dimension (default 8)
    """

    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, d_ff: int = 512, max_seq_len: int = 50,
                 dropout: float = 0.1, d_osc: int = 8, p: int = 1):
        super().__init__()
        assert d_model % n_heads == 0
        d_head = d_model // n_heads

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc   = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)
        self.layers    = nn.ModuleList([
            _LoheTransformerLayer(d_model, n_heads, d_head, d_ff, dropout,
                                  d_osc, max_seq_len, p=p)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x:            (B, T) token indices
        padding_mask: (B, T) bool, True = padding position
        Returns:      (B, T, vocab_size) logits
        """
        h = self.pos_enc(self.embedding(x))
        for layer in self.layers:
            h = layer(h, padding_mask=padding_mask)
        return self.head(self.norm(h))


# ── SVA transformer ───────────────────────────────────────────────────────────

class _SVATransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float,
                 d_osc: int, max_seq_len: int, attn_type: str):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        if attn_type == "kuramoto":
            d_head = d_model // n_heads
            self.attn = LoheAttention(d_model, n_heads, d_head, d_osc,
                                      max_seq_len, p=1, causal=False)
        elif attn_type == "softmax":
            self.attn = SoftmaxAttention(d_model, n_heads)
        else:
            raise ValueError(f"Unknown attn_type: {attn_type!r}")
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self._is_lohe = isinstance(self.attn, LoheAttention)

    def forward(self, x: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        normed = self.norm1(x)
        if self._is_lohe:
            out, _ = self.attn(normed, padding_mask=padding_mask)
        else:
            out, _ = self.attn(normed, padding_mask=padding_mask, causal=False)
        x = x + out
        x = x + self.ffn(self.norm2(x))
        return x


class SVATransformer(nn.Module):
    """
    Subject-verb agreement transformer (binary classification at verb position).

    Args:
        vocab_size:  vocabulary size
        d_model:     model dimension (default 64)
        n_heads:     attention heads (default 2)
        n_layers:    transformer layers (default 2)
        d_ff:        FFN width (default 256)
        max_seq_len: maximum sequence length (default 50)
        dropout:     dropout rate (default 0.1)
        attn_type:   'kuramoto' or 'softmax'
        d_osc:       oscillator dimension (for Kuramoto; default 2)
    """

    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 2,
                 n_layers: int = 2, d_ff: int = 256, max_seq_len: int = 50,
                 dropout: float = 0.1, attn_type: str = "kuramoto",
                 d_osc: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc   = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)
        self.layers    = nn.ModuleList([
            _SVATransformerLayer(d_model, n_heads, d_ff, dropout, d_osc,
                                 max_seq_len, attn_type)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None,
                verb_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x:            (B, T) token indices
        padding_mask: (B, T) bool, True = padding
        verb_idx:     (B,) int, position of the verb token
        Returns:      (B,) logits (positive = plural)
        """
        h = self.pos_enc(self.embedding(x))
        for layer in self.layers:
            h = layer(h, padding_mask=padding_mask)
        h = self.norm(h)
        verb_h = h[range(x.shape[0]), verb_idx]
        return self.head(verb_h).squeeze(-1)
