"""KWS transformer with an optional positional-encoding stage (for kws_position_matched).

The stock ``oscillator_attention.transformer.KWSTransformer`` has NO positional encoding and is
left unchanged. This module reproduces it exactly and adds a ``pe`` switch:

    pe='none'        -> architecturally identical to stock KWSTransformer
                        (same module-creation order => seed-identical weights).
    pe='sinusoidal'  -> stock SinusoidalPositionalEncoding after the input proj.
    pe='learned_abs' -> learned absolute position embedding table (T, d_model).

Everything else (proj, GELU FFN with mult=4, dropout=0.1, mean-pool head, and the attention
modules) is the stock model unchanged; ``training/selftest.py`` asserts the pe='none' parity.
"""
from training import paths
paths.ensure_paths()

import torch
import torch.nn as nn

from oscillator_attention.attention import LoheAttention, SoftmaxAttention
from oscillator_attention.transformer import (
    _KWSTransformerLayer, SinusoidalPositionalEncoding,
)


class KWSTransformerPE(nn.Module):
    """KWSTransformer + optional positional encoding.

    Args mirror stock KWSTransformer; adds ``pe`` in {'none','sinusoidal','learned_abs'}.
    """

    def __init__(self, n_feats: int = 40, d_model: int = 32, n_heads: int = 2,
                 n_layers: int = 1, n_classes: int = 10, T: int = 49,
                 attn_type: str = 'lohe', p: int = 1, d_osc: int = 2,
                 N: int = 500, dt: float = 0.05, dropout: float = 0.1,
                 pe: str = 'none'):
        super().__init__()
        self.pe = pe
        self.T = T
        self.proj = nn.Linear(n_feats, d_model)
        self.dropout = nn.Dropout(dropout)

        # Positional-encoding params are created ONLY when needed, so that
        # pe='none' preserves the stock module-creation/RNG order exactly.
        if pe == 'sinusoidal':
            self.pos_enc = SinusoidalPositionalEncoding(d_model, T, dropout)
        elif pe == 'learned_abs':
            self.pos_emb = nn.Parameter(torch.zeros(1, T, d_model))
            nn.init.normal_(self.pos_emb, std=0.02)
        elif pe != 'none':
            raise ValueError(f"unknown pe={pe!r}")

        def _make_attn():
            if attn_type == 'lohe':
                return LoheAttention(d_model, n_heads, d_model // n_heads,
                                     d_osc, T, p=p, causal=False)
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
        h = self.proj(x)
        if self.pe == 'none':
            h = self.dropout(h)
        elif self.pe == 'sinusoidal':
            h = self.pos_enc(h)                       # adds pe + dropout
        elif self.pe == 'learned_abs':
            h = h + self.pos_emb[:, :x.size(1)]
            h = self.dropout(h)
        for layer in self.layers:
            h = layer(h)
        return self.head(h.mean(dim=1))
