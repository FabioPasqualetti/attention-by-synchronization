"""Models that plug LoheAttentionSigma (selectable coupling nonlinearity) into the
stock KWS and TinyStories architectures, for lm_coupling_function.

- build_kws_sigma: stock KWSTransformer structure (no PE, mean-pool) with
  LoheAttentionSigma attention. sigma='softplus' reproduces stock KWS B1.
- LoheLMSigma: stock LoheLanguageTransformer structure (sinusoidal PE, causal)
  with LoheAttentionSigma. sigma='softplus' reproduces stock TinyStories LoheLM.

Both re-use stock building blocks (_KWSTransformerLayer, SinusoidalPositionalEncoding).
"""
from training import paths
paths.ensure_paths()

import torch
import torch.nn as nn
from typing import Optional

from oscillator_attention.transformer import (
    _KWSTransformerLayer, SinusoidalPositionalEncoding,
)
from .coupling_variants import LoheAttentionSigma


def build_kws_sigma(sigma, n_feats=40, d_model=32, n_heads=2, n_layers=1,
                    n_classes=10, T=49, d_osc=2, p=1, dropout=0.1):
    """KWSTransformer (pe='none') with LoheAttentionSigma coupling."""
    class _KWS(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(n_feats, d_model)
            self.dropout = nn.Dropout(dropout)
            attns = [LoheAttentionSigma(d_model, n_heads, d_model // n_heads,
                                        d_osc, T, p=p, causal=False, sigma=sigma)
                     for _ in range(n_layers)]
            self.layers = nn.ModuleList(
                [_KWSTransformerLayer(d_model, a, dropout=dropout) for a in attns])
            self.head = nn.Linear(d_model, n_classes)

        def forward(self, x):
            h = self.dropout(self.proj(x))
            for layer in self.layers:
                h = layer(h)
            return self.head(h.mean(dim=1))
    return _KWS()


class _LoheLayerSigma(nn.Module):
    def __init__(self, d_model, n_heads, d_head, d_ff, dropout, d_osc,
                 max_seq_len, p, sigma):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = LoheAttentionSigma(d_model, n_heads, d_head, d_osc,
                                       max_seq_len, p=p, causal=True, sigma=sigma)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout))

    def forward(self, x, padding_mask=None):
        out, _ = self.attn(self.norm1(x), padding_mask=padding_mask)
        x = x + out
        return x + self.ffn(self.norm2(x))


class LoheLMSigma(nn.Module):
    """LoheLanguageTransformer with LoheAttentionSigma coupling."""

    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2, d_ff=512,
                 max_seq_len=128, dropout=0.1, d_osc=8, p=1, sigma='softplus'):
        super().__init__()
        assert d_model % n_heads == 0
        d_head = d_model // n_heads
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)
        self.layers = nn.ModuleList([
            _LoheLayerSigma(d_model, n_heads, d_head, d_ff, dropout, d_osc,
                            max_seq_len, p, sigma) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, padding_mask: Optional[torch.Tensor] = None):
        h = self.pos_enc(self.embedding(x))
        for layer in self.layers:
            h = layer(h, padding_mask=padding_mask)
        return self.head(self.norm(h))
