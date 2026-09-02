"""Coupling-nonlinearity variants of LoheAttention (for lm_coupling_function).

Exact copy of ``oscillator_attention.attention.LoheAttention`` (attention.py:135-219 at
commit 2be382d) with a SINGLE change: the coupling nonlinearity applied to
``raw = q^T k / sqrt(d_head)`` is selectable.

    sigma='softplus' : F.softplus(raw)          (reference, = stock LoheAttention)
    sigma='relu_eps' : F.relu(raw) + 1e-3       (no exponential; positive)
    sigma='elu1'     : F.elu(raw) + 1.0         (positive; exp only for raw<0)

All other computation (anchors, analytic fixed point, readout, causal/padding masks,
value aggregation) is byte-identical to the source. Verified against stock in
training/selftest.py (the softplus path reproduces LoheAttention exactly).
"""
import math
from typing import Optional

from training import paths
paths.ensure_paths()

import torch
import torch.nn as nn
import torch.nn.functional as F


def _coupling(sigma: str, raw: torch.Tensor) -> torch.Tensor:
    if sigma == 'softplus':
        return F.softplus(raw)
    if sigma == 'relu_eps':
        return F.relu(raw) + 1e-3
    if sigma == 'elu1':
        return F.elu(raw) + 1.0
    raise ValueError(f"unknown sigma={sigma!r}")


class LoheAttentionSigma(nn.Module):
    """LoheAttention with a selectable coupling nonlinearity ``sigma``."""

    def __init__(self, d_model: int, n_heads: int, d_head: int,
                 d_osc: int, max_seq_len: int, p: int = 1,
                 causal: bool = True, sigma: str = 'softplus'):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_osc = d_osc
        self.p = p
        self.causal = causal
        self.max_seq_len = max_seq_len
        self.sigma = sigma

        self.W_q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_o = nn.Linear(n_heads * d_head, d_model, bias=True)

        anchors = torch.randn(n_heads, max_seq_len, d_osc)
        self.anchors = nn.Parameter(F.normalize(anchors, dim=-1))

        if causal:
            self.register_buffer("_causal_mask",
                                 torch.tril(torch.ones(max_seq_len, max_seq_len)),
                                 persistent=False)

    def forward(self, x: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None):
        B, T, _ = x.shape
        H, D_h, D_osc = self.n_heads, self.d_head, self.d_osc

        q = self.W_q(x).view(B, T, H, D_h).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, D_h).transpose(1, 2)
        raw = torch.einsum('bhid,bhjd->bhij', q, k) / math.sqrt(D_h)
        W = _coupling(self.sigma, raw)  # (B, H, T, T), all positive

        if self.causal:
            W = W * self._causal_mask[:T, :T]
        if padding_mask is not None:
            col_mask = (~padding_mask).float().view(B, 1, 1, T)
            W = W * col_mask

        anc = self.anchors[:, :T, :]
        h = torch.einsum('bhij,hjd->bhid', W, anc)
        x_star = F.normalize(h, dim=-1, eps=1e-8)

        cos_sim = torch.einsum('bhid,hjd->bhij', x_star, anc)
        attn = (1.0 + cos_sim).clamp(min=0.0)
        if self.p != 1:
            attn = attn ** self.p
        if self.causal:
            attn = attn * self._causal_mask[:T, :T]
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        v = self.W_v(x).view(B, T, H, D_h).transpose(1, 2)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, H * D_h)
        out = self.W_o(out)
        return out, attn
