"""
Appendix C ablation attention modules — analytic Lohe-framework variants.

Both classes are drop-in replacements for LoheAttention with minimal
modifications. They use the analytic fixed point z* = h/||h|| (same as
LoheAttention) and differ only in how W or the anchors are formed.

LoheRelativeAnchorAttention (Appendix C, ablation L3)
    Adds a learned rotation delta(i-j) to each anchor before the fixed-point
    computation, exposing relative token distance to the attention mechanism.
    Restricted to d_osc=2 (rotation in 2D is a single angle).

    NOTE on sign convention: the correct relative distance for causal LM is
    i-j (how many steps back the key is from the query). An earlier ODE-based
    prototype used j-i, which was always ≤ 0 for valid (past) keys and clamped
    to zero, making delta ineffective. This implementation uses the correct
    i-j formula.

LoheSharpAttention (Appendix C, ablation L4)
    Adds a learned per-head per-query-position coupling scale beta_{h,i}.
    W_ij = softplus(beta_{h,i} * q_i^T k_j / sqrt(d_head)).
    Works for any d_osc.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoheRelativeAnchorAttention(nn.Module):
    """
    LoheAttention at d_osc=2 with learned per-query-position anchor rotation.

    Anchor for token j at query i: r_{i,j} = R(delta[i-j]) @ anchor_j
    where R(.) is 2D rotation and delta is a learned (max_seq_len,) vector
    initialised to zero (recovers standard LoheAttention at init).

    Fixed point:  z_i* = normalize(sum_j W_ij * r_{i,j})
    Readout:      a_{ij} = (1 + z_i*^T r_{i,j})^p / norm

    Args: identical to LoheAttention except d_osc is fixed at 2.
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int,
                 max_seq_len: int, p: int = 1, causal: bool = True):
        super().__init__()
        self.n_heads    = n_heads
        self.d_head     = d_head
        self.d_osc      = 2
        self.p          = p
        self.causal     = causal
        self.max_seq_len = max_seq_len

        self.W_q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_o = nn.Linear(n_heads * d_head, d_model, bias=True)

        # Anchor vectors on S^1: (n_heads, max_seq_len, 2), unit-normalised at init
        anchors = torch.randn(n_heads, max_seq_len, 2)
        self.anchors = nn.Parameter(F.normalize(anchors, dim=-1))

        # Relative-position rotation angle: delta[k] applied when key is k steps
        # behind query (i-j = k). delta[0] = 0 by init → no rotation for
        # same-position or future (clamped) keys.
        self.delta = nn.Parameter(torch.zeros(max_seq_len))

        if causal:
            self.register_buffer("_causal_mask",
                                 torch.tril(torch.ones(max_seq_len, max_seq_len)),
                                 persistent=False)

    def forward(self, x: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None):
        """
        x:            (B, T, d_model)
        padding_mask: (B, T) bool, True = padding
        Returns: (out, attn)
        """
        B, T, _ = x.shape
        H, D_h  = self.n_heads, self.d_head
        device  = x.device

        # ── Coupling weights W_ij ─────────────────────────────────────────
        q = self.W_q(x).view(B, T, H, D_h).transpose(1, 2)  # (B, H, T, D_h)
        k = self.W_k(x).view(B, T, H, D_h).transpose(1, 2)
        raw = torch.einsum('bhid,bhjd->bhij', q, k) / math.sqrt(D_h)
        W = F.softplus(raw)  # (B, H, T_q, T_k)

        if self.causal:
            W = W * self._causal_mask[:T, :T]
        if padding_mask is not None:
            W = W * (~padding_mask).float().view(B, 1, 1, T)

        # ── Position-dependent anchor rotation ────────────────────────────
        # rel[i, j] = i - j  (positive = key is behind query; correct formula)
        pos     = torch.arange(T, device=device)
        rel     = (pos.unsqueeze(1) - pos.unsqueeze(0)).clamp(0, T - 1)  # (T_q, T_k)
        angle   = self.delta[:T][rel]          # (T_q, T_k)
        cos_a   = angle.cos()                  # (T_q, T_k)
        sin_a   = angle.sin()                  # (T_q, T_k)

        anc = self.anchors[:, :T, :]           # (H, T_k, 2)
        a0, a1 = anc[:, :, 0], anc[:, :, 1]   # (H, T_k) each

        # r_{i,j} = R(delta[i-j]) @ anchor_j
        # r0[h,i,j] = a0[h,j]*cos(δ[i,j]) - a1[h,j]*sin(δ[i,j])
        r0 = (a0.unsqueeze(1) * cos_a.unsqueeze(0)
              - a1.unsqueeze(1) * sin_a.unsqueeze(0))  # (H, T_q, T_k)
        r1 = (a0.unsqueeze(1) * sin_a.unsqueeze(0)
              + a1.unsqueeze(1) * cos_a.unsqueeze(0))  # (H, T_q, T_k)
        rot_anc = torch.stack([r0, r1], dim=-1)        # (H, T_q, T_k, 2)

        # ── Analytic fixed point ──────────────────────────────────────────
        h = torch.einsum('bhij,hijd->bhid', W, rot_anc)   # (B, H, T_q, 2)
        z_star = F.normalize(h, dim=-1, eps=1e-8)          # (B, H, T_q, 2)

        # ── Readout ───────────────────────────────────────────────────────
        cos_sim = torch.einsum('bhid,hijd->bhij', z_star, rot_anc)  # (B, H, T_q, T_k)
        attn = (1.0 + cos_sim).clamp(min=0.0)
        if self.p != 1:
            attn = attn ** self.p
        if self.causal:
            attn = attn * self._causal_mask[:T, :T]
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # ── Value aggregation ─────────────────────────────────────────────
        v = self.W_v(x).view(B, T, H, D_h).transpose(1, 2)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, H * D_h)
        return self.W_o(out), attn


class LoheSharpAttention(nn.Module):
    """
    LoheAttention with learned per-head per-query-position coupling scale.

    W_ij = softplus(beta_{h,i} * q_i^T k_j / sqrt(d_head))

    beta_{h,i} is a (n_heads, max_seq_len) parameter initialised to 1.0,
    constrained positive via abs().clamp(min=0.1).
    At init recovers standard LoheAttention exactly.

    Works for any d_osc.
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int,
                 d_osc: int, max_seq_len: int, p: int = 1,
                 causal: bool = True):
        super().__init__()
        self.n_heads    = n_heads
        self.d_head     = d_head
        self.d_osc      = d_osc
        self.p          = p
        self.causal     = causal
        self.max_seq_len = max_seq_len

        self.W_q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_o = nn.Linear(n_heads * d_head, d_model, bias=True)

        anchors = torch.randn(n_heads, max_seq_len, d_osc)
        self.anchors = nn.Parameter(F.normalize(anchors, dim=-1))

        # Per-head per-query-position coupling scale; init 1.0 → standard Lohe
        self.beta = nn.Parameter(torch.ones(n_heads, max_seq_len))

        if causal:
            self.register_buffer("_causal_mask",
                                 torch.tril(torch.ones(max_seq_len, max_seq_len)),
                                 persistent=False)

    def forward(self, x: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None):
        B, T, _ = x.shape
        H, D_h, D_osc = self.n_heads, self.d_head, self.d_osc

        # ── Coupling weights with per-position scale ──────────────────────
        q = self.W_q(x).view(B, T, H, D_h).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, D_h).transpose(1, 2)
        raw = torch.einsum('bhid,bhjd->bhij', q, k) / math.sqrt(D_h)

        beta = self.beta[:, :T].abs().clamp(min=0.1)   # (H, T_q), positive
        # Broadcast: (1, H, T_q, 1) * (B, H, T_q, T_k) → (B, H, T_q, T_k)
        W = F.softplus(raw * beta.unsqueeze(0).unsqueeze(-1))

        if self.causal:
            W = W * self._causal_mask[:T, :T]
        if padding_mask is not None:
            W = W * (~padding_mask).float().view(B, 1, 1, T)

        # ── Analytic fixed point (identical to LoheAttention) ─────────────
        anc    = self.anchors[:, :T, :]                       # (H, T, D_osc)
        h      = torch.einsum('bhij,hjd->bhid', W, anc)       # (B, H, T, D_osc)
        z_star = F.normalize(h, dim=-1, eps=1e-8)

        cos_sim = torch.einsum('bhid,hjd->bhij', z_star, anc) # (B, H, T, T)
        attn = (1.0 + cos_sim).clamp(min=0.0)
        if self.p != 1:
            attn = attn ** self.p
        if self.causal:
            attn = attn * self._causal_mask[:T, :T]
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        v   = self.W_v(x).view(B, T, H, D_h).transpose(1, 2)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, H * D_h)
        return self.W_o(out), attn
