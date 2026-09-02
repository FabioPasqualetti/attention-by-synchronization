"""
Fixed-Query Oscillator Attention — core attention modules.

Two attention variants:

KuramotoAttention (d_osc=2, scalar):
  Oscillators live on S^1 (the circle).
  Coupling:  W_ij = softplus(alpha * f(x_i)^T g(x_j) / sqrt(d_head) + beta)
  ODE:       dθ_i/dt = −sum_j W_ij sin(θ_i − φ_j)   [Euler, N steps of dt]
  Readout:   a_ij = (1 + cos(θ*_i − φ_j))^p / norm   [no softmax]
  Used for:  KWS experiments (Tables 1, 2, 6).

LoheAttention (d_osc ≥ 2, vector):
  Oscillators live on S^{d_osc-1}.
  Coupling:  W_ij = softplus(q_i^T k_j / sqrt(d_head))
  Fixed point: x_i* = normalize(sum_j W_ij * anchor_j)   [analytic]
  Readout:   a_ij = (1 + x_i*^T anchor_j)^p / norm       [no softmax]
  Used for:  WikiText-2 and TinyStories scaling experiments (Table 5, 5b).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

TWO_PI = 2.0 * math.pi


# ── Scalar Kuramoto attention (S^1, Euler ODE) ────────────────────────────────

class KuramotoAttention(nn.Module):
    """
    Multi-head fixed-query Kuramoto attention on S^1 (Euler ODE).

    Deprecated as of 2026-05-12. Use LoheAttention(d_osc=2) instead,
    which computes the same fixed point analytically and is faster and
    more numerically stable. KuramotoAttention is retained for
    reference and backward compatibility with pre-existing checkpoints.

    Args:
        d_model:  token embedding dimension
        n_heads:  number of attention heads
        T:        maximum sequence length (for anchor phase init)
        N:        number of Euler ODE steps
        dt:       Euler step size
        p:        readout sharpening exponent (1, 2, or 4)
        causal:   apply causal mask (True for autoregressive LM)
    """

    def __init__(self, d_model: int, n_heads: int, T: int,
                 N: int, dt: float, p: int, causal: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.T = T
        self.N = N
        self.dt = dt
        self.p = p
        self.causal = causal

        self.W_f = nn.Linear(d_model, d_model, bias=False)
        self.W_g = nn.Linear(d_model, d_model, bias=False)

        self.alpha = nn.Parameter(torch.ones(n_heads))
        self.beta  = nn.Parameter(torch.zeros(n_heads))

        phi_init = (TWO_PI * torch.arange(T, dtype=torch.float32) / T
                    ).unsqueeze(0).expand(n_heads, -1).clone()
        self.phi = nn.Parameter(phi_init)

        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=True)

        if causal:
            self.register_buffer("_causal_mask",
                                 torch.tril(torch.ones(T, T)),
                                 persistent=False)

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, d_model)
        Returns: (out, attn)
            out:  (B, T, d_model)
            attn: (B*H, T, T)  row-stochastic
        """
        B, T, _ = x.shape
        H, dh = self.n_heads, self.d_head

        fi = self.W_f(x).view(B, T, H, dh).transpose(1, 2).reshape(B * H, T, dh)
        gj = self.W_g(x).view(B, T, H, dh).transpose(1, 2).reshape(B * H, T, dh)
        raw = torch.bmm(fi, gj.transpose(1, 2)) / (dh ** 0.5)  # (B*H, T, T)

        alpha = self.alpha.unsqueeze(0).expand(B, H).reshape(B * H, 1, 1)
        beta  = self.beta.unsqueeze(0).expand(B, H).reshape(B * H, 1, 1)
        W = F.softplus(alpha * raw + beta)

        if self.causal:
            W = W * self._causal_mask[:T, :T]

        # Row-normalize: bounds |dθ/dt| ≤ 1, ensuring Euler stability at dt≤0.1
        W = W / (W.sum(dim=2, keepdim=True) + 1e-8)

        phi = self.phi[:, :T].unsqueeze(0).expand(B, H, T).reshape(B * H, T)

        theta = TWO_PI * torch.rand(B * H, T, device=x.device, dtype=x.dtype)
        for _ in range(self.N):
            sin_diff = torch.sin(theta.unsqueeze(2) - phi.unsqueeze(1))
            dtheta = -(W * sin_diff).sum(dim=2)
            theta = (theta + self.dt * dtheta) % TWO_PI

        cos_sim = torch.cos(theta.unsqueeze(2) - phi.unsqueeze(1))
        shifted = (1.0 + cos_sim).clamp(min=0.0).pow(self.p)

        if self.causal:
            shifted = shifted * self._causal_mask[:T, :T]

        attn = shifted / (shifted.sum(dim=2, keepdim=True) + 1e-8)

        self._last_theta = theta.detach()
        self._last_phi   = phi.detach()

        V   = self.W_v(x).view(B, T, H, dh).transpose(1, 2).reshape(B * H, T, dh)
        out = torch.bmm(attn, V)
        out = out.view(B, H, T, dh).transpose(1, 2).reshape(B, T, H * dh)
        out = self.W_o(out)
        return out, attn


# ── Lohe attention (S^{d_osc-1}, analytic fixed point) ───────────────────────

class LoheAttention(nn.Module):
    """
    Multi-head Lohe attention on S^{d_osc-1}.

    At d_osc=2, identical to analytic scalar Kuramoto (closed-form fixed point).
    At d_osc > 2, oscillators live on S^{d_osc-1}; fixed point is analytic.

    Args:
        d_model:     token embedding dimension
        n_heads:     number of attention heads
        d_head:      per-head QK projection dimension
        d_osc:       oscillator sphere dimension (d_osc=2 → S^1)
        max_seq_len: maximum sequence length (for anchor parameter shape)
        p:           readout sharpening exponent (default 1)
        causal:      apply causal mask (True for autoregressive LM)
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int,
                 d_osc: int, max_seq_len: int, p: int = 1,
                 causal: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_osc = d_osc
        self.p = p
        self.causal = causal
        self.max_seq_len = max_seq_len

        self.W_q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_o = nn.Linear(n_heads * d_head, d_model, bias=True)

        # Anchor vectors on S^{d_osc-1}: (n_heads, max_seq_len, d_osc)
        anchors = torch.randn(n_heads, max_seq_len, d_osc)
        self.anchors = nn.Parameter(F.normalize(anchors, dim=-1))

        if causal:
            self.register_buffer("_causal_mask",
                                 torch.tril(torch.ones(max_seq_len, max_seq_len)),
                                 persistent=False)

    def forward(self, x: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None):
        """
        x:            (B, T, d_model)
        padding_mask: (B, T) bool, True = padding position
        Returns: (out, attn)
            out:  (B, T, d_model)
            attn: (B, H, T, T) row-stochastic
        """
        B, T, _ = x.shape
        H, D_h, D_osc = self.n_heads, self.d_head, self.d_osc
        device = x.device

        q = self.W_q(x).view(B, T, H, D_h).transpose(1, 2)  # (B, H, T, D_h)
        k = self.W_k(x).view(B, T, H, D_h).transpose(1, 2)
        raw = torch.einsum('bhid,bhjd->bhij', q, k) / math.sqrt(D_h)
        W = F.softplus(raw)  # (B, H, T, T), all positive

        if self.causal:
            W = W * self._causal_mask[:T, :T]

        if padding_mask is not None:
            col_mask = (~padding_mask).float().view(B, 1, 1, T)
            W = W * col_mask

        # Analytic fixed point: x_i* = normalize(sum_j W_ij * anchor_j)
        anc = self.anchors[:, :T, :]                         # (H, T, D_osc)
        h = torch.einsum('bhij,hjd->bhid', W, anc)           # (B, H, T, D_osc)
        x_star = F.normalize(h, dim=-1, eps=1e-8)            # (B, H, T, D_osc)

        cos_sim = torch.einsum('bhid,hjd->bhij', x_star, anc)  # (B, H, T, T)
        attn = (1.0 + cos_sim).clamp(min=0.0)
        if self.p != 1:
            attn = attn ** self.p
        if self.causal:
            attn = attn * self._causal_mask[:T, :T]
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        v = self.W_v(x).view(B, T, H, D_h).transpose(1, 2)  # (B, H, T, D_h)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, H * D_h)
        out = self.W_o(out)
        return out, attn


# ── Softmax attention (baseline) ──────────────────────────────────────────────

class SoftmaxAttention(nn.Module):
    """Standard scaled dot-product softmax attention (baseline)."""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 2048):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = self.d_head ** 0.5
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=True)
        self.register_buffer("_causal_mask",
                             torch.triu(torch.ones(max_seq_len, max_seq_len,
                                                   dtype=torch.bool), diagonal=1),
                             persistent=False)

    def forward(self, x: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None,
                causal: bool = False):
        B, T, _ = x.shape
        H, dh = self.n_heads, self.d_head

        Q = self.W_q(x).view(B, T, H, dh).transpose(1, 2).reshape(B * H, T, dh)
        K = self.W_k(x).view(B, T, H, dh).transpose(1, 2).reshape(B * H, T, dh)
        V = self.W_v(x).view(B, T, H, dh).transpose(1, 2).reshape(B * H, T, dh)

        logits = torch.bmm(Q, K.transpose(1, 2)) / self.scale

        if causal:
            logits = logits.masked_fill(self._causal_mask[:T, :T].unsqueeze(0),
                                        float("-inf"))

        if padding_mask is not None:
            mask = padding_mask.unsqueeze(1).expand(B, H, T).reshape(B * H, 1, T)
            logits = logits.masked_fill(mask, float("-inf"))

        attn = F.softmax(logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        out = torch.bmm(attn, V)
        out = out.view(B, H, T, dh).transpose(1, 2).reshape(B, T, H * dh)
        return self.W_o(out), attn
