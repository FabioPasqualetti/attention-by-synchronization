"""
oscillator_attention_L6.py — Lohe model on S^{d_osc-1}.

At d_osc=2 reduces to the analytic fixed-point variant of scalar Kuramoto
(exact solution, no ODE approximation).

At d_osc > 2: oscillators live on S^{d_osc-1}; fixed point computed
analytically as x_i* = normalize(sum_j W_ij * anchor_j).

Hypothesis: larger d_osc reduces the scalar compression bottleneck
and closes the Kuramoto–softmax PTB gap.
"""

import math
from typing import Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


# ── Lohe ODE utilities ────────────────────────────────────────────────────────

@torch.no_grad()
def lohe_ode_steps(x_init: torch.Tensor, driving: torch.Tensor,
                   N: int, dt: float = 0.05) -> torch.Tensor:
    """
    Run N Euler steps of the Lohe ODE on S^{d_osc-1}.

    ODE: dx_i/dt = (I - x_i x_i^T) driving_i
    Fixed point: x_i* = normalize(driving_i)  [same as analytic solution]

    x_init:  (..., D_osc) initial state (any vectors, will be normalized)
    driving: (..., D_osc) constant forcing = sum_j W_ij * anchor_j
    Returns: (..., D_osc) state after N Euler steps
    """
    x = F.normalize(x_init, dim=-1, eps=1e-8)
    for _ in range(N):
        radial = (x * driving).sum(dim=-1, keepdim=True)  # x^T driving
        tangential = driving - radial * x                  # project to tangent plane
        x = F.normalize(x + dt * tangential, dim=-1, eps=1e-8)
    return x


@torch.no_grad()
def compute_driving_force(model: "LoheLanguageTransformer",
                           tokens: torch.Tensor,
                           pad_mask: Optional[torch.Tensor],
                           layer_idx: int = 0) -> torch.Tensor:
    """
    Compute driving force F_i = sum_{j<=i} W_ij * anchor_j for each position.

    Returns: (B, H, T, D_osc)
    """
    B, T = tokens.shape
    device = tokens.device

    h = model.pos_enc(model.embedding(tokens))
    layer = model.layers[layer_idx]
    attn = layer.attn
    normed = layer.norm1(h)

    H, D_h, D_osc = attn.n_heads, attn.d_head, attn.d_osc

    q = attn.W_q(normed).view(B, T, H, D_h).transpose(1, 2)
    k = attn.W_k(normed).view(B, T, H, D_h).transpose(1, 2)
    raw = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(D_h)
    W = F.softplus(raw)

    W = W * torch.tril(torch.ones(T, T, device=device))
    if pad_mask is not None:
        W = W * (~pad_mask).float().view(B, 1, 1, T)

    anc = attn.anchors[:, :T, :]                          # (H, T, D_osc)
    driving = torch.einsum("bhij,hjd->bhid", W, anc)      # (B, H, T, D_osc)
    return driving


def run_ode_verification(
    model: "LoheLanguageTransformer",
    val_loader,
    device: torch.device,
    N_steps_list: List[int] = (50, 100, 200, 500, 1000),
    dt: float = 0.05,
    n_batches: int = 5,
) -> Dict:
    """
    Compare ODE fixed-point to analytic fixed-point for each N in N_steps_list.

    Reports mean/max convergence error per sequence position, averaged over
    n_batches batches.  Both solutions should converge to the same point —
    discrepancy signals hardware-inference inconsistency.

    Returns dict: {N: {"mean": float, "max": float, "per_pos": list[float]}}
    """
    model.eval()
    layer = model.layers[0]
    attn  = layer.attn
    H, D_osc = attn.n_heads, attn.d_osc

    errors: Dict[int, List[torch.Tensor]] = {N: [] for N in N_steps_list}

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= n_batches:
                break
            tokens   = batch["tokens"].to(device)
            pad_mask = batch["pad_mask"].to(device)
            B, T = tokens.shape

            driving    = compute_driving_force(model, tokens, pad_mask, layer_idx=0)
            x_analytic = F.normalize(driving, dim=-1, eps=1e-8)  # (B,H,T,D_osc)

            for N in N_steps_list:
                x_init = torch.randn(B, H, T, D_osc, device=device)
                x_ode  = lohe_ode_steps(x_init, driving, N, dt)
                err    = (x_ode - x_analytic).norm(dim=-1)        # (B,H,T)
                errors[N].append(err.mean(dim=(0, 1)).cpu())       # (T,)

    print(f"\n{'='*64}")
    print(f"ODE VERIFICATION  d_osc={D_osc}  dt={dt}  layer=0  n_batches={n_batches}")
    print(f"{'='*64}")
    print(f"  {'N':>6}  {'mean_err':>10}  {'max_err':>10}  "
          f"{'err@pos0':>10}  {'err@pos[-1]':>12}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*12}")

    summary = {}
    for N in N_steps_list:
        if not errors[N]:
            continue
        per_pos  = torch.stack(errors[N]).mean(dim=0)       # (T,)
        mean_err = per_pos.mean().item()
        max_err  = per_pos.max().item()
        print(f"  {N:>6}  {mean_err:>10.5f}  {max_err:>10.5f}  "
              f"{per_pos[0].item():>10.5f}  {per_pos[-1].item():>12.5f}")
        summary[N] = {"mean": mean_err, "max": max_err,
                      "per_pos": per_pos.tolist()}

    n500_err = summary.get(500, {}).get("mean", float("inf"))
    if n500_err < 0.01:
        verdict = "PASS — ODE converges to analytic FP (hardware-consistent)"
    elif n500_err < 0.05:
        verdict = "MARGINAL — ODE approaches analytic FP with residual error"
    else:
        verdict = "FAIL — ODE does not converge to analytic FP"
    print(f"\n  N=500 verdict: {verdict}")
    print(f"{'='*64}\n")

    return summary


class LoheAttention(nn.Module):
    """
    Multi-head Lohe attention on S^{d_osc-1}.

    Coupling weights W_ij remain scalars (same as Kuramoto).
    What changes is the space the oscillators live in.

    Args:
        d_model:     token embedding dimension
        n_heads:     number of attention heads
        d_head:      per-head projection dimension (for W_ij computation)
        d_osc:       oscillator sphere dimension (d_osc=2 → S^1 = scalar Kuramoto)
        max_seq_len: maximum sequence length (for anchor parameter shape)
        p:           readout exponent (default 1)
        causal:      apply causal mask (True for PTB)
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

        # Query-key projections for coupling weights W_ij
        self.W_q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_head, bias=False)

        # Value and output projections
        self.W_v = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_o = nn.Linear(n_heads * d_head, d_model, bias=True)

        # Anchor vectors on S^{d_osc-1}: (n_heads, max_seq_len, d_osc)
        anchors = torch.randn(n_heads, max_seq_len, d_osc)
        self.anchors = nn.Parameter(F.normalize(anchors, dim=-1))

    def forward(self, x: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None):
        """
        x:            (B, T, d_model)
        padding_mask: (B, T) bool, True = pad position
        Returns: (out, attn)
            out:  (B, T, d_model)
            attn: (B, H, T, T) row-stochastic
        """
        B, T, _ = x.shape
        H, D_h, D_osc = self.n_heads, self.d_head, self.d_osc
        device = x.device

        # 1. Coupling weights W_ij = softplus(q_i^T k_j / sqrt(d_head))
        q = self.W_q(x).view(B, T, H, D_h).transpose(1, 2)  # (B, H, T, D_h)
        k = self.W_k(x).view(B, T, H, D_h).transpose(1, 2)  # (B, H, T, D_h)
        raw = torch.einsum('bhid,bhjd->bhij', q, k) / math.sqrt(D_h)  # (B,H,T,T)
        W = F.softplus(raw)  # (B, H, T, T), all positive

        # 2. Causal mask: zero out j > i (future tokens)
        if self.causal:
            causal_mask = torch.tril(torch.ones(T, T, device=device))  # (T,T)
            W = W * causal_mask

        # 3. Padding mask: zero out columns of pad tokens
        if padding_mask is not None:
            col_mask = (~padding_mask).float().view(B, 1, 1, T)  # (B,1,1,T)
            W = W * col_mask

        # 4. Analytic fixed point on S^{d_osc-1}
        #    h_i = sum_{j<=i} W_ij * anchor_j
        #    x_i* = h_i / ||h_i||
        anc = self.anchors[:, :T, :]              # (H, T, D_osc)
        h = torch.einsum('bhij,hjd->bhid', W, anc)   # (B, H, T, D_osc)
        x_star = F.normalize(h, dim=-1, eps=1e-8)    # (B, H, T, D_osc)

        # 5. Cosine similarity with anchors → attention logits
        cos_sim = torch.einsum('bhid,hjd->bhij', x_star, anc)  # (B,H,T,T)

        # 6. Polynomial readout + causal mask + row-normalize
        attn = (1.0 + cos_sim).clamp(min=0.0) ** self.p  # (B,H,T,T)
        if self.causal:
            attn = attn * causal_mask
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # 7. Value aggregation
        v = self.W_v(x).view(B, T, H, D_h).transpose(1, 2)   # (B,H,T,D_h)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)         # (B,H,T,D_h)
        out = out.transpose(1, 2).contiguous().view(B, T, H * D_h)
        out = self.W_o(out)                                     # (B,T,d_model)

        return out, attn


class LoheTransformerLayer(nn.Module):
    """Standard pre-norm transformer layer using LoheAttention."""

    def __init__(self, d_model: int, n_heads: int, d_head: int, d_ff: int,
                 dropout: float, d_osc: int, max_seq_len: int,
                 causal: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = LoheAttention(d_model, n_heads, d_head, d_osc,
                                  max_seq_len, p=1, causal=causal)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
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
    Language model (PTB) using LoheAttention.
    Architecture matches L5c: d_model=128, n_heads=4, n_layers=2, d_ff=512.
    Only d_osc varies across L6 conditions.
    """

    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int, max_seq_len: int, dropout: float,
                 d_osc: int):
        super().__init__()
        assert d_model % n_heads == 0
        d_head = d_model // n_heads

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)
        self.layers = nn.ModuleList([
            LoheTransformerLayer(d_model, n_heads, d_head, d_ff, dropout,
                                 d_osc, max_seq_len, causal=True)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None,
                verb_idx=None) -> torch.Tensor:
        h = self.pos_enc(self.embedding(x))
        for layer in self.layers:
            h = layer(h, padding_mask=padding_mask)
        h = self.norm(h)
        return self.head(h)
