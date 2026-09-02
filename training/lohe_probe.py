"""Instrumentation for robustness_perturbations (robustness) and theory_degenerate_tokens (degenerate-set stats).

All math mirrors oscillator_attention.attention.LoheAttention exactly; trained modules are
wrapped by reference, never modified.

Two facilities:

1. HNormCapture — forward_pre_hook on a LoheAttention/LoheAttentionSigma module that
   recomputes h = sum_j W_ij anchor_j (the pre-normalization "driving" vector,
   attention.py:204) and records ||h_i|| per (query position, head). Optional pad
   mask filters padded query positions.

2. PerturbableLohe — an nn.Module that wraps a trained LoheAttention (sharing its
   parameters) and reimplements forward with inference-time perturbations:
     - coupling mismatch:  W_ij <- W_ij * exp(eps_ij), eps ~ N(0, s^2)
     - state noise:        z* <- normalize(z* + eta), eta ~ N(0, s^2 I)
     - finite settling:    integrate the Lohe ODE (scipy RK45, rtol=atol=1e-6, CPU)
                           from random z(0) to time Tset, read out there.
   swap_lohe() replaces every LoheAttention-like module in a model in place.
"""
import math
from typing import Optional

from . import paths
paths.ensure_paths()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from oscillator_attention.attention import LoheAttention
from oscillator_attention.coupling_variants import LoheAttentionSigma

LOHE_TYPES = (LoheAttention, LoheAttentionSigma)


def _sigma_of(mod):
    return getattr(mod, "sigma", "softplus")


def _coupling(sigma, raw):
    if sigma == "relu_eps":
        return F.relu(raw) + 1e-3
    if sigma == "elu1":
        return F.elu(raw) + 1.0
    return F.softplus(raw)


# ── theory_degenerate_tokens: ||h|| capture ────────────────────────────────────────────────────────

class HNormCapture:
    """Registers pre-hooks on all Lohe modules; accumulates ||h_i|| values."""

    def __init__(self, model):
        self.norms = []            # list of 1D numpy arrays
        self._pad = None           # (B,T) bool, True=pad; set per batch
        self.handles = []
        for m in model.modules():
            if isinstance(m, LOHE_TYPES):
                self.handles.append(m.register_forward_pre_hook(self._hook))

    def set_pad(self, pad_mask):
        self._pad = pad_mask

    def _hook(self, module, args):
        x = args[0]
        B, T, _ = x.shape
        H, D_h = module.n_heads, module.d_head
        q = module.W_q(x).view(B, T, H, D_h).transpose(1, 2)
        k = module.W_k(x).view(B, T, H, D_h).transpose(1, 2)
        raw = torch.einsum('bhid,bhjd->bhij', q, k) / math.sqrt(D_h)
        W = _coupling(_sigma_of(module), raw)
        if module.causal:
            W = W * module._causal_mask[:T, :T]
        if self._pad is not None:
            W = W * (~self._pad).float().view(B, 1, 1, T)
        anc = module.anchors[:, :T, :]
        h = torch.einsum('bhij,hjd->bhid', W, anc)     # (B,H,T,D_osc)
        hn = h.norm(dim=-1)                             # (B,H,T)
        if self._pad is not None:
            keep = (~self._pad).view(B, 1, T).expand(B, H, T)
            hn = hn[keep]
        else:
            hn = hn.reshape(-1)
        self.norms.append(hn.detach().cpu().numpy())

    def remove(self):
        for h in self.handles:
            h.remove()

    def all_norms(self):
        return np.concatenate(self.norms) if self.norms else np.array([])


# ── robustness_perturbations: perturbable Lohe ─────────────────────────────────────────────────────

# Counter for how often the RK45 batch fell back to renormalized Euler (for reporting).
RK45_FALLBACKS = [0, 0]  # [fallback_calls, total_calls]


def _integrate_ode_rk45(driving_np, x0_np, Tset, rtol=1e-6, atol=1e-6):
    """Integrate dx/dt = (I - x x^T) g for M independent oscillators to time Tset.

    Primary: scipy RK45 (rtol=atol=1e-6), all oscillators as one decoupled system.
    Robust fallback: if the batched adaptive RK45 fails to reach Tset (a stiff mix of
    |g| magnitudes across thousands of oscillators can force the global step below the
    minimum, leaving sol.y empty/malformed), fall back to the repo's renormalized Euler
    integrator (oscillator_attention.ode.lohe_ode_steps), which is unconditionally stable
    (it projects back to the sphere each step). Both integrate the same Lohe ODE.

    driving_np, x0_np: (M, D) numpy. Returns (M, D) state at Tset.
    """
    RK45_FALLBACKS[1] += 1
    M, D = driving_np.shape

    def rhs(t, y):
        x = y.reshape(M, D)
        radial = (x * driving_np).sum(axis=1, keepdims=True)
        return (driving_np - radial * x).reshape(-1)

    try:
        from scipy.integrate import solve_ivp
        sol = solve_ivp(rhs, (0.0, float(Tset)), x0_np.reshape(-1),
                        method="RK45", rtol=rtol, atol=atol, t_eval=[float(Tset)])
        y = np.asarray(sol.y, dtype=float)
        if getattr(sol, "success", False) and y.ndim == 2 and y.shape[1] >= 1:
            return y[:, -1].reshape(M, D)
    except Exception:
        pass

    # Fallback: stable renormalized Euler (dt=0.05, N=Tset/dt).
    RK45_FALLBACKS[0] += 1
    from oscillator_attention.ode import lohe_ode_steps
    N = max(1, int(round(float(Tset) / 0.05)))
    x = lohe_ode_steps(torch.from_numpy(x0_np).float(),
                       torch.from_numpy(driving_np).float(), N=N, dt=0.05)
    return x.detach().numpy()


class PerturbableLohe(nn.Module):
    def __init__(self, src, coupling_s=0.0, state_s=0.0, ode_T=None,
                 rng_seed=0, rtol=1e-6, atol=1e-6):
        super().__init__()
        self.src = src                       # shares parameters by reference
        self.coupling_s = coupling_s
        self.state_s = state_s
        self.ode_T = ode_T
        self.rng_seed = rng_seed
        self.rtol = rtol
        self.atol = atol

    def forward(self, x, padding_mask: Optional[torch.Tensor] = None):
        m = self.src
        B, T, _ = x.shape
        H, D_h, D_osc = m.n_heads, m.d_head, m.d_osc
        gen = torch.Generator(device="cpu").manual_seed(self.rng_seed)

        q = m.W_q(x).view(B, T, H, D_h).transpose(1, 2)
        k = m.W_k(x).view(B, T, H, D_h).transpose(1, 2)
        raw = torch.einsum('bhid,bhjd->bhij', q, k) / math.sqrt(D_h)
        W = _coupling(_sigma_of(m), raw)

        # (a) coupling mismatch: W_ij <- W_ij * exp(eps_ij)
        if self.coupling_s > 0:
            eps = torch.randn(W.shape, generator=gen).to(W.device) * self.coupling_s
            W = W * torch.exp(eps)

        if m.causal:
            W = W * m._causal_mask[:T, :T]
        if padding_mask is not None:
            W = W * (~padding_mask).float().view(B, 1, 1, T)

        anc = m.anchors[:, :T, :]
        h = torch.einsum('bhij,hjd->bhid', W, anc)          # driving (B,H,T,D)

        if self.ode_T is not None:
            # (c) finite settling via RK45 from random z(0)
            g = h.detach().cpu().numpy().reshape(-1, D_osc)
            x0 = torch.randn(g.shape[0], D_osc, generator=gen).numpy()
            x0 = x0 / (np.linalg.norm(x0, axis=1, keepdims=True) + 1e-12)
            xset = _integrate_ode_rk45(g, x0, self.ode_T, self.rtol, self.atol)
            x_star = torch.from_numpy(xset).to(h.device, h.dtype).view(B, H, T, D_osc)
            x_star = F.normalize(x_star, dim=-1, eps=1e-8)
        else:
            x_star = F.normalize(h, dim=-1, eps=1e-8)

        # (b) state noise: z* <- normalize(z* + eta)
        if self.state_s > 0:
            eta = torch.randn(x_star.shape, generator=gen).to(x_star.device) * self.state_s
            x_star = F.normalize(x_star + eta, dim=-1, eps=1e-8)

        cos_sim = torch.einsum('bhid,hjd->bhij', x_star, anc)
        attn = (1.0 + cos_sim).clamp(min=0.0)
        if m.p != 1:
            attn = attn ** m.p
        if m.causal:
            attn = attn * m._causal_mask[:T, :T]
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        v = m.W_v(x).view(B, T, H, D_h).transpose(1, 2)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, H * D_h)
        return m.W_o(out), attn


def swap_lohe(model, **cfg):
    """Replace every LoheAttention-like submodule with PerturbableLohe(cfg). In place.

    Targets are collected in a single pass BEFORE any mutation, so we neither mutate
    the tree while iterating it nor re-wrap an already-wrapped module.
    """
    targets = []
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, LOHE_TYPES):
                targets.append((parent, name, child))
    for parent, name, child in targets:
        setattr(parent, name, PerturbableLohe(child, **cfg))
    return model
