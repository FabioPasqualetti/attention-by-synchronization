"""
ODE integration utilities for Fixed-Query Oscillator Attention.

Provides:
  lohe_ode_steps          — Euler integration of the Lohe ODE on S^{d-1}
  antipodal_prob_theory   — Proposition 4: P(angle(k(0), -k*) < alpha)
  antipodal_prob_empirical — Empirical estimate via Monte Carlo sampling
"""

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


# ── Euler ODE (PyTorch) ───────────────────────────────────────────────────────

@torch.no_grad()
def lohe_ode_steps(x_init: torch.Tensor, driving: torch.Tensor,
                   N: int, dt: float = 0.05) -> torch.Tensor:
    """
    Euler integration of the Lohe ODE on S^{d_osc-1}.

    ODE: dx_i/dt = (I - x_i x_i^T) driving_i
    Fixed point: x_i* = normalize(driving_i)  [analytic]

    Args:
        x_init:  (..., D) initial state vectors (need not be normalized)
        driving: (..., D) constant forcing = sum_j W_ij * anchor_j
        N:       number of Euler steps
        dt:      step size
    Returns:
        (..., D) state after N steps, normalized to unit sphere
    """
    x = F.normalize(x_init, dim=-1, eps=1e-8)
    for _ in range(N):
        radial = (x * driving).sum(dim=-1, keepdim=True)
        tangential = driving - radial * x
        x = F.normalize(x + dt * tangential, dim=-1, eps=1e-8)
    return x


# ── Proposition 4: antipodal initialization probability ───────────────────────

def antipodal_prob_theory(alpha: float, d: int) -> float:
    """
    Compute P(angle(k(0), -k*) < alpha) from Proposition 4.

    Closed-form expression:
        P = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * integral_0^alpha sin^{d-2}(t) dt

    Args:
        alpha: angle threshold in radians (0 < alpha < pi/2)
        d:     oscillator dimension (d_osc)
    Returns:
        Probability as a float
    """
    from scipy.integrate import quad
    from scipy.special import gamma

    prefactor = gamma(d / 2) / (math.sqrt(math.pi) * gamma((d - 1) / 2))
    integral, _ = quad(lambda t: math.sin(t) ** (d - 2), 0, alpha)
    return prefactor * integral


def antipodal_prob_empirical(alpha: float, d: int, N: int = 10_000,
                              rng: np.random.Generator = None) -> float:
    """
    Empirically estimate P(angle(k(0), -k*) < alpha) by Monte Carlo.

    Draws N uniform samples from S^{d-1} and counts the fraction
    whose angle to the north pole e_1 = [1, 0, ..., 0] is less than alpha.

    Args:
        alpha: angle threshold in radians
        d:     sphere dimension
        N:     number of Monte Carlo samples
        rng:   numpy random generator (for reproducibility)
    Returns:
        Empirical probability estimate
    """
    if rng is None:
        rng = np.random.default_rng()
    samples = rng.standard_normal(size=(N, d))
    norms = np.linalg.norm(samples, axis=1, keepdims=True)
    samples = samples / norms
    pole = np.zeros(d)
    pole[0] = 1.0
    cos_angles = samples @ pole
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    angles = np.arccos(cos_angles)
    return float((angles < alpha).mean())


def antipodal_sweep(d_list: Sequence[int], alpha_list: Sequence[float],
                    N: int = 10_000, seed: int = 0) -> dict:
    """
    Compute empirical and theoretical antipodal probabilities for all (d, alpha) pairs.

    Returns:
        {d: {alpha: [empirical, theoretical]}}
    """
    rng = np.random.default_rng(seed)
    results = {}
    for d in d_list:
        results[d] = {}
        samples = rng.standard_normal(size=(N, d))
        samples = samples / np.linalg.norm(samples, axis=1, keepdims=True)
        pole = np.zeros(d)
        pole[0] = 1.0
        cos_angles = np.clip(samples @ pole, -1.0, 1.0)
        angles = np.arccos(cos_angles)
        for alpha in alpha_list:
            empirical = float((angles < alpha).mean())
            theoretical = antipodal_prob_theory(float(alpha), d)
            results[d][float(alpha)] = [empirical, theoretical]
    return results
