"""
model_backend.py — Model loading and inference for the Kuramoto demo.
"""

import math
import re
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from oscillator_attention_L6 import (LoheLanguageTransformer,
                                    SinusoidalPositionalEncoding)

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


def _tokenize(text: str) -> list:
    text = text.lower()
    text = re.sub(r"([.,!?;:()\[\]{}\"'])", r" \1 ", text)
    return [t for t in text.split() if t]


# ── Softmax transformer (for comparison model) ───────────────────────────────

class _SoftmaxAttn(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = self.d_head ** 0.5
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x, causal=True):
        B, T, _ = x.shape
        H, dh = self.n_heads, self.d_head
        Q = self.W_q(x).view(B, T, H, dh).transpose(1,2).reshape(B*H, T, dh)
        K = self.W_k(x).view(B, T, H, dh).transpose(1,2).reshape(B*H, T, dh)
        V = self.W_v(x).view(B, T, H, dh).transpose(1,2).reshape(B*H, T, dh)
        logits = torch.bmm(Q, K.transpose(1,2)) / self.scale
        if causal:
            mask = torch.triu(torch.ones(T,T, device=x.device, dtype=torch.bool), 1)
            logits = logits.masked_fill(mask.unsqueeze(0), float("-inf"))
        attn = F.softmax(logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        out  = torch.bmm(attn, V).view(B, H, T, dh).transpose(1,2).reshape(B,T,H*dh)
        return self.W_o(out), attn.view(B, H, T, T)


class _SoftmaxLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = _SoftmaxAttn(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout))

    def forward(self, x):
        out, attn = self.attn(self.norm1(x), causal=True)
        x = x + out
        x = x + self.ffn(self.norm2(x))
        return x, attn


class SoftmaxLanguageTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff,
                 max_seq_len, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc   = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)
        self.layers    = nn.ModuleList([
            _SoftmaxLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.pos_enc(self.embedding(x))
        for layer in self.layers:
            h, _ = layer(h)
        return self.head(self.norm(h))


# ── Shared predictor base ─────────────────────────────────────────────────────

class _BasePredictor:
    """Common encode/decode/sample logic for all models."""

    def _load_vocab(self, demo_dir, mcfg: dict = None):
        # Optional per-model vocab (e.g. TinyStories uses vocab_ts.pt)
        rel = (mcfg or {}).get("vocab_path", "cache/vocab.pt")
        path = os.path.join(demo_dir, rel)
        vd = torch.load(path, map_location="cpu", weights_only=False)
        self.vocab      = vd["vocab"]
        self.idx2word   = vd["idx2word"]
        self.vocab_size = len(self.vocab)

    def encode(self, text: str) -> list:
        words = _tokenize(text)
        return [BOS_IDX] + [self.vocab.get(w, UNK_IDX) for w in words]

    def decode(self, idx: int) -> str:
        return self.idx2word.get(idx, "<unk>")

    def sample_next_token(self, logits: torch.Tensor, temperature: float = 0.7,
                          greedy: bool = False):
        logits = logits.clone()
        logits[[PAD_IDX, BOS_IDX, UNK_IDX]] = float("-inf")
        # Greedy: always take the argmax (temperature only affects display probs)
        token_id = int(logits.argmax().item()) if greedy else \
                   int(torch.multinomial(F.softmax(logits / max(temperature, 0.01), dim=-1), 1).item())
        # Display probabilities use temperature for visual spread
        disp = F.softmax(logits / max(temperature, 0.01), dim=-1)
        top5 = torch.topk(disp, 5)
        top5_list = [(self.decode(i.item()), p.item())
                     for i, p in zip(top5.indices, top5.values)]
        return token_id, self.decode(token_id), top5_list


# ── Kuramoto / Lohe predictor ─────────────────────────────────────────────────

class KuramotoWordPredictor(_BasePredictor):

    def __init__(self, model_key: str, config: dict, device_str: str = "cpu"):
        self.model_key      = model_key
        self.config         = config
        self.device         = torch.device(device_str)
        self.last_coupling  = None   # (T,) W_ij for last position, last layer
        self.last_h_vec     = None   # (d_osc,) drive vector h, last get_fixed_points
        self.last_ancs      = None   # (T, d_osc) normalised anchors, last layer
        demo_dir = os.path.dirname(os.path.abspath(__file__))
        mcfg = config["models"][model_key]
        self.d_osc    = mcfg["d_osc"]
        self.d_model  = mcfg["d_model"]
        self.n_heads  = mcfg["n_heads"]
        self.n_layers = mcfg["n_layers"]
        self.label    = mcfg["label"]
        self.ppl      = mcfg["ppl"]
        self._load_vocab(demo_dir, mcfg)

        max_seq = mcfg.get("max_seq_len", 50)
        ckpt = os.path.join(demo_dir, mcfg["checkpoint"])
        self.model = LoheLanguageTransformer(
            vocab_size=self.vocab_size, d_model=self.d_model,
            n_heads=self.n_heads, n_layers=self.n_layers,
            d_ff=self.d_model * 4, max_seq_len=max_seq,
            dropout=0.0, d_osc=self.d_osc)
        self.model.load_state_dict(
            torch.load(ckpt, map_location=self.device, weights_only=False))
        self.model.to(self.device).eval()

    @torch.no_grad()
    def get_fixed_points(self, token_ids: list):
        ids_t  = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        T      = len(token_ids)
        H, D_h = self.n_heads, self.d_model // self.n_heads
        x_stars, attentions = [], []
        h = self.model.pos_enc(self.model.embedding(ids_t))
        for layer in self.model.layers:
            attn   = layer.attn
            normed = layer.norm1(h)
            q = attn.W_q(normed).view(1,T,H,D_h).transpose(1,2)
            k = attn.W_k(normed).view(1,T,H,D_h).transpose(1,2)
            raw = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(D_h)
            W   = F.softplus(raw) * torch.tril(torch.ones(T,T, device=self.device))
            anc  = attn.anchors[:, :T, :]
            hh   = torch.einsum("bhij,hjd->bhid", W, anc)
            xstar = F.normalize(hh, dim=-1, eps=1e-8)
            # Renormalize head-avg so result stays on S^{d_osc-1}
            x_stars.append(
                F.normalize(xstar[0].mean(0), dim=-1, eps=1e-8).cpu().numpy())
            cos_sim = torch.einsum("bhid,hjd->bhij", xstar, anc)
            aw = (1.0 + cos_sim).clamp(min=0.0)
            aw = aw * torch.tril(torch.ones(T,T, device=self.device))
            aw = aw / aw.sum(-1, keepdim=True).clamp(min=1e-8)
            attentions.append(aw[0].cpu().numpy())
            v   = attn.W_v(normed).view(1,T,H,D_h).transpose(1,2)
            out = torch.einsum("bhij,bhjd->bhid", aw, v)
            out = out.transpose(1,2).contiguous().view(1,T,H*D_h)
            h   = h + attn.W_o(out)
            h   = h + layer.ffn(layer.norm2(h))
        logits = self.model.head(self.model.norm(h))[0, -1, :]
        # h_sum = Σ_h normalize(hh_h) — same direction as active_x (the green dot).
        # Decompose per-token: contrib_j = Σ_h (W_{hj}/||hh_h||) * anc_{hj}
        # so Σ_j contrib_j = h_sum and normalize(h_sum) == active_x exactly.
        xstar_heads  = F.normalize(hh[0, :, -1, :], dim=-1)          # (H, d_osc) per-head unit x★
        self.last_h_vec = xstar_heads.sum(0).cpu().numpy()            # (d_osc,) h_sum
        hh_norms = hh[0, :, -1, :].norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (H, 1)
        contrib  = torch.einsum("hj,hjd->jd",
                                W[0, :, -1, :] / hh_norms, anc)      # (T, d_osc)
        self.last_coupling = contrib.norm(dim=-1).cpu().numpy()       # (T,)
        self.last_ancs     = F.normalize(contrib, dim=-1, eps=1e-8).cpu().numpy()  # (T, d_osc)
        return np.array(x_stars[-1]), np.array(attentions), logits.cpu()

    @torch.no_grad()
    def get_logits_at_position(self, token_ids: list, x_t_np: np.ndarray):
        """Run forward pass with last token's oscillator fixed at x_t_np."""
        ids_t = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        T = len(token_ids)
        H, D_h = self.n_heads, self.d_model // self.n_heads
        x_t = torch.from_numpy(np.array(x_t_np, dtype=np.float32)).to(self.device)
        x_t = F.normalize(x_t.unsqueeze(0), dim=-1)  # (1, d_osc)

        h = self.model.pos_enc(self.model.embedding(ids_t))
        for layer in self.model.layers:
            attn   = layer.attn
            normed = layer.norm1(h)
            q = attn.W_q(normed).view(1, T, H, D_h).transpose(1, 2)
            k = attn.W_k(normed).view(1, T, H, D_h).transpose(1, 2)
            raw = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(D_h)
            W   = F.softplus(raw) * torch.tril(torch.ones(T, T, device=self.device))
            anc  = attn.anchors[:, :T, :]
            hh   = torch.einsum("bhij,hjd->bhid", W, anc)
            xstar = F.normalize(hh, dim=-1, eps=1e-8)
            # Override last token's oscillator with x_t
            x_t_exp = F.normalize(
                x_t.view(1, 1, 1, self.d_osc).expand(1, H, 1, self.d_osc), dim=-1)
            xstar = torch.cat([xstar[:, :, :-1, :], x_t_exp], dim=2)
            cos_sim = torch.einsum("bhid,hjd->bhij", xstar, anc)
            aw = (1.0 + cos_sim).clamp(min=0.0)
            aw = aw * torch.tril(torch.ones(T, T, device=self.device))
            aw = aw / aw.sum(-1, keepdim=True).clamp(min=1e-8)
            v   = attn.W_v(normed).view(1, T, H, D_h).transpose(1, 2)
            out = torch.einsum("bhij,bhjd->bhid", aw, v)
            out = out.transpose(1, 2).contiguous().view(1, T, H * D_h)
            h   = h + attn.W_o(out)
            h   = h + layer.ffn(layer.norm2(h))
        return self.model.head(self.model.norm(h))[0, -1, :].cpu()

    def get_anchors_2d(self, T: int) -> np.ndarray:
        """Normalized anchor positions on S^1 for head-0, positions 0..T-1."""
        raw = self.model.layers[0].attn.anchors[0, :T, :].detach()
        # Re-normalize: anchors may have drifted off sphere during training
        return F.normalize(raw, dim=-1).cpu().numpy()   # (T, d_osc)

    def get_ode_trajectory(self, x_prev=None, n_pts: int = 100):
        """RK45 trajectory in (x_init, x*) plane. Works for all d_osc.

        Call after get_fixed_points — uses stored last_h_vec / last_ancs.
        d_osc=2  : returns absolute 2D coords (consistent with settled positions).
        d_osc>2  : returns trajectory projected onto {x_init, x*} plane (exact).

        Returns (traj_2d, e1, e2, ancs_2d) or None on failure.
        """
        from scipy.integrate import solve_ivp
        if self.last_h_vec is None:
            return None
        h      = self.last_h_vec.astype(np.float64)
        h_norm = np.linalg.norm(h)
        if h_norm < 1e-8:
            return None
        e1 = h / h_norm   # direction of x*

        # ── Initial condition ────────────────────────────────────────────
        x0 = None
        if x_prev is not None:
            c  = np.array(x_prev, dtype=np.float64)[:self.d_osc]
            cn = np.linalg.norm(c)
            if cn > 1e-8:
                c /= cn
                if np.dot(c, e1) < 0.98:   # more than ~11° away → use it
                    x0 = c
        if x0 is None:
            # Random start 90°–150° from x* so convergence is visible
            v  = np.random.randn(self.d_osc)
            v -= np.dot(v, e1) * e1
            v /= np.linalg.norm(v)
            ang = np.radians(90 + np.random.rand() * 60)
            x0  = np.cos(ang) * e1 + np.sin(ang) * v

        # ── Basis vectors (trajectory lives in this 2D plane) ───────────
        x_perp = x0 - np.dot(x0, e1) * e1
        pn = np.linalg.norm(x_perp)
        if pn > 1e-6:
            e2 = x_perp / pn
        else:
            v  = np.random.randn(self.d_osc)
            v -= np.dot(v, e1) * e1
            e2 = v / np.linalg.norm(v)

        # ── RK45 integration ─────────────────────────────────────────────
        def rhs(t, y):
            nrm = np.linalg.norm(y)
            x   = y / max(nrm, 1e-10)
            return h - np.dot(x, h) * x

        theta0 = math.acos(float(np.clip(np.dot(x0, e1), -1, 1)))
        t_max  = min(50.0, max(4.0, 3.0 * (theta0 + 0.5) / (h_norm + 0.1)))
        t_eval = np.linspace(0, t_max, n_pts)
        sol    = solve_ivp(rhs, (0, t_max), x0.copy(),
                           method='RK45', t_eval=t_eval, rtol=1e-3, atol=1e-4)
        traj  = sol.y.T                                         # (n_pts, d_osc)
        nrms  = np.linalg.norm(traj, axis=1, keepdims=True)
        traj /= np.maximum(nrms, 1e-8)
        if np.isnan(traj).any() or np.isinf(traj).any():
            return None

        # ── Project to 2D ────────────────────────────────────────────────
        if self.d_osc == 2:
            traj_2d = traj                    # absolute 2D, already on S^1
            ancs_2d = self.last_ancs          # (T, 2), pull dirs on S^1
        else:
            traj_2d = np.column_stack([traj.dot(e1), traj.dot(e2)])
            if self.last_ancs is not None:
                raw     = np.column_stack([self.last_ancs.dot(e1),
                                           self.last_ancs.dot(e2)])
                # Normalise to circle: show the direction each anchor pulls
                # in the trajectory plane (norm may be <1 since anchor may be
                # partly orthogonal to the plane).
                norms   = np.linalg.norm(raw, axis=1, keepdims=True)
                ancs_2d = raw / np.maximum(norms, 1e-8)
            else:
                ancs_2d = None

        return traj_2d, e1, e2, ancs_2d


# ── Softmax predictor ─────────────────────────────────────────────────────────

class SoftmaxWordPredictor(_BasePredictor):
    """Standard softmax transformer — used for comparison with Lohe models."""

    def __init__(self, model_key: str, config: dict, device_str: str = "cpu"):
        self.model_key = model_key
        self.config    = config
        self.device    = torch.device(device_str)
        self.d_osc     = 0        # sentinel: no oscillators
        demo_dir = os.path.dirname(os.path.abspath(__file__))
        mcfg = config["models"][model_key]
        self.d_model  = mcfg["d_model"]
        self.n_heads  = mcfg["n_heads"]
        self.n_layers = mcfg["n_layers"]
        self.label    = mcfg["label"]
        self.ppl      = mcfg["ppl"]
        self._load_vocab(demo_dir, mcfg)

        max_seq = mcfg.get("max_seq_len", 50)
        ckpt = os.path.join(demo_dir, mcfg["checkpoint"])
        self.model = SoftmaxLanguageTransformer(
            vocab_size=self.vocab_size, d_model=self.d_model,
            n_heads=self.n_heads, n_layers=self.n_layers,
            d_ff=self.d_model * 4, max_seq_len=max_seq, dropout=0.0)
        self.model.load_state_dict(
            torch.load(ckpt, map_location=self.device, weights_only=False))
        self.model.to(self.device).eval()

    @torch.no_grad()
    def get_fixed_points(self, token_ids: list):
        """
        No oscillators. x_star positions are evenly spaced on a circle
        (by sequence order) — used for visualisation only.
        Attention weights are real softmax values.
        """
        ids_t = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        T = len(token_ids)
        attentions = []
        h = self.model.pos_enc(self.model.embedding(ids_t))
        for layer in self.model.layers:
            h, attn = layer(h)
            attentions.append(attn[0].cpu().numpy())   # (H, T, T)
        logits = self.model.head(self.model.norm(h))[0, -1, :]

        # Place tokens evenly around the circle (position-based, not phase-based)
        angles = np.linspace(0, 2 * np.pi, T, endpoint=False)
        x_star = np.stack([np.cos(angles), np.sin(angles)], axis=-1)  # (T, 2)
        return x_star, np.array(attentions), logits.cpu()

    def get_anchors_2d(self, T: int):
        return None   # softmax has no anchor vectors

    def pca_project(self, x_star: np.ndarray) -> np.ndarray:
        return x_star  # already 2D


# ── Factory ───────────────────────────────────────────────────────────────────

def make_predictor(model_key: str, config: dict,
                   device_str: str = "cpu"):
    mcfg = config["models"][model_key]
    if mcfg.get("d_osc", 0) == 0:
        return SoftmaxWordPredictor(model_key, config, device_str)
    return KuramotoWordPredictor(model_key, config, device_str)
