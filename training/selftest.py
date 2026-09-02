"""Parity self-tests: derived modules must match the stock oscillator_attention modules exactly.

Run:  python -m training.selftest    (from repo root)
"""
import torch

from . import paths
paths.ensure_paths()

from oscillator_attention.transformer import KWSTransformer
from oscillator_attention.attention import LoheAttention
from oscillator_attention.kws_pe_model import KWSTransformerPE
from oscillator_attention.coupling_variants import LoheAttentionSigma


def test_kws_pe_none_parity():
    kw = dict(n_feats=40, d_model=32, n_heads=2, n_layers=1, n_classes=10,
              T=49, attn_type='lohe', p=1, d_osc=2, dropout=0.1)
    torch.manual_seed(123)
    stock = KWSTransformer(**kw)
    torch.manual_seed(123)
    mine = KWSTransformerPE(pe='none', **kw)

    sk = stock.state_dict(); mk = mine.state_dict()
    assert set(sk) == set(mk), (set(sk) ^ set(mk))
    for k in sk:
        assert torch.equal(sk[k], mk[k]), f"param mismatch: {k}"

    stock.eval(); mine.eval()
    x = torch.randn(4, 49, 40)
    with torch.no_grad():
        assert torch.allclose(stock(x), mine(x), atol=1e-6)
    print("OK  KWSTransformerPE(pe='none') == stock KWSTransformer "
          f"({sum(p.numel() for p in mine.parameters())} params)")


def test_sigma_softplus_parity():
    kw = dict(d_model=32, n_heads=2, d_head=16, d_osc=2, max_seq_len=49,
              p=1, causal=False)
    torch.manual_seed(7)
    stock = LoheAttention(**kw)
    torch.manual_seed(7)
    mine = LoheAttentionSigma(sigma='softplus', **kw)
    sk = stock.state_dict(); mk = mine.state_dict()
    assert set(sk) == set(mk)
    for k in sk:
        assert torch.equal(sk[k], mk[k]), k
    x = torch.randn(3, 20, 32)
    stock.eval(); mine.eval()
    with torch.no_grad():
        o1, a1 = stock(x); o2, a2 = mine(x)
    assert torch.allclose(o1, o2, atol=1e-6) and torch.allclose(a1, a2, atol=1e-6)
    print("OK  LoheAttentionSigma(sigma='softplus') == stock LoheAttention")


def test_pe_variants_build():
    for pe in ('sinusoidal', 'learned_abs'):
        m = KWSTransformerPE(pe=pe)
        x = torch.randn(2, 49, 40)
        y = m(x)
        assert y.shape == (2, 10)
        print(f"OK  KWSTransformerPE(pe={pe!r}) forward -> {tuple(y.shape)}, "
              f"{sum(p.numel() for p in m.parameters())} params")


def test_sigma_variants_build():
    for s in ('relu_eps', 'elu1'):
        m = LoheAttentionSigma(32, 2, 16, 2, 49, causal=False, sigma=s)
        o, a = m(torch.randn(2, 20, 32))
        assert o.shape == (2, 20, 32)
        # attn rows sum to 1
        assert torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=1e-5)
        print(f"OK  LoheAttentionSigma(sigma={s!r}) forward + row-stochastic")


if __name__ == "__main__":
    test_kws_pe_none_parity()
    test_sigma_softplus_parity()
    test_pe_variants_build()
    test_sigma_variants_build()
    print("\nALL PARITY TESTS PASSED")
