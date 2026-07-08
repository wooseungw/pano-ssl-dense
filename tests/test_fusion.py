"""Unit tests for the F-2 set-fusion module (fusion.py)."""

from __future__ import annotations

import torch

from fusion import SetFusion, masked_mean, pack_sets


def test_masked_mean_ignores_padding():
    x = torch.zeros(2, 3, 4)
    x[0, 0] = 2.0; x[0, 1] = 4.0; x[0, 2] = 99.0        # slot 2 is padding
    x[1, 0] = 6.0
    mask = torch.tensor([[True, True, False], [True, False, False]])
    out = masked_mean(x, mask)
    assert torch.allclose(out[0], torch.full((4,), 3.0))
    assert torch.allclose(out[1], torch.full((4,), 6.0))


def test_setfusion_equals_mean_at_init():
    """Zero-init residual: at init the module IS the uniform-mean baseline."""
    torch.manual_seed(0)
    fu = SetFusion(dim=16, geo_dim=4, d_model=32, n_layers=1, n_heads=2).eval()
    f = torch.randn(5, 6, 16)
    g = torch.randn(5, 6, 4)
    mask = torch.rand(5, 6) > 0.3
    mask[:, 0] = True                                    # >=1 valid per row
    with torch.no_grad():
        out = fu(f, g, mask)
    assert torch.allclose(out, masked_mean(f, mask), atol=1e-5)


def test_setfusion_padding_invariance_after_training_perturbation():
    """Padded slots must never influence the output, even with nonzero weights."""
    torch.manual_seed(1)
    fu = SetFusion(dim=8, geo_dim=4, d_model=16, n_layers=1, n_heads=2).eval()
    with torch.no_grad():                                # de-zero the residual path
        fu.out.weight.normal_(); fu.out.bias.normal_()
    f = torch.randn(3, 5, 8); g = torch.randn(3, 5, 4)
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.bool)
    f2, g2 = f.clone(), g.clone()
    f2[~mask] = 123.0; g2[~mask] = -55.0                 # scribble on padding
    with torch.no_grad():
        a, b = fu(f, g, mask), fu(f2, g2, mask)
    assert torch.allclose(a, b, atol=1e-5), "padding leaked into the fused output"


def test_pack_sets_groups_and_caps():
    cid = torch.tensor([0, 2, 0, 2, 2, 1])
    feats = torch.arange(6, dtype=torch.float32).unsqueeze(1).repeat(1, 3)
    geo = torch.arange(6, dtype=torch.float32).unsqueeze(1)
    f, g, m = pack_sets(cid, feats, geo, ncell=3, max_cov=2)
    assert m.sum().item() == 5                           # cell2 had 3, capped to 2
    assert set(f[0, m[0], 0].tolist()) == {0.0, 2.0}     # cell 0 got rows 0,2
    assert set(f[1, m[1], 0].tolist()) == {5.0}          # cell 1 got row 5
    assert m[2].sum().item() == 2                        # cell 2 capped
    assert torch.equal(g[0, m[0]].squeeze(-1), f[0, m[0], 0])
