"""Unit tests for the consolidated 3-role VICReg loss (split API) + Expander."""

from __future__ import annotations

import torch

import losses as L
from encoder import Expander, GeometryHead, GlobalExpander, SubtokenExpander


def _identity_grid(gh: int, gw: int) -> torch.Tensor:
    ys = torch.linspace(-1 + 1.0 / gh, 1 - 1.0 / gh, gh)
    xs = torch.linspace(-1 + 1.0 / gw, 1 - 1.0 / gw, gw)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)


def _grid_valid_weight(gh, gw):
    return _identity_grid(gh, gw), torch.ones(gh * gw, dtype=torch.bool), torch.ones(gh * gw)


def test_invariance_zero_on_identical_views():
    torch.manual_seed(0)
    z = torch.randn(2, 8, 8, 8)
    inv = L.overlap_invariance(z, z.clone(), *_grid_valid_weight(8, 8))
    assert inv.item() < 1e-3, f"identical views -> ~0 invariance, got {inv.item()}"


def test_invariance_channel_mean_scale():
    """MEAN over P channels (canonical F.mse_loss scale), NOT sum — so it can't outweigh
    var/cov by a factor of P (the bug the smoke exposed)."""
    torch.manual_seed(1)
    za, zb = torch.randn(1, 256, 8, 8), torch.randn(1, 256, 8, 8)
    inv = L.overlap_invariance(za, zb, *_grid_valid_weight(8, 8))
    # two independent unit-Gaussians differ by ~2 per element in MSE; must be O(1), not O(256)
    assert 0.5 < inv.item() < 6.0, f"invariance must be O(1) per-channel-mean, got {inv.item()}"


def test_variance_fires_on_collapse_active_target():
    collapsed = torch.ones(2, 16, 8, 8) * 0.3
    var_c, _ = L.vicreg_vc(collapsed, gamma=1.0)
    var_d, _ = L.vicreg_vc(torch.randn(2, 16, 8, 8), gamma=1.0)
    assert var_c.item() > 0.9, f"collapse must fire variance ~gamma, got {var_c.item()}"
    assert var_d.item() < var_c.item()


def test_covariance_detects_rank_contraction():
    base = torch.randn(2, 1, 8, 8)
    rank1 = base.repeat(1, 16, 1, 1) + 0.01 * torch.randn(2, 16, 8, 8)
    _, cov_r = L.vicreg_vc(rank1)
    _, cov_i = L.vicreg_vc(torch.randn(2, 16, 8, 8))
    assert cov_r.item() > cov_i.item(), "rank-contracted features must raise covariance"


def test_vc_over_full_stack_not_single_tile():
    """var/cov must be callable over a whole pano's tile stack (B=T), not just one tile."""
    z = torch.randn(12, 32, 8, 8)                        # 12 tiles at once
    var, cov = L.vicreg_vc(z, gamma=1.0)
    assert torch.isfinite(var) and torch.isfinite(cov)


def test_roles_backprop_together():
    torch.manual_seed(2)
    z = torch.randn(4, 8, 6, 6, requires_grad=True)      # 4 tiles
    inv = L.overlap_invariance(z[0:1], z[1:2], *_grid_valid_weight(6, 6))
    var, cov = L.vicreg_vc(z)
    (25 * inv + 25 * var + cov).backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_expander_shape_and_backbone_gradient():
    exp = Expander(dim=32, proj_dim=24, hidden=48).train()
    feat = torch.randn(3, 32, 8, 8, requires_grad=True)
    z = exp(feat)
    assert z.shape == (3, 24, 8, 8)
    z.sum().backward()
    assert feat.grad is not None


def test_vector_vicreg_and_pair_invariance():
    torch.manual_seed(3)
    a = torch.randn(16, 32, requires_grad=True)
    b = a.detach().clone()
    inv = L.vicreg_pair_invariance(a, b)
    var, cov = L.vicreg_vc_vectors(torch.cat([a, b], dim=0))
    assert inv.item() == 0.0 and torch.isfinite(var) and torch.isfinite(cov)
    (inv + var + cov).backward()
    assert a.grad is not None


def test_global_subtoken_and_geometry_heads():
    torch.manual_seed(4)
    feat = torch.randn(6, 32, 8, 8, requires_grad=True)
    glob = GlobalExpander(32, proj_dim=24, hidden=48)(feat)
    sub = SubtokenExpander(32, proj_dim=20, hidden=32)(feat)
    geo = GeometryHead(32)(feat)
    assert glob.shape == (6, 24)
    assert sub.shape == (6, 20, 16, 16)
    assert geo.shape == (6, 4, 8, 8)
    (glob.mean() + sub.mean() + geo.mean()).backward()
    assert feat.grad is not None and torch.isfinite(feat.grad).all()
