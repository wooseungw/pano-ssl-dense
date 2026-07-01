"""Unit tests for the SSL losses — focused on the failure modes the review flagged."""

from __future__ import annotations

import torch

import losses as L


def _identity_grid(gh: int, gw: int) -> torch.Tensor:
    ys = torch.linspace(-1 + 1.0 / gh, 1 - 1.0 / gh, gh)
    xs = torch.linspace(-1 + 1.0 / gw, 1 - 1.0 / gw, gw)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)


def test_warp_loss_zero_on_identity_correspondence():
    """If B == A and the grid is identity, the equivariance loss must be ~0."""
    torch.manual_seed(0)
    feat = torch.randn(2, 16, 8, 8)
    grid = _identity_grid(8, 8)
    valid = torch.ones(64, dtype=torch.bool)
    weight = torch.ones(64)
    loss = L.warp_equivariance_loss(feat, feat.clone(), grid, valid, weight)
    assert loss.item() < 1e-3, f"identity warp loss should be ~0, got {loss.item()}"


def test_warp_loss_high_on_mismatch():
    torch.manual_seed(1)
    fa = torch.randn(2, 16, 8, 8)
    fb = torch.randn(2, 16, 8, 8)
    grid = _identity_grid(8, 8)
    valid = torch.ones(64, dtype=torch.bool)
    weight = torch.ones(64)
    loss = L.warp_equivariance_loss(fa, fb, grid, valid, weight)
    assert loss.item() > 0.5, f"random-vs-random cosine loss should be ~1, got {loss.item()}"


def test_vicreg_fires_on_collapse():
    """Constant (collapsed) features -> large variance loss; diverse -> small."""
    collapsed = torch.ones(2, 32, 8, 8) * 0.3
    var_c, cov_c = L.vicreg_var_cov(collapsed)
    diverse = torch.randn(2, 32, 8, 8)
    var_d, cov_d = L.vicreg_var_cov(diverse)
    assert var_c.item() > 0.9, f"collapse should give variance_loss ~gamma, got {var_c.item()}"
    assert var_d.item() < var_c.item()


def test_vicreg_covariance_detects_redundancy():
    """Perfectly correlated dims (rank-1) -> high covariance loss vs decorrelated."""
    base = torch.randn(2, 1, 8, 8)
    redundant = base.repeat(1, 32, 1, 1) + 0.01 * torch.randn(2, 32, 8, 8)
    _, cov_r = L.vicreg_var_cov(redundant)
    _, cov_i = L.vicreg_var_cov(torch.randn(2, 32, 8, 8))
    assert cov_r.item() > cov_i.item(), "redundant dims should have higher covariance loss"


def test_distill_zero_on_identity():
    feat = torch.randn(2, 24, 6, 6)
    token, rel = L.distill_loss(feat, feat.clone())
    assert token.item() < 1e-5 and rel.item() < 1e-5


def test_combined_loss_runs_and_backprops():
    torch.manual_seed(2)
    sa = torch.randn(2, 16, 8, 8, requires_grad=True)
    sb = torch.randn(2, 16, 8, 8, requires_grad=True)
    ta = torch.randn(2, 16, 8, 8)
    tb = torch.randn(2, 16, 8, 8)
    grid = _identity_grid(8, 8)
    valid = torch.ones(64, dtype=torch.bool)
    weight = torch.ones(64)
    total, comps = L.combined_loss(sa, sb, ta, tb, (grid, valid, weight), w_warp=0.5)
    total.backward()
    assert torch.isfinite(total) and sa.grad is not None
    assert set(comps) >= {"warp", "distill_token", "vic_var", "vic_cov", "total"}
