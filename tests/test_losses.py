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


def test_sinkhorn_rows_sum_to_one_and_balanced():
    """Q rows are distributions; column (prototype) usage is balanced ~1/K."""
    torch.manual_seed(3)
    scores = torch.randn(256, 16)
    q = L.sinkhorn(scores)
    assert torch.allclose(q.sum(1), torch.ones(256), atol=1e-3)
    usage = q.mean(0)
    # 3-iter sinkhorn is APPROXIMATELY balanced (SwAV standard): what matters for
    # anti-collapse is that no prototype hoards or fully starves, not exact 1/K.
    assert usage.max().item() < 3.0 / 16, f"prototype hoards mass: {usage.max().item()}"
    assert usage.min().item() > 1.0 / (8.0 * 16), f"prototype starves: {usage.min().item()}"


def test_sinkhorn_shift_invariant():
    """A global score offset must not change the assignment (max-shift safety)."""
    torch.manual_seed(4)
    scores = torch.randn(64, 8)
    assert torch.allclose(L.sinkhorn(scores), L.sinkhorn(scores + 100.0), atol=1e-4)


def _onehot_codes(classes: torch.Tensor, k: int, scale: float = 10.0) -> torch.Tensor:
    """(gh,gw) int -> (1,K,gh,gw) strong one-hot logits."""
    gh, gw = classes.shape
    z = torch.zeros(1, k, gh, gw)
    z.scatter_(1, classes.view(1, 1, gh, gw), scale)
    return z


def test_code_swap_loss_low_on_matching_codes_high_on_mismatch():
    """Identical codes at identity correspondence -> ~0; shifted codes -> large."""
    torch.manual_seed(5)
    k, gh = 8, 8
    classes = torch.randint(0, k, (gh, gh))
    logits = _onehot_codes(classes, k)
    q_same = torch.softmax(_onehot_codes(classes, k) * 10.0, dim=1)      # ~one-hot target
    q_diff = torch.softmax(_onehot_codes((classes + 1) % k, k) * 10.0, dim=1)
    grid = _identity_grid(gh, gh)
    valid = torch.ones(gh * gh, dtype=torch.bool)
    weight = torch.ones(gh * gh)
    lo = L.code_swap_loss(logits, q_same, grid, valid, weight)
    hi = L.code_swap_loss(logits, q_diff, grid, valid, weight)
    assert lo.item() < 0.1, f"matching codes should give ~0 CE, got {lo.item()}"
    assert hi.item() > 10.0, f"mismatched codes should give large CE, got {hi.item()}"


def test_code_swap_loss_backprops_and_respects_stopgrad():
    """Gradient flows to the student logits, never to the target codes."""
    torch.manual_seed(6)
    logits = torch.randn(1, 8, 8, 8, requires_grad=True)
    q = torch.softmax(torch.randn(1, 8, 8, 8), dim=1).requires_grad_(True)
    grid = _identity_grid(8, 8)
    valid = torch.ones(64, dtype=torch.bool)
    weight = torch.ones(64)
    loss = L.code_swap_loss(logits, q, grid, valid, weight)
    loss.backward()
    assert torch.isfinite(loss) and logits.grad is not None
    assert q.grad is None, "target codes must be stop-grad"


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
