"""Unit tests for Term B (cross-view masked completion): predictor + loss."""
import torch

from encoder import CrossViewPredictor
from losses import cross_view_completion_loss


def test_predictor_starts_as_identity_on_b_evidence():
    # zero-init output layer => at step 0 the prediction == the naive cross-view baseline (b_ev).
    torch.manual_seed(0)
    P = CrossViewPredictor(dim=32)
    a_ctx, b_ev = torch.randn(10, 32), torch.randn(10, 32)
    out = P(a_ctx, b_ev)
    assert out.shape == (10, 32)
    assert torch.allclose(out, b_ev, atol=1e-6), "zero-init predictor must return b_ev exactly"


def test_predictor_learns_a_correction():
    # zero-init LAST layer is the learning entry point -> it must receive gradient; earlier layers
    # get gradient once it moves off zero (standard zero-init residual). And b_ev gets gradient via
    # the residual identity path from step 1, so the encoder is trained immediately.
    P = CrossViewPredictor(dim=16)
    a_ctx = torch.randn(8, 16)
    b_ev = torch.randn(8, 16, requires_grad=True)
    out = P(a_ctx, b_ev)
    out.pow(2).mean().backward()
    assert P.net[-1].weight.grad.abs().sum() > 0, "last layer (learning entry) must get gradient"
    assert b_ev.grad.abs().sum() > 0, "encoder (b_ev) must get gradient via the residual path"


def test_completion_loss_zero_when_aligned():
    x = torch.randn(12, 24)
    w = torch.ones(12)
    assert cross_view_completion_loss(x, x.clone(), w).item() < 1e-5


def test_completion_loss_one_when_orthogonal():
    d = 4
    pred = torch.tensor([[1.0, 0, 0, 0]] * 5)
    tgt = torch.tensor([[0.0, 1, 0, 0]] * 5)
    w = torch.ones(5)
    assert abs(cross_view_completion_loss(pred, tgt, w).item() - 1.0) < 1e-5


def test_completion_loss_weight_normalized():
    # weighting by a constant c must not change the (weighted-mean) value.
    pred, tgt = torch.randn(20, 8), torch.randn(20, 8)
    w = torch.rand(20) + 0.1
    a = cross_view_completion_loss(pred, tgt, w)
    b = cross_view_completion_loss(pred, tgt, w * 3.0)
    assert torch.allclose(a, b, atol=1e-5)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
