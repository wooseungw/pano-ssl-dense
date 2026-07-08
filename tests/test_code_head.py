"""Unit tests for the M1 CodeHead (projector + normalized prototypes)."""

from __future__ import annotations

import torch

from encoder import CodeHead


def test_code_head_shapes_and_cosine_scores():
    torch.manual_seed(0)
    head = CodeHead(dim=32, proj_dim=16, n_proto=24)
    head.normalize_prototypes()
    s = head(torch.randn(2, 32, 8, 8))
    assert s.shape == (2, 24, 8, 8)
    assert s.abs().max().item() <= 1.0 + 1e-4, "normalized z & prototypes -> cosine in [-1,1]"


def test_prototype_normalization_is_unit_norm():
    head = CodeHead(dim=16, proj_dim=8, n_proto=12)
    with torch.no_grad():
        head.prototypes.weight.mul_(3.7)                  # de-normalize
    head.normalize_prototypes()
    norms = head.prototypes.weight.norm(dim=1)
    assert torch.allclose(norms, torch.ones(12), atol=1e-5)


def test_code_head_backprops():
    head = CodeHead(dim=16, proj_dim=8, n_proto=12)
    feat = torch.randn(1, 16, 4, 4, requires_grad=True)
    head(feat).sum().backward()
    assert feat.grad is not None
    assert head.prototypes.weight.grad is not None
