"""Unit tests for F-3 utilities: EMA adapter update + differentiable field scatter."""

from __future__ import annotations

import torch
import torch.nn as nn

from encoder import ema_update
from fusion import scatter_mean_field


def _pair():
    torch.manual_seed(0)
    s = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    t = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    t.load_state_dict(s.state_dict())
    return s, t


def test_ema_moves_teacher_toward_student():
    s, t = _pair()
    with torch.no_grad():
        s[0].weight.add_(1.0)                            # student moved by training
    before = t[0].weight.clone()
    ema_update(s, t, momentum=0.9)
    expect = 0.9 * before + 0.1 * s[0].weight
    assert torch.allclose(t[0].weight, expect, atol=1e-6)


def test_ema_skips_frozen_student_params():
    s, t = _pair()
    s[1].weight.requires_grad_(False)                    # frozen on the student side
    with torch.no_grad():
        s[0].weight.add_(2.0); s[1].weight.add_(2.0)
    frozen_before = t[1].weight.clone()
    ema_update(s, t, momentum=0.5)
    assert torch.allclose(t[1].weight, frozen_before), "frozen params must not be EMA'd"
    assert not torch.allclose(t[0].weight, s[0].weight)  # partial move only


def test_scatter_mean_field_values_and_grad():
    cids = torch.tensor([0, 0, 2])
    feats = torch.tensor([[2.0, 0.0], [4.0, 2.0], [1.0, 1.0]], requires_grad=True)
    field, counts = scatter_mean_field(cids, feats, ncell=3)
    assert torch.allclose(field[0], torch.tensor([3.0, 1.0]))       # mean of two
    assert torch.allclose(field[1], torch.zeros(2))                 # uncovered
    assert torch.allclose(field[2], torch.tensor([1.0, 1.0]))
    assert counts.tolist() == [2.0, 0.0, 1.0]
    field.sum().backward()
    assert feats.grad is not None and torch.isfinite(feats.grad).all()
