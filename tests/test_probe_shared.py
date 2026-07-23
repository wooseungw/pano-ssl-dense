import os
os.environ.setdefault("DEV", "cpu")

import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch

import probe_seg_dinov3 as P

assert P.DEVICE == "cpu"

P.configure("structured3d")
assert (P.N_CLASS, P.IGNORE) == (41, 0)


def fixed_inputs():
    g = torch.Generator().manual_seed(5678)
    Xtr = torch.randn(40, 8, generator=g)
    ytr = torch.randint(1, 6, (40,), generator=g)
    Xva = torch.randn(24, 8, generator=g)
    yva = torch.randint(1, 6, (24,), generator=g)
    ytr[::9] = P.IGNORE
    yva[::7] = P.IGNORE
    return Xtr, ytr, Xva, yva


def test_linear_probe_is_deterministic_for_same_pre_call_seed():
    inputs = fixed_inputs()
    torch.manual_seed(77)
    first = P.linear_probe(*inputs, steps=40)
    torch.manual_seed(77)
    second = P.linear_probe(*inputs, steps=40)

    assert first == second


def test_ignore_labels_do_not_affect_training_and_absent_classes_are_excluded():
    Xtr, ytr, Xva, yva = fixed_inputs()
    changed = Xtr.clone()
    changed[ytr == P.IGNORE] += 1000.0

    torch.manual_seed(91)
    baseline = P.linear_probe(Xtr, ytr, Xva, yva, steps=40)
    torch.manual_seed(91)
    ignored_rows_changed = P.linear_probe(changed, ytr, Xva, yva, steps=40)

    assert baseline == ignored_rows_changed
    assert P.miou_acc(torch.tensor([2, 1, 2]), torch.tensor([P.IGNORE, 1, 1])) == (0.5, 0.5, 1)


def test_linear_probe_return_shape_and_types():
    torch.manual_seed(12)
    result = P.linear_probe(*fixed_inputs(), steps=1)

    assert isinstance(result, tuple)
    assert len(result) == 3
    assert type(result[0]) is float
    assert type(result[1]) is float
    assert type(result[2]) is int


def test_linear_probe_does_not_reseed_internally():
    inputs = fixed_inputs()
    torch.manual_seed(29)
    expected = P.linear_probe(*inputs, steps=20)

    torch.manual_seed(29)
    with mock.patch.object(P.torch, "manual_seed", side_effect=AssertionError("unexpected reseed")):
        actual = P.linear_probe(*inputs, steps=20)

    assert actual == expected
