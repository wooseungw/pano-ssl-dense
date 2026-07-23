import os
os.environ.setdefault("DEV", "cpu")

import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pytest
import torch

import probe_seg_dinov3 as P

with mock.patch.object(sys, "argv", [sys.argv[0]]):
    import adaptive_field
    import fair_indoor
    import field_res_sweep
    import sweep_backbones
    import sweep_backbones_ext
    import sweep_fov

assert P.DEVICE == "cpu"

P.configure("structured3d")
assert (P.N_CLASS, P.IGNORE) == (41, 0)

GOLDEN = {
    "field_res_sweep.probe.steps800": 0.07017543859649122,
    "sweep_backbones.head.steps800": 0.07017543859649122,
    "sweep_backbones.head.steps40": 0.0,
    "sweep_fov.seeded_head.steps800": (0.07017543859649122, 0.0833333358168602, 19),
    "sweep_fov.seeded_head.steps40": (0.0, 0.0, 19),
    "fair_indoor.head.steps800": 0.07017543859649122,
    "fair_indoor.head.steps40": 0.0,
    "adaptive_field.probe.steps800": 0.07017543859649122,
    "sweep_backbones_ext.head.steps800": 0.07017543859649122,
    "sweep_backbones_ext.head.steps40": 0.0,
}


def fixed_inputs():
    g = torch.Generator().manual_seed(1234)
    Xtr = torch.randn(48, 8, generator=g)
    ytr = torch.randint(0, P.N_CLASS, (48,), generator=g)
    Xva = torch.randn(32, 8, generator=g)
    yva = torch.randint(0, P.N_CLASS, (32,), generator=g)
    ytr[::11] = P.IGNORE
    yva[::7] = P.IGNORE
    return Xtr, ytr, Xva, yva


CASES = [
    ("field_res_sweep.probe.steps800", field_res_sweep.probe, {}),
    ("sweep_backbones.head.steps800", sweep_backbones.head, {}),
    ("sweep_backbones.head.steps40", sweep_backbones.head, {"steps": 40}),
    ("sweep_fov.seeded_head.steps800", sweep_fov.seeded_head, {}),
    ("sweep_fov.seeded_head.steps40", sweep_fov.seeded_head, {"steps": 40}),
    ("fair_indoor.head.steps800", fair_indoor.head, {}),
    ("fair_indoor.head.steps40", fair_indoor.head, {"steps": 40}),
    ("adaptive_field.probe.steps800", adaptive_field.probe, {}),
    ("sweep_backbones_ext.head.steps800", sweep_backbones_ext.head, {}),
    ("sweep_backbones_ext.head.steps40", sweep_backbones_ext.head, {"steps": 40}),
]


@pytest.mark.parametrize("name,caller,kwargs", CASES)
def test_probe_caller_golden(name, caller, kwargs):
    actual = caller(*fixed_inputs(), **kwargs)
    expected = GOLDEN[name]

    assert type(actual) is type(expected)
    if isinstance(expected, tuple):
        assert actual[:2] == pytest.approx(expected[:2], rel=1e-6)
        assert actual[2] == expected[2]
    else:
        assert actual == pytest.approx(expected, rel=1e-6)
