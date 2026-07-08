"""Unit tests for the depth/normal benchmark scaffold — pure geometry + metric functions
(no encoder / no dataset). Guards the load-bearing pieces the advisor flagged: metric
correctness, world-frame normal averaging, and stitch coverage (the coverage-collapse bug).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bench_common as B  # noqa: E402
import depth_s2d3d_bench as D  # noqa: E402
import normal_s2d3d_bench as N  # noqa: E402


# ------------------------------------------------------------------ fold splits
def test_area_num():
    f = "/x/extracted_data/area_5a/area_5a/pano/rgb/foo_rgb.png"
    assert B.area_num(f) == "5"


def test_fold_splits_disjoint():
    files = [f"/x/extracted_data/area_{a}/area_{a}/pano/rgb/f_rgb.png" for a in ("1", "2", "3", "4", "5a", "5b", "6")]
    for fold in (1, 2, 3):
        tr, va = B.split_files(files, fold)
        assert set(tr).isdisjoint(set(va)), "train/test leak"
        assert va, "empty test fold"
        assert {B.area_num(f) for f in va} == B.FOLD_TEST[fold]


# ------------------------------------------------------------------ depth metrics
def test_depth_perfect_prediction():
    gt = np.random.RandomState(0).uniform(0.5, 8.0, (16, 16)).astype(np.float32)
    valid = np.ones_like(gt, bool)
    r = D.finalize_depth(D.depth_pixel_stats(gt.copy(), gt, valid))
    assert r["AbsRel"] == pytest.approx(0.0, abs=1e-6)
    assert r["RMSE"] == pytest.approx(0.0, abs=1e-6)
    assert r["d1"] == pytest.approx(1.0)
    assert r["d1_SI"] == pytest.approx(1.0)


def test_depth_absrel_known_value():
    # constant 10% over-prediction -> AbsRel = 0.1, delta1 = 1 (1.1 < 1.25)
    gt = np.full((8, 8), 4.0, np.float32)
    pred = gt * 1.1
    valid = np.ones_like(gt, bool)
    r = D.finalize_depth(D.depth_pixel_stats(pred, gt, valid))
    assert r["AbsRel"] == pytest.approx(0.1, abs=1e-4)
    assert r["d1"] == pytest.approx(1.0)


def test_depth_delta1_threshold():
    # ratio exactly 1.3 > 1.25 -> outside delta1, inside delta2 (1.5625)
    gt = np.full((10, 10), 2.0, np.float32)
    pred = gt * 1.3
    valid = np.ones_like(gt, bool)
    r = D.finalize_depth(D.depth_pixel_stats(pred, gt, valid))
    assert r["d1"] == pytest.approx(0.0)
    assert r["d2"] == pytest.approx(1.0)


def test_depth_si_invariant_to_global_scale():
    # SI-delta1 must ignore a global scale error that metric delta1 penalizes.
    # Keep 2*gt under DEPTH_CAP so the cap clamp doesn't distort the SI alignment.
    gt = np.random.RandomState(1).uniform(0.5, 4.0, (20, 20)).astype(np.float32)
    pred = gt * 2.0                               # 2x global scale error
    valid = np.ones_like(gt, bool)
    r = D.finalize_depth(D.depth_pixel_stats(pred, gt, valid))
    assert r["d1"] < 0.5                           # metric hates the scale error
    assert r["d1_SI"] == pytest.approx(1.0)        # SI removes it -> perfect


def test_depth_aggregation_matches_single_pass():
    rng = np.random.RandomState(2)
    gt = rng.uniform(0.5, 8.0, (30, 30)).astype(np.float32)
    pred = gt * rng.uniform(0.8, 1.2, gt.shape).astype(np.float32)
    valid = np.ones_like(gt, bool)
    one = D.finalize_depth(D.depth_pixel_stats(pred, gt, valid))
    # split into two halves, aggregate the sums -> identical means
    agg = {k: 0.0 for k in D.STAT_KEYS}
    for sl in (slice(0, 15), slice(15, 30)):
        s = D.depth_pixel_stats(pred[sl], gt[sl], valid[sl])
        for k in D.STAT_KEYS:
            agg[k] += s[k]
    two = D.finalize_depth(agg)
    for k in ("AbsRel", "RMSE", "d1"):
        assert one[k] == pytest.approx(two[k], rel=1e-6)


def test_depth_unit_conversion():
    # raw 5120 (=10m at 1/512) is exactly at the cap -> valid; 5121 -> excluded
    for raw, want_valid in ((512, True), (5120, True), (5121, False), (0, False), (65535, False)):
        d_m = raw / D.DEPTH_SCALE
        valid = (raw > 0) and (raw < 65535) and (d_m <= D.DEPTH_CAP)
        assert valid == want_valid, f"raw={raw}"


# ------------------------------------------------------------------ normal metrics
def test_normal_perfect_prediction():
    rng = np.random.RandomState(0)
    n = rng.randn(12, 12, 3).astype(np.float32)
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    valid = np.ones(n.shape[:2], bool)
    s = N.normal_pixel_stats(n.reshape(-1, 3), n.reshape(-1, 3), valid.reshape(-1))
    r = N.finalize_normal(s, s["hist"])
    assert r["mean"] == pytest.approx(0.0, abs=0.05)   # arccos(≈1) float32 precision floor
    assert r["pct_11"] == pytest.approx(100.0)


def test_normal_known_angle():
    # every pred is 30 deg off its GT about a fixed axis -> mean≈median≈30, <30 excludes it (strict <)
    a = np.array([0.0, 0.0, 1.0])
    theta = np.radians(30.0)
    b = np.array([np.sin(theta), 0.0, np.cos(theta)])
    gt = np.tile(a, (100, 1)).astype(np.float32)
    pred = np.tile(b, (100, 1)).astype(np.float32)
    valid = np.ones(100, bool)
    s = N.normal_pixel_stats(pred, gt, valid)
    r = N.finalize_normal(s, s["hist"])
    assert r["mean"] == pytest.approx(30.0, abs=0.3)
    assert r["median"] == pytest.approx(30.0, abs=0.3)
    assert r["pct_22"] == pytest.approx(0.0)          # 30 !< 22.5
    assert r["pct_30"] == pytest.approx(0.0)          # strict < 30


def test_normal_flipped_is_180():
    gt = np.tile([0.0, 1.0, 0.0], (50, 1)).astype(np.float32)
    pred = -gt
    s = N.normal_pixel_stats(pred, gt, np.ones(50, bool))
    r = N.finalize_normal(s, s["hist"])
    assert r["mean"] == pytest.approx(180.0, abs=0.5)


def test_normal_median_from_histogram():
    # half at 10deg, half at 40deg -> median falls in the empty middle (~25deg bin center)
    a = np.array([0.0, 0.0, 1.0])
    def rot(deg):
        t = np.radians(deg)
        return np.array([np.sin(t), 0.0, np.cos(t)])
    gt = np.tile(a, (200, 1)).astype(np.float32)
    pred = np.vstack([np.tile(rot(10), (100, 1)), np.tile(rot(40), (100, 1))]).astype(np.float32)
    s = N.normal_pixel_stats(pred, gt, np.ones(200, bool))
    r = N.finalize_normal(s, s["hist"])
    assert 10.0 <= r["median"] <= 40.0
    assert r["mean"] == pytest.approx(25.0, abs=0.5)


# ------------------------------------------------------------------ stitch coverage (the collapse bug)
def _tiny_env(monkeypatch):
    """Shrink the stitch/eval grids so coord_map+stitch run fast on CPU."""
    for name, val in (("SH", 16), ("SW", 32), ("EH", 64), ("EW", 128), ("TILE_OUT", 64), ("DEVICE", "cpu")):
        monkeypatch.setattr(B, name, val)


def test_full_sphere_plan_covers_stitch_grid(monkeypatch):
    _tiny_env(monkeypatch)
    plan = B.build_plan()
    reached = torch.zeros(B.SH * B.SW, dtype=torch.bool)
    for tp in plan:
        reached[B.coord_map(tp.yaw_deg, tp.pitch_deg)] = True
    frac = reached.float().mean().item()
    assert frac > 0.95, f"full-sphere tiling left {(1-frac)*100:.0f}% of the stitch grid uncovered"


def test_stitch_field_shape_and_coverage(monkeypatch):
    _tiny_env(monkeypatch)
    plan = B.build_plan()
    cids = [B.coord_map(tp.yaw_deg, tp.pitch_deg) for tp in plan]
    d, gh, gw = 8, B.TILE // 16, B.TILE // 16
    feat = torch.zeros(len(plan), d, gh, gw)
    head = torch.nn.Conv2d(d, 3, 1)                    # cheap stand-in head, 32->? then upsample inside?
    # stitch_field upsamples head output to TILE_OUT internally; a 1x1 conv keeps (gh,gw) then interpolate.
    field, cov, covered = B.stitch_field(head, feat, cids, 3)
    assert field.shape == (3, B.EH, B.EW)
    assert covered.shape == (B.EH, B.EW)
    assert cov > 0.95
    assert covered.mean() > 0.95


def test_stitch_field_averages_constants(monkeypatch):
    _tiny_env(monkeypatch)
    plan = B.build_plan()
    cids = [B.coord_map(tp.yaw_deg, tp.pitch_deg) for tp in plan]
    d = 4
    feat = torch.zeros(len(plan), d, B.TILE // 16, B.TILE // 16)

    class Const(torch.nn.Module):                      # every pixel -> constant 2.5
        def forward(self, x):
            return torch.full((x.shape[0], 1, x.shape[2], x.shape[3]), 2.5)
    field, _, covered = B.stitch_field(Const(), feat, cids, 1)
    assert field[0][covered].mean().item() == pytest.approx(2.5, abs=1e-4)


class _Const(torch.nn.Module):
    def forward(self, x):
        return torch.full((x.shape[0], 1, x.shape[2], x.shape[3]), 2.5)


def test_stitch_field_covered_mask_drops_bleed_ring(monkeypatch):
    # A single tile covers only the top cell-row (ids 0..3); rows 1-3 stay uncovered (zero-filled).
    # The bilinear field upsample bleeds those zeros downward; the covered mask (>0.999) must keep
    # ONLY clean pixels so nothing scored carries the pulled-down value (the depth-board bias fix).
    for name, val in (("SH", 4), ("SW", 4), ("EH", 16), ("EW", 16), ("TILE_OUT", 4), ("DEVICE", "cpu")):
        monkeypatch.setattr(B, name, val)
    cids = [torch.tensor([0, 1, 2, 3] * 4, dtype=torch.long)]      # 16 tile px -> top-row cells
    feat = torch.zeros(1, 4, 8, 8)
    field, _, covered = B.stitch_field(_Const(), feat, cids, 1)
    assert covered.any(), "expected some clean covered pixels in the top row"
    vals = field[0].numpy()[covered]
    assert np.allclose(vals, 2.5, atol=1e-4), f"covered pixels contaminated by zero-bleed: {vals.min()}..{vals.max()}"
    assert not covered[-1].any(), "bottom row is uncovered/bled and must not be scored"


def test_train_head_patch_size_agnostic(monkeypatch):
    # A non-patch-16 backbone gives a different feature grid (here 9x9 -> DenseHead 36x36), but GT
    # tiles are HEAD_OUT=128. train_head must interpolate the head output, not crash on the mismatch.
    for name, val in (("HEAD_OUT", 128), ("CHUNK", 2), ("EPOCHS", 1), ("SEED", 0), ("DEVICE", "cpu")):
        monkeypatch.setattr(B, name, val)
    feat = torch.zeros(2, 8, 9, 9)                                  # 9x9 != 32x32 patch-16 grid
    d_tiles = [(torch.zeros(128, 128), torch.ones(128, 128, dtype=torch.bool)) for _ in range(2)]
    D.train_head(B.DenseHead(8, 1), [(feat, d_tiles)])             # must not raise
    n_tiles = [(torch.zeros(128, 128, 3), torch.ones(128, 128, dtype=torch.bool)) for _ in range(2)]
    N.train_head(B.DenseHead(8, 3), [(feat, n_tiles)])            # must not raise
