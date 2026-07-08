"""Unit tests for the runs/ experiment-folder convention (runlog.py)."""

from __future__ import annotations

import json
import os

import numpy as np

import runlog


def test_create_run_layout(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS", str(tmp_path))
    run = runlog.create_run("smoke_test", {"lr": 1e-4, "note": "한글 ok"})
    assert os.path.isdir(os.path.join(run, "weights"))
    assert os.path.isdir(os.path.join(run, "viz"))
    cfg = json.load(open(os.path.join(run, "config.json")))
    assert cfg["lr"] == 1e-4 and cfg["note"] == "한글 ok"
    assert os.path.basename(run).endswith("_smoke_test")


def test_spread_indices_picks_representative_samples():
    assert runlog.spread_indices(0) == []
    assert runlog.spread_indices(2) == [0, 1]              # n <= k -> all
    assert runlog.spread_indices(3) == [0, 1, 2]
    assert runlog.spread_indices(10, 3) == [0, 4, 9]       # endpoints + middle, not [0,1,2]
    assert runlog.spread_indices(100, 4) == [0, 33, 66, 99]


def test_save_integration_figure_writes_one_png(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS", str(tmp_path))
    run = runlog.create_run("integ_test", {})
    pal = runlog.seg_palette(5)
    rgb = np.random.rand(16, 32, 3)
    seg = runlog.colorize(np.random.randint(0, 5, (16, 32)), pal)     # pre-colorized "field"
    count = np.random.randint(0, 4, (8, 16))                          # 2D -> turbo + colorbar
    out = runlog.save_integration_figure(
        run, "s2d3d", 1,
        [("input", rgb, "rgb"), ("footprint", seg, "field"),
         ("overlap", count, "count"), ("stitched", seg, "field")],
        suptitle="INTEGRATION test")
    assert out == os.path.join(run, "viz", "s2d3d_s1_integration.png")
    assert os.path.exists(out)
    # fname override (viz_stitch_demo keeps its historical filename)
    out2 = runlog.save_integration_figure(run, "x", 0, [("a", rgb, "rgb")], fname="custom.png")
    assert os.path.basename(out2) == "custom.png" and os.path.exists(out2)


def test_save_seg_sample_writes_comparable_pngs(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS", str(tmp_path))
    run = runlog.create_run("viz_test", {})
    pal = runlog.seg_palette(5)
    rgb = np.random.rand(64, 128, 3)
    gt = np.random.randint(0, 5, (8, 16))
    pred = np.full((8, 16), -1)                          # uncovered cells -> gray
    runlog.save_seg_sample(run, "s3d", 0, rgb, gt, {"mean": pred}, pal, scale=4)
    viz = os.path.join(run, "viz")
    names = sorted(os.listdir(viz))
    assert names == ["s3d_s0_gt.png", "s3d_s0_input.png", "s3d_s0_pred_mean.png"]
    from PIL import Image
    assert Image.open(os.path.join(viz, "s3d_s0_gt.png")).size == (16 * 4, 8 * 4)
