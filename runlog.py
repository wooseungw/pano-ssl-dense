"""Experiment run-folder convention (user-specified, 2026-07-02).

runs/<YYYYMMDD>_<slug>/
  config.json      hyperparameters / env snapshot of the run
  weights/         checkpoints
  viz/             downstream result PNGs, arranged WITH GT in the same folder so
                   they compare side-by-side: <dataset>_s<idx>_{input,gt,pred_<tag>}.png
                   — 1..3 FIXED (deterministically designated) val samples per dataset.
"""
from __future__ import annotations

import datetime
import json
import os
from typing import Dict

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "runs")


def create_run(slug: str, config: Dict) -> str:
    """Create runs/<MMDD_HHMM>_<slug>/ with config.json + weights/ + viz/. Returns path."""
    stamp = datetime.datetime.now().strftime("%m%d_%H%M")   # no year; date + time to the minute
    run = os.path.join(RUNS, f"{stamp}_{slug}")
    os.makedirs(os.path.join(run, "weights"), exist_ok=True)
    os.makedirs(os.path.join(run, "viz"), exist_ok=True)
    with open(os.path.join(run, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2, default=str, ensure_ascii=False)
    return run


def seg_palette(n_class: int) -> np.ndarray:
    """Deterministic categorical palette; class 0 (void/ignore) = black."""
    rng = np.random.RandomState(0)
    pal = rng.uniform(0.15, 0.95, size=(n_class, 3))
    pal[0] = 0.0
    return pal


def colorize(grid: np.ndarray, pal: np.ndarray) -> np.ndarray:
    """Int class grid (-1 = uncovered -> gray) -> float [0,1] HWC image."""
    img = pal[np.clip(grid, 0, len(pal) - 1)]
    img[grid < 0] = 0.25
    return img


def _save_png(img01: np.ndarray, path: str) -> None:
    Image.fromarray((np.clip(img01, 0.0, 1.0) * 255).astype(np.uint8)).save(path)


def save_panel(run: str, dataset: str, idx: int, panels, dpi: int = 110) -> None:
    """One side-by-side comparison figure of (title, HWC float [0,1]) panels
    -> viz/<dataset>_s<idx>_compare.png. Complements the individual PNGs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.3 * n))
    for ax, (title, img) in zip(np.atleast_1d(axes), panels):
        ax.imshow(np.clip(img, 0, 1), interpolation="nearest", aspect="auto")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(run, "viz", f"{dataset}_s{idx}_compare.png"), dpi=dpi)
    plt.close(fig)


def spread_indices(n: int, k: int = 3) -> list:
    """k indices spread across [0, n-1] (endpoints + evenly between), deduped & sorted.

    Picks representative val samples instead of the first k, which on grouped datasets
    (consecutive frames of one room/area) are often near-duplicates."""
    if n <= 0:
        return []
    if n <= k:
        return list(range(n))
    return sorted({int(round(i * (n - 1) / (k - 1))) for i in range(k)})


def save_integration_figure(run: str, name: str, idx: int, panels, suptitle: str = "",
                            dpi: int = 120, fname: str = None) -> str:
    """Stacked 'integration' figure mirroring viz_stitch_demo: one row per panel showing the
    per-tile -> overlap -> stitched-as-scored -> GT story at the metric eval resolution.

    panels: list of (title, img, kind).
      'rgb' / 'field' -> HWC float [0,1] image (caller pre-colorizes seg/depth/normal);
      'count'         -> a 2D array rendered with the turbo colormap + a colorbar.
    Written to viz/<name>_s<idx>_integration.png (or `fname` if given)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(11, 1.7 * n))
    for ax, (title, img, kind) in zip(np.atleast_1d(axes), panels):
        if kind == "count":
            im = ax.imshow(img, cmap="turbo", interpolation="nearest", aspect="auto")
            fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
        else:
            ax.imshow(np.clip(img, 0, 1), interpolation="nearest", aspect="auto")
        ax.set_title(title, fontsize=10, loc="left")
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = os.path.join(run, "viz", fname or f"{name}_s{idx}_integration.png")
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def _turbo(x01: np.ndarray) -> np.ndarray:
    """Cheap turbo-ish colormap for a [0,1] scalar field -> RGB (no matplotlib dep)."""
    x = np.clip(x01, 0, 1)
    r = np.clip(1.5 - abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], -1)


def save_depth_sample(run: str, dataset: str, idx: int, rgb01: np.ndarray,
                      gt_log: np.ndarray, preds_log: Dict[str, np.ndarray],
                      valid: np.ndarray, scale: int = 8) -> None:
    """Depth sample: input + GT + per-model pred as turbo PNGs (GT alongside preds).
    log-depth fields are min-max normalized over GT-valid cells for a shared color scale."""
    viz = os.path.join(run, "viz")
    up = np.ones((scale, scale, 1))
    v = valid.astype(bool)
    lo, hi = (gt_log[v].min(), gt_log[v].max()) if v.any() else (0.0, 1.0)
    rng = max(hi - lo, 1e-6)

    def col(field):
        img = _turbo((field - lo) / rng)
        img[~v] = 0.15
        return np.kron(img, up)
    _save_png(rgb01, os.path.join(viz, f"{dataset}_s{idx}_input.png"))
    _save_png(col(gt_log), os.path.join(viz, f"{dataset}_s{idx}_gt.png"))
    for tag, pl in preds_log.items():
        _save_png(col(pl), os.path.join(viz, f"{dataset}_s{idx}_pred_{tag}.png"))


def save_normal_sample(run: str, dataset: str, idx: int, rgb01: np.ndarray,
                       gt_n: np.ndarray, preds_n: Dict[str, np.ndarray],
                       valid: np.ndarray, scale: int = 8) -> None:
    """Normal sample: input + GT + per-model pred as (n+1)/2 RGB PNGs (GT alongside preds).
    gt_n/preds_n are (H,W,3) unit-normal fields; invalid pixels rendered gray."""
    viz = os.path.join(run, "viz")
    up = np.ones((scale, scale, 1))
    v = valid.astype(bool)

    def col(n: np.ndarray) -> np.ndarray:
        img = np.clip((n + 1.0) / 2.0, 0.0, 1.0)
        img[~v] = 0.15
        return np.kron(img, up)
    _save_png(rgb01, os.path.join(viz, f"{dataset}_s{idx}_input.png"))
    _save_png(col(gt_n), os.path.join(viz, f"{dataset}_s{idx}_gt.png"))
    for tag, pn in preds_n.items():
        _save_png(col(pn), os.path.join(viz, f"{dataset}_s{idx}_pred_{tag}.png"))


def save_seg_sample(run: str, dataset: str, idx: int, rgb01: np.ndarray,
                    gt_grid: np.ndarray, preds: Dict[str, np.ndarray],
                    pal: np.ndarray, scale: int = 8) -> None:
    """One designated val sample: input + GT + one PNG per prediction tag, all in viz/.

    rgb01 (H,W,3) float [0,1]; gt_grid/preds (h,w) int class grids (cell resolution,
    upsampled x`scale` with nearest for visibility).
    """
    viz = os.path.join(run, "viz")
    up = np.ones((scale, scale, 1))
    _save_png(rgb01, os.path.join(viz, f"{dataset}_s{idx}_input.png"))
    _save_png(np.kron(colorize(gt_grid, pal), up), os.path.join(viz, f"{dataset}_s{idx}_gt.png"))
    for tag, grid in preds.items():
        _save_png(np.kron(colorize(grid, pal), up),
                  os.path.join(viz, f"{dataset}_s{idx}_pred_{tag}.png"))
