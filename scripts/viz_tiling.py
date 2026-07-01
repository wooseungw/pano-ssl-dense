"""Visualize the E2P tiling: tile FOOTPRINTS on the ERP (image-independent geometry, exact via
render_coordmap) + a montage of the rendered perspective tiles, for the two production configs:
  indoor : band, hfov 65, 3 pitch rings (-45/0/45), overlap 0.25  -> ~22 tiles
  outdoor: hfov 50, single equator ring, overlap 0.25             -> ~10 tiles

Run: CUDA_VISIBLE_DEVICES= conda run --no-capture-output -n pano python scripts/viz_tiling.py
Out: docs/figures/viz_tiling/tiling.png
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import anyres_e2p as a2p  # noqa: E402
import geometry as G  # noqa: E402
import data  # noqa: E402

ERP_W, ERP_H = 2048, 1024
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs/figures/viz_tiling")


def indoor_plan():
    return a2p.plan_tiles("band", 65.0, 65.0, 0.25, pmax_deg=45.0)


def outdoor_plan():
    yaws = a2p.make_yaw_centers_closed_loop(50.0, 0.25, start_deg=-180.0)
    return [a2p.TilePlan(y, 0.0) for y in yaws]


def _seam_split(x, y):
    """Insert NaN where a footprint border wraps the +/-180 seam, so plot() doesn't draw a
    spurious horizontal line across the whole ERP."""
    x = x.astype(float); y = y.astype(float)
    jump = np.where(np.abs(np.diff(x)) > ERP_W / 2)[0]
    return np.insert(x, jump + 1, np.nan), np.insert(y, jump + 1, np.nan)


def footprint_outline(ax, plan, hfov, out=48):
    """Exact footprints via coordmaps; draw each tile's BORDER (colored by pitch ring) so the
    overlap structure is legible, + the tile center."""
    rings = sorted({round(tp.pitch_deg, 1) for tp in plan})
    colors = plt.cm.rainbow(np.linspace(0, 1, max(2, len(rings))))
    for tp in plan:
        cm = G.render_coordmap(ERP_H, ERP_W, tp.yaw_deg, tp.pitch_deg, hfov, out)  # (out,out,2) erp (x,y)
        border = np.concatenate([cm[0, :], cm[:, -1], cm[-1, ::-1], cm[::-1, 0]], 0)  # CW loop
        x, y = _seam_split(border[:, 0], border[:, 1])
        c = colors[rings.index(round(tp.pitch_deg, 1))]
        ax.plot(x, y, "-", color=c, lw=1.6, alpha=0.85)
        cx, cy = cm[out // 2, out // 2]
        ax.plot(cx, cy, "o", color=c, ms=5, mec="k", mew=0.6)
    ax.set_xlim(0, ERP_W); ax.set_ylim(ERP_H, 0)
    return rings


def montage(erp_np, plan, hfov, cell=110, ncol=8):
    n = len(plan); nrow = int(np.ceil(n / ncol))
    canvas = np.zeros((nrow * cell, ncol * cell, 3), np.uint8)
    for i, tp in enumerate(plan):
        t = np.asarray(a2p.erp_to_pinhole_tile(erp_np, tp.yaw_deg, tp.pitch_deg, hfov, cell))
        r, c = divmod(i, ncol)
        canvas[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = t
    return canvas


def load_indoor():
    f = data.list_erps("stanford2d3d")[0]
    return np.array(Image.open(f).convert("RGB").resize((ERP_W, ERP_H), Image.BILINEAR))


def load_outdoor():
    try:
        f = data.list_densepass()[0]
        sh = round(400 * ERP_W / 2048)
        img = np.array(Image.open(f).convert("RGB").resize((ERP_W, sh), Image.BILINEAR))
        erp = np.zeros((ERP_H, ERP_W, 3), np.uint8); top = (ERP_H - sh) // 2
        erp[top:top + sh] = img
        return erp
    except Exception as e:
        print(f"  (densepass unavailable: {e}; reusing indoor ERP for outdoor footprint geometry)")
        return load_indoor()


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    configs = [("indoor: hfov 65°, rings -45/0/45, overlap 0.25", indoor_plan(), 65.0, load_indoor()),
               ("outdoor: hfov 50°, equator ring, overlap 0.25", outdoor_plan(), 50.0, load_outdoor())]

    fig, axes = plt.subplots(2, 2, figsize=(20, 11),
                             gridspec_kw={"width_ratios": [1.55, 1.0]})
    for row, (title, plan, hfov, erp) in enumerate(configs):
        axf = axes[row, 0]
        axf.imshow(erp, extent=[0, ERP_W, ERP_H, 0])
        rings = footprint_outline(axf, plan, hfov)
        axf.set_title(f"{title}  —  {len(plan)} tiles, {len(rings)} ring(s)  [footprints on ERP]", fontsize=12)
        axf.set_xlabel("ERP x (longitude)"); axf.set_ylabel("ERP y (latitude)")

        axm = axes[row, 1]
        axm.imshow(montage(erp, plan, hfov))
        axm.set_title(f"rendered perspective tiles (×{len(plan)})", fontsize=12)
        axm.axis("off")

    fig.suptitle("E2P tiling — patch-decompose of one ERP panorama into overlapping perspective tiles",
                 fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = os.path.join(OUTDIR, "tiling.png")
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"saved -> {out}")
    for title, plan, hfov, _ in configs:
        rings = sorted({round(tp.pitch_deg, 1) for tp in plan})
        per = [sum(1 for tp in plan if abs(tp.pitch_deg - r) < 1e-6) for r in rings]
        print(f"  {title}: {len(plan)} tiles | rings {rings} | per-ring {per}")


if __name__ == "__main__":
    main()
