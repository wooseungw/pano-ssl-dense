"""Stanford2D3D seg/depth: our method vs SOTA (SphereUFormer paper Table 4) + my repros.
CAVEAT: SOTA = full models @ sphere rank7 (256x512), official split. OURS = frozen DINOv3 +
light task decoder @ ERP 128x256, area5-fold 782. Grids/resolutions/decoder differ — not apples-to-apples.
Run: conda run -n pano python scripts/viz_sota_compare.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "docs/figures/viz_results"; os.makedirs(OUT, exist_ok=True)

# SphereUFormer paper Table 4, Stanford2D3D (sphere rank7 256x512)
sota = ["PanoFormer", "EGFormer", "SFSS", "HexRUNet", "HEAL-SWIN", "Elite360D", "SphereUFormer"]
seg = [60.6, 66.4, 68.2, 56.1, 63.2, 71.4, 72.2]
d1 = [92.5, 93.1, 92.2, 90.1, 92.2, 93.5, 94.0]
OURS_SEG, OURS_D1 = 54.6, 84.4          # e2p_full65, 782 area5-fold, ERP128x256

GRAY, ORANGE = "#9aa7b0", "#e67e22"


def bar(vals, ours, ylab, title, ylim, fname):
    labels = sota + ["OURS\n(transfer)"]
    allv = vals + [ours]
    cols = [GRAY] * len(sota) + [ORANGE]
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    bars = ax.bar(labels, allv, color=cols, edgecolor="black", lw=0.6)
    for b, v in zip(bars, allv):
        ax.text(b.get_x() + b.get_width() / 2, v + (ylim[1] - ylim[0]) * 0.01, f"{v:.1f}", ha="center", fontsize=8.5)
    ax.set_ylabel(ylab); ax.set_title(title, fontsize=10)
    ax.set_ylim(*ylim); ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8.5)
    plt.tight_layout(); plt.savefig(f"{OUT}/{fname}", dpi=130); plt.close()


bar(seg, OURS_SEG, "seg mIoU ↑",
    "Stanford2D3D Semantic Segmentation\nSOTA: full models @sphere rank7 256x512  |  OURS: frozen DINOv3 + light decoder @ERP128x256",
    (0, 80), "sota_seg.png")
bar(d1, OURS_D1, "depth δ1 ↑",
    "Stanford2D3D Depth Estimation (δ1)\n(same caveats — grids/resolution/decoder differ)",
    (80, 96), "sota_depth.png")
print("saved sota_seg.png, sota_depth.png")
