"""Bar-chart visualizations of the head-to-head + decomposition-ablation results.
Run: conda run -n pano python scripts/viz_results.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "docs/figures/viz_results"
os.makedirs(OUT, exist_ok=True)
TRANSFER, SCRATCH, OVL, NOOVL = "#2a9d57", "#c0392b", "#2a6fb0", "#9aa7b0"

# 1) seg head-to-head (area5 373 val, each on native grid)
fig, ax = plt.subplots(figsize=(7.4, 4.6))
labels = ["Ours hp12\n(transfer)", "Ours e2p\n(transfer)", "SphereUFormer\n(scratch ico)", "HEAL-SWIN\n(scratch HEALPix)"]
vals = [0.552, 0.548, 0.325, 0.207]
cols = [TRANSFER, TRANSFER, SCRATCH, SCRATCH]
bars = ax.bar(labels, vals, color=cols, edgecolor="black", linewidth=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", fontweight="bold")
ax.axhspan(0, 0, color=TRANSFER, label="planar transfer (frozen DINOv3)")
ax.bar(0, 0, color=SCRATCH, label="from-scratch spherical")
ax.set_ylabel("sphere mIoU  (area5 373 val)")
ax.set_title("Seg head-to-head: planar transfer ≫ from-scratch spherical\nStanford2D3D, 14 classes, 782 train panos")
ax.set_ylim(0, 0.65); ax.grid(axis="y", alpha=0.3); ax.legend(loc="upper right", fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/seg_head2head.png", dpi=130); plt.close()

# 2) decomposition × overlap ablation (frozen DINOv3, tr32/va16 linear probe, ERP128x256)
fig, ax = plt.subplots(figsize=(9.6, 4.8))
meth = ["e2p_full65", "hp12_h65", "tangent_ico20", "cube_ovl120", "cube_ovl110",
        "erp_direct", "tangent_ico80", "cube_rot", "cube6", "healpix_mosaic"]
vals2 = [0.574, 0.545, 0.545, 0.538, 0.532, 0.519, 0.517, 0.513, 0.481, 0.358]
ovl = [1, 0, 0, 1, 1, 0, 0, 0, 0, 0]   # overlap-style decomposition?
cols2 = [OVL if o else NOOVL for o in ovl]
cols2[0] = "#e67e22"      # ours E2P highlight
cols2[-1] = "#7d3c98"     # HEALPix mosaic highlight
bars = ax.bar(meth, vals2, color=cols2, edgecolor="black", linewidth=0.6)
for b, v in zip(bars, vals2):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}", ha="center", fontsize=8.5)
ax.annotate("overlap lifts\ncube +5.7%p", xy=(8, 0.481), xytext=(6.4, 0.40),
            arrowprops=dict(arrowstyle="->", color="black"), fontsize=8.5, ha="center")
ax.annotate("mosaic input\nout-of-distribution", xy=(9, 0.358), xytext=(8.0, 0.30),
            arrowprops=dict(arrowstyle="->", color="black"), fontsize=8.5, ha="center")
ax.set_ylabel("sphere mIoU"); ax.set_ylim(0, 0.63); ax.grid(axis="y", alpha=0.3)
ax.set_title("Decomposition × overlap (frozen DINOv3, sphere mIoU)\norange=ours E2P, blue=overlap, gray=no-overlap, purple=HEALPix mosaic")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8.5)
plt.tight_layout(); plt.savefig(f"{OUT}/decomp_overlap_ablation.png", dpi=130); plt.close()

print(f"saved {OUT}/seg_head2head.png  and  {OUT}/decomp_overlap_ablation.png")
