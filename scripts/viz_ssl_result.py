"""Frozen vs LoRA-SSL result: the SSL hugely improves held-out overlap feature
consistency but that is ~orthogonal to linear-probe segmentation. 3 panels from the
measured eval_ssl.py + diag_cos.py numbers (2026-06-20).

Run: conda run -n pano python scripts/viz_ssl_result.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures", "viz_ssl_result")

# domain: (frozen_cos, lora_cos, frozen_single, lora_single, frozen_blend, lora_blend, frozen_dis, lora_dis)
R = {
    "DensePASS\n(outdoor@50)": (0.680, 0.914, 0.326, 0.338, 0.445, 0.449, 0.323, 0.321),
    "Stanford2D3D\n(indoor@65)": (0.723, 0.876, 0.577, 0.576, 0.611, 0.605, 0.292, 0.283),
}
ds = list(R)
x = np.arange(len(ds)); w = 0.36
FZ, LO = "#c44", "#48c"

fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))

ax[0].bar(x - w / 2, [R[d][0] for d in ds], w, label="frozen", color=FZ)
ax[0].bar(x + w / 2, [R[d][1] for d in ds], w, label="LoRA-SSL", color=LO)
for i, d in enumerate(ds):
    ax[0].text(i + w / 2, R[d][1] + .01, f"+{R[d][1]-R[d][0]:.2f}", ha="center", fontsize=10, color=LO, fontweight="bold")
ax[0].set_title("Held-out overlap FEATURE cosine\n(what SSL optimizes) — BIG GAIN ✓")
ax[0].set_ylabel("mean overlap cosine"); ax[0].set_ylim(0, 1.0)

# single + blend mIoU
ax[1].bar(x - 1.5 * w / 2, [R[d][2] for d in ds], w / 1.5, label="frozen single", color=FZ)
ax[1].bar(x - 0.0, [R[d][3] for d in ds], w / 1.5, label="LoRA single", color=LO)
ax[1].bar(x + 1.5 * w / 2, [R[d][4] for d in ds], w / 1.5, label="frozen blend (ceiling)", color="#999", alpha=.6)
for i, d in enumerate(ds):
    ax[1].text(i, R[d][3] + .01, f"{R[d][3]-R[d][2]:+.3f}", ha="center", fontsize=9, color=LO, fontweight="bold")
ax[1].set_title("Single-tile seg mIoU\n(downstream) — FLAT ✗")
ax[1].set_ylabel("mIoU")

ax[2].bar(x - w / 2, [R[d][6] for d in ds], w, label="frozen", color=FZ)
ax[2].bar(x + w / 2, [R[d][7] for d in ds], w, label="LoRA-SSL", color=LO)
for i, d in enumerate(ds):
    ax[2].text(i + w / 2, R[d][7] + .005, f"{R[d][7]-R[d][6]:+.3f}", ha="center", fontsize=9, color=LO, fontweight="bold")
ax[2].set_title("Cross-tile prediction DISAGREEMENT\n(argmax) — barely moves ✗")
ax[2].set_ylabel("disagreement rate")

for a in ax:
    a.set_xticks(x); a.set_xticklabels(ds, fontsize=9); a.legend(fontsize=8)
fig.suptitle("Overlap-SSL works at the FEATURE level (+0.15–0.23 cosine, generalizes) but it is "
             "ORTHOGONAL to linear-probe semantics", fontsize=12, y=1.02)
out = os.path.join(DOCS, "ssl_result.png")
fig.tight_layout(); fig.savefig(out, dpi=120, bbox_inches="tight")
print("saved", out)
