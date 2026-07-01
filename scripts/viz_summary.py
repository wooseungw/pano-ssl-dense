"""Capstone: E2P-overlap SSL is a CONSISTENCY adapter, not an accuracy adapter.
Left = consistency metrics (big frozen->LoRA gains); right = accuracy metrics (flat).
All bars normalized to [0,1], higher=better; raw deltas annotated.

Run: conda run -n pano python scripts/viz_summary.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures", "viz_summary")
FZ, LO = "#c44", "#48c"

# (label, frozen_norm, lora_norm, raw_annotation)
CONS = [
    ("overlap cos\n(outdoor)", 0.680, 0.914, "+0.23"),
    ("overlap cos\n(indoor)", 0.723, 0.876, "+0.15"),
    ("corresp.\nretrieval@1", 0.208, 0.855, "+0.65"),
    ("retrieval@1\n(indoor)", 0.247, 0.620, "+0.37"),
    ("normal\nconsist.", 1 - 35.01 / 90, 1 - 29.89 / 90, "-5°"),
]
ACC = [
    ("seg mIoU\n(outdoor)", 0.326, 0.338, "+0.01"),
    ("seg mIoU\n(indoor)", 0.577, 0.576, "0.00"),
    ("normal acc\n(1-err/90)", 1 - 57.25 / 90, 1 - 57.84 / 90, "0°"),
]

fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [5, 3]})
for a, data, title in [(ax[0], CONS, "CONSISTENCY metrics — SSL WINS ✓ (generalizes, head-free)"),
                       (ax[1], ACC, "ACCURACY metrics — UNCHANGED (teacher-bounded)")]:
    x = np.arange(len(data)); w = 0.38
    a.bar(x - w / 2, [d[1] for d in data], w, label="frozen", color=FZ)
    a.bar(x + w / 2, [d[2] for d in data], w, label="LoRA-SSL", color=LO)
    for i, d in enumerate(data):
        a.text(i + w / 2, d[2] + .015, d[3], ha="center", fontsize=10, color=LO, fontweight="bold")
    a.set_xticks(x); a.set_xticklabels([d[0] for d in data], fontsize=8.5)
    a.set_ylim(0, 1.0); a.set_ylabel("normalized (higher=better)"); a.set_title(title, fontsize=11)
    a.legend(fontsize=9, loc="upper left")
fig.suptitle("E2P-overlap SSL = a cross-view CONSISTENCY adapter, not an accuracy adapter "
             "(DINOv3 frozen + 0.59M LoRA)", fontsize=13, y=1.01)
out = os.path.join(DOCS, "ssl_summary.png")
fig.tight_layout(); fig.savefig(out, dpi=120, bbox_inches="tight")
print("saved", out)
