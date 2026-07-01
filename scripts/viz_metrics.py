"""Consistency metrics beyond cosine: corr / lift / retrieval@1 / Hungarian@1 / CKA,
frozen vs LoRA, per domain. From diag_consistency_metrics.py (2026-06-21).

Run: conda run -n pano python scripts/viz_metrics.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures", "viz_metrics")
M = ["corr", "lift", "ret@1", "hun@1", "CKA"]
R = {  # domain -> (frozen[5], lora[5])
    "DensePASS (outdoor)": ([0.678, 0.243, 0.208, 0.266, 0.898], [0.896, 0.398, 0.865, 0.865, 0.924]),
    "Stanford2D3D (indoor)": ([0.711, 0.246, 0.246, 0.312, 0.517], [0.864, 0.324, 0.607, 0.660, 0.828]),
}
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for ax, (name, (fz, lo)) in zip(axes, R.items()):
    x = np.arange(len(M)); w = 0.38
    ax.bar(x - w / 2, fz, w, label="frozen", color="#c44")
    ax.bar(x + w / 2, lo, w, label="LoRA-SSL", color="#48c")
    for i in range(len(M)):
        ax.text(i + w / 2, lo[i] + .015, f"+{lo[i]-fz[i]:.2f}", ha="center", fontsize=9,
                color="#48c", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(M); ax.set_ylim(0, 1.05)
    ax.set_title(name); ax.set_ylabel("metric (higher=better)"); ax.legend(fontsize=9, loc="upper left")
fig.suptitle("Cross-tile consistency beyond cosine — retrieval / Hungarian / CKA all rise (frozen→LoRA)",
             y=1.02, fontsize=13)
out = os.path.join(DOCS, "viz_metrics.png")
fig.tight_layout(); fig.savefig(out, dpi=120, bbox_inches="tight")
print("saved", out)
