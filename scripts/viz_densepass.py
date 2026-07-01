"""Visualize the DensePASS (real outdoor) DINOv3 ERP-vs-E2P segmentation probe.

docs/seg_probe_densepass.png       : RGB | GT | ERP-pred strips + per-class IoU + mIoU/acc
docs/seg_probe_densepass_tiles.png : per-crop (E2P equator tiles) RGB | GT | pred

Run: CUDA_VISIBLE_DEVICES=1 conda run -n pano python scripts/viz_densepass.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import probe_seg_dinov3 as P  # noqa: E402
import viz_seg_probe as V  # noqa: E402
from encoder import PanoEncoder  # noqa: E402

DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures", "viz_densepass")
CITYSCAPES = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
              "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
              "truck", "bus", "train", "motorcycle", "bicycle"]


def main():
    enc = PanoEncoder(model_id=P.MODEL, lora_rank=0).to(P.DEVICE).eval()
    res, panos, train = V.run("densepass", enc)        # configures densepass, trains heads
    nc = P.N_CLASS
    val_f = next(f for g, f in panos if g not in train)
    rgb, lab, pred = V.erp_predict(enc, res["erp"]["clf"], val_f)
    sh = round(400 * P.WORK_HW[1] / 2048); top = (P.WORK_HW[0] - sh) // 2
    sl = slice(top, top + sh)                          # crop padded ERP back to content strip
    rgb_s, gt_s, pr_s = rgb[sl], V.colorize(lab[sl], nc), V.colorize(pred[sl], nc)

    fig = plt.figure(figsize=(16, 7.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[2.3, 1.0], hspace=0.35, wspace=0.12)
    for i, (im, t) in enumerate([(rgb_s, "RGB (DensePASS outdoor strip)"),
                                 (gt_s, "GT (Cityscapes 19-cls)"),
                                 (pr_s, "ERP-direct prediction")]):
        ax = fig.add_subplot(gs[i, 0]); ax.imshow(im); ax.axis("off")
        ax.set_title(t, fontsize=10, loc="left")

    ax = fig.add_subplot(gs[0:2, 1])                   # per-class IoU (horizontal)
    cls = sorted(set(res["erp"]["iou"]) | set(res["e2p"]["iou"]))
    y = np.arange(len(cls)); h = 0.4
    ax.barh(y + h / 2, [res["erp"]["iou"].get(c, 0) for c in cls], h, label="ERP", color="#c44")
    ax.barh(y - h / 2, [res["e2p"]["iou"].get(c, 0) for c in cls], h, label="E2P", color="#48c")
    ax.set_yticks(y); ax.set_yticklabels([CITYSCAPES[c - 1] for c in cls], fontsize=7)
    ax.invert_yaxis(); ax.set_xlabel("IoU"); ax.set_title("per-class IoU"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[2, 1]); x = np.arange(2); w = 0.35   # mIoU / pixAcc
    erp = [res["erp"]["miou"], res["erp"]["acc"]]; e2p = [res["e2p"]["miou"], res["e2p"]["acc"]]
    ax.bar(x - w / 2, erp, w, label="ERP", color="#c44")
    ax.bar(x + w / 2, e2p, w, label="E2P", color="#48c")
    ax.set_xticks(x); ax.set_xticklabels(["mIoU", "pixAcc"]); ax.set_ylim(0, 1); ax.legend(fontsize=8)
    for xi in range(2):
        ax.text(xi - w / 2, erp[xi] + .02, f"{erp[xi]:.3f}", ha="center", fontsize=7)
        ax.text(xi + w / 2, e2p[xi] + .02, f"{e2p[xi]:.3f}", ha="center", fontsize=7)

    fig.suptitle(f"DensePASS (real outdoor) — DINOv3 frozen seg probe   |   "
                 f"E2P {res['e2p']['miou']:.3f} vs ERP {res['erp']['miou']:.3f} mIoU "
                 f"(Δ{res['e2p']['miou'] - res['erp']['miou']:+.3f}, patch-matched)", fontsize=13)
    out = os.path.join(DOCS, "seg_probe_densepass.png")
    fig.savefig(out, dpi=110, bbox_inches="tight"); print("saved", out)

    rows = V.e2p_tiles_qual(enc, res["e2p"]["clf"], val_f)   # per-crop tiles
    n = len(rows)
    fig2, axs = plt.subplots(n, 3, figsize=(7.5, 2.2 * n)); axs = np.atleast_2d(axs)
    for i, (t, gt, pr, yaw) in enumerate(rows):
        for j, im in enumerate((t, gt, pr)):
            axs[i, j].imshow(im); axs[i, j].axis("off")
        axs[i, 0].set_title(f"RGB crop yaw {yaw:+.0f}°", fontsize=9)
        if i == 0:
            axs[i, 1].set_title("GT", fontsize=9); axs[i, 2].set_title("E2P pred", fontsize=9)
    fig2.suptitle("DensePASS per-crop (AnyRes-E2P equator tiles) segmentation", fontsize=12)
    out2 = os.path.join(DOCS, "seg_probe_densepass_tiles.png")
    fig2.savefig(out2, dpi=110, bbox_inches="tight"); print("saved", out2)
    print(f"DensePASS ERP {res['erp']['miou']:.3f}/{res['erp']['acc']:.3f}  "
          f"E2P {res['e2p']['miou']:.3f}/{res['e2p']['acc']:.3f}")


if __name__ == "__main__":
    main()
