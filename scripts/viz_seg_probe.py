"""Visualize the DINOv3 ERP-vs-E2P segmentation probe (scripts/probe_seg_dinov3.py).

Saves docs/seg_probe_dinov3.png:
  (1) mIoU + pixAcc bars, ERP vs E2P, both datasets
  (2) per-class IoU, Stanford2D3D (13 cls), ERP vs E2P -> explains mIoU/pixAcc split
  (3) qualitative ERP segmentation (RGB | GT | pred) for one Stanford2D3D val pano

Run: CUDA_VISIBLE_DEVICES=1 conda run -n pano python scripts/viz_seg_probe.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import probe_seg_dinov3 as P  # noqa: E402
from encoder import PanoEncoder  # noqa: E402

DEVICE = P.DEVICE
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures", "viz_seg_probe",
                   "seg_probe_dinov3.png")


def train_head(Xtr, ytr, steps=800):
    Xtr, ytr = Xtr.to(DEVICE).float(), ytr.to(DEVICE)
    clf = torch.nn.Linear(Xtr.shape[1], P.N_CLASS).to(DEVICE)
    opt = torch.optim.Adam(clf.parameters(), 1e-3, weight_decay=1e-4)
    lf = torch.nn.CrossEntropyLoss(ignore_index=P.IGNORE)
    for _ in range(steps):
        opt.zero_grad(); lf(clf(Xtr), ytr).backward(); opt.step()
    return clf


def per_class_iou(pred, gt):
    out = {}
    for c in range(1, P.N_CLASS):
        g = gt == c
        if g.sum() == 0:
            continue
        p = pred == c
        u = (p | g).sum().item()
        out[c] = (p & g).sum().item() / u if u else 0.0
    return out


def collect(enc):
    panos, groups, train = P.grouped()
    b = {r: {"tr": ([], []), "va": ([], [])} for r in ("erp", "e2p")}
    for g, f in panos:
        rgb, lab = P.load_rgb_label(f)
        ef, el = P.feats_erp(enc, rgb, lab)
        pf, pl, _ = P.feats_e2p(enc, rgb, lab)
        sp = "tr" if g in train else "va"
        b["erp"][sp][0].append(ef); b["erp"][sp][1].append(el)
        b["e2p"][sp][0].append(pf); b["e2p"][sp][1].append(pl)
    out = {r: {k: (torch.cat(b[r][k][0]), torch.cat(b[r][k][1])) for k in ("tr", "va")}
           for r in ("erp", "e2p")}
    for k in ("tr", "va"):                     # patch-count match
        out["e2p"][k] = P.subsample(*out["e2p"][k], out["erp"][k][0].shape[0])
    return out, panos, train


def run(dataset, enc):
    P.configure(dataset)
    out, panos, train = collect(enc)
    res = {}
    for r in ("erp", "e2p"):
        clf = train_head(*out[r]["tr"])
        Xva, yva = out[r]["va"]
        with torch.no_grad():
            pred = clf(Xva.to(DEVICE).float()).argmax(1).cpu()
        m = yva != P.IGNORE
        ic = per_class_iou(pred, yva)
        res[r] = {"miou": float(np.mean(list(ic.values()))) if ic else 0.0,
                  "acc": (pred[m] == yva[m]).float().mean().item(), "iou": ic, "clf": clf}
    return res, panos, train


def colorize(lab, ncls):
    cmap = plt.get_cmap("tab20", max(ncls, 20))
    pal = (np.array([cmap(i)[:3] for i in range(ncls)]) * 255).astype(np.uint8)
    pal[0] = [35, 35, 35]                       # void -> dark
    return pal[np.clip(lab, 0, ncls - 1)]


@torch.no_grad()
def erp_predict(enc, clf, f):
    rgb, lab = P.load_rgb_label(f)
    x = torch.from_numpy(rgb).float().permute(2, 0, 1)[None] / 255.0
    feat = P.dense(enc, P.normalize_tiles(x.to(DEVICE)))[0]      # (D,gh,gw)
    d, gh, gw = feat.shape
    predg = clf(feat.reshape(d, -1).t().float()).argmax(1).cpu().numpy().reshape(gh, gw)
    pred = np.array(Image.fromarray(predg.astype(np.uint8)).resize(
        (rgb.shape[1], rgb.shape[0]), Image.NEAREST))
    return rgb, lab, pred


@torch.no_grad()
def e2p_tiles_qual(enc, clf, f, max_tiles=6):
    """Per-crop view: for one pano's equator E2P tiles, return
    (rgb_tile, gt_color, pred_color, yaw) using the trained E2P head."""
    rgb, lab = P.load_rgb_label(f)
    plan = [tp for tp in P.a2p.plan_tiles("band", P.HFOV, P.HFOV, P.OVERLAP, pmax_deg=P.PMAX)
            if abs(tp.pitch_deg) < 1e-6][:max_tiles]
    rows = []
    for tp in plan:
        tile = np.asarray(P.a2p.erp_to_pinhole_tile(rgb, tp.yaw_deg, tp.pitch_deg, P.HFOV, P.TILE))
        x = torch.from_numpy(tile).float().permute(2, 0, 1)[None] / 255.0
        feat = P.dense(enc, P.normalize_tiles(x.to(DEVICE)))[0]
        d, gh, gw = feat.shape
        predg = clf(feat.reshape(d, -1).t().float()).argmax(1).cpu().numpy().reshape(gh, gw)
        pred = np.array(Image.fromarray(predg.astype(np.uint8)).resize((P.TILE, P.TILE), Image.NEAREST))
        gt = P.e2p_label(lab, tp.yaw_deg, tp.pitch_deg, P.HFOV, P.TILE)
        rows.append((tile, colorize(gt, P.N_CLASS), colorize(pred, P.N_CLASS), tp.yaw_deg))
    return rows


def main():
    enc = PanoEncoder(model_id=P.MODEL, lora_rank=0).to(DEVICE).eval()
    print("running structured3d ..."); rs3, _, _ = run("structured3d", enc)
    print("running stanford2d3d ..."); rs2, panos2, train2 = run("stanford2d3d", enc)
    # P now configured for stanford2d3d (N_CLASS=14) -> use for qualitative
    val_f = next(f for g, f in panos2 if g not in train2)
    rgb, lab, pred = erp_predict(enc, rs2["erp"]["clf"], val_f)
    nc = P.N_CLASS

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.32, wspace=0.25)

    # (1) mIoU + pixAcc bars
    dsets = [("Structured3D", rs3), ("Stanford2D3D", rs2)]
    for col, metric, title in [(0, "miou", "mIoU"), (1, "acc", "pixel acc")]:
        ax = fig.add_subplot(gs[0, col]); x = np.arange(len(dsets)); w = 0.35
        erp = [r[metric] for _, r in [(n, d["erp"]) for n, d in dsets]]
        e2p = [r[metric] for _, r in [(n, d["e2p"]) for n, d in dsets]]
        b1 = ax.bar(x - w / 2, erp, w, label="ERP-direct", color="#c44")
        b2 = ax.bar(x + w / 2, e2p, w, label="E2P-pinhole", color="#48c")
        ax.set_xticks(x); ax.set_xticklabels([n for n, _ in dsets]); ax.set_title(title)
        ax.set_ylim(0, max(erp + e2p) * 1.25); ax.legend(fontsize=8)
        for bars in (b1, b2):
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)

    # (2) per-class IoU, Stanford2D3D
    ax = fig.add_subplot(gs[0, 2])
    cls = sorted(set(rs2["erp"]["iou"]) | set(rs2["e2p"]["iou"]))
    names = [P.S2D3D_CLASSES[c - 1] for c in cls]
    xe = np.arange(len(cls)); w = 0.4
    ax.bar(xe - w / 2, [rs2["erp"]["iou"].get(c, 0) for c in cls], w, label="ERP", color="#c44")
    ax.bar(xe + w / 2, [rs2["e2p"]["iou"].get(c, 0) for c in cls], w, label="E2P", color="#48c")
    ax.set_xticks(xe); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax.set_title("Stanford2D3D per-class IoU"); ax.set_ylabel("IoU"); ax.legend(fontsize=8)

    # (3) qualitative RGB | GT | ERP-pred
    for col, (img, ttl) in enumerate([
            (rgb, "RGB (ERP)"),
            (colorize(lab, nc), "GT semantic"),
            (colorize(pred, nc), "ERP-direct prediction")]):
        ax = fig.add_subplot(gs[1, col]); ax.imshow(img); ax.set_title(ttl, fontsize=10)
        ax.axis("off")

    fig.suptitle("DINOv3 (frozen) segmentation probe — ERP-direct vs AnyRes-E2P  "
                 "[patch-matched, group-disjoint]", fontsize=13)
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print(f"saved {OUT}")

    # per-crop (E2P tile) segmentation figure
    rows = e2p_tiles_qual(enc, rs2["e2p"]["clf"], val_f)
    n = len(rows)
    fig2, axs = plt.subplots(n, 3, figsize=(7.0, 2.4 * n))
    axs = np.atleast_2d(axs)
    for i, (t, gt, pr, yaw) in enumerate(rows):
        for j, im in enumerate((t, gt, pr)):
            axs[i, j].imshow(im); axs[i, j].axis("off")
        axs[i, 0].set_title(f"RGB crop  yaw {yaw:+.0f}°", fontsize=9)
        if i == 0:
            axs[i, 1].set_title("GT", fontsize=9); axs[i, 2].set_title("E2P prediction", fontsize=9)
    fig2.suptitle("Per-crop (AnyRes-E2P tile) segmentation — Stanford2D3D, DINOv3 + E2P head",
                  fontsize=12)
    out2 = OUT.replace(".png", "_tiles.png")
    fig2.savefig(out2, dpi=110, bbox_inches="tight")
    print(f"saved {out2}")
    print(f"S3D  ERP {rs3['erp']['miou']:.3f}/{rs3['erp']['acc']:.3f}  "
          f"E2P {rs3['e2p']['miou']:.3f}/{rs3['e2p']['acc']:.3f}")
    print(f"S2D3D ERP {rs2['erp']['miou']:.3f}/{rs2['erp']['acc']:.3f}  "
          f"E2P {rs2['e2p']['miou']:.3f}/{rs2['e2p']['acc']:.3f}")


if __name__ == "__main__":
    main()
