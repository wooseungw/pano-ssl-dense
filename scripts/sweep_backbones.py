"""Frozen-backbone comparison on the same seg probe: DINOv2 / DINOv2-registers /
DINOv3, ERP-direct vs E2P@50deg, patch-matched + seeded. Answers: which frozen
encoder is best for panoramic segmentation, and does the E2P gain depend on backbone?

Per-backbone tile/ERP sizes are snapped to the backbone's patch size.
Run: CUDA_VISIBLE_DEVICES=1 conda run -n pano python scripts/sweep_backbones.py [dataset]
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import probe_seg_dinov3 as P  # noqa: E402
from encoder import PanoEncoder  # noqa: E402

DATASET = sys.argv[1] if len(sys.argv) > 1 else "densepass"
MODELS = [("DINOv2", "facebook/dinov2-base"),
          ("DINOv2-reg", "facebook/dinov2-with-registers-base"),
          ("DINOv3", "facebook/dinov3-vitb16-pretrain-lvd1689m")]
FOV, OVERLAP, SEED = 50.0, 0.25, 0
DEVICE = P.DEVICE
DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures", "sweep_backbones")


def e2p_feats(enc, rgb, lab, pmax):
    plan = P.a2p.plan_tiles("band", FOV, FOV, OVERLAP, pmax_deg=pmax)
    fs, ls = [], []
    for tp in plan:
        tile = np.asarray(P.a2p.erp_to_pinhole_tile(rgb, tp.yaw_deg, tp.pitch_deg, FOV, P.TILE))
        x = torch.from_numpy(tile).float().permute(2, 0, 1)[None] / 255.0
        feat = P.dense(enc, P.normalize_tiles(x.to(DEVICE)))[0]
        d, gh, gw = feat.shape
        gl = P.label_to_grid(P.e2p_label(lab, tp.yaw_deg, tp.pitch_deg, FOV, P.TILE), gh, gw)
        fs.append(feat.reshape(d, -1).t().cpu()); ls.append(torch.from_numpy(gl.reshape(-1)))
    return torch.cat(fs), torch.cat(ls)


def head(Xtr, ytr, Xva, yva, steps=800):
    torch.manual_seed(SEED)
    clf = torch.nn.Linear(Xtr.shape[1], P.N_CLASS).to(DEVICE)
    opt = torch.optim.Adam(clf.parameters(), 1e-3, weight_decay=1e-4)
    lf = torch.nn.CrossEntropyLoss(ignore_index=P.IGNORE)
    Xtr, ytr = Xtr.to(DEVICE).float(), ytr.to(DEVICE)
    for _ in range(steps):
        opt.zero_grad(); lf(clf(Xtr), ytr).backward(); opt.step()
    with torch.no_grad():
        pred = clf(Xva.to(DEVICE).float()).argmax(1).cpu()
    return P.miou_acc(pred, yva)[0]


def run_backbone(model):
    enc = PanoEncoder(model_id=model, lora_rank=0).to(DEVICE).eval()
    p = enc.patch
    P.TILE = (512 // p) * p
    P.WORK_HW = ((512 // p) * p, (1024 // p) * p)        # snap to patch multiples
    pmax = 35.0 if DATASET == "densepass" else 45.0
    panos, groups, train = P.grouped()
    cache = [("tr" if g in train else "va", P.load_rgb_label(f)) for g, f in panos]

    etf, etl, evf, evl = [], [], [], []
    for sp, (rgb, lab) in cache:
        ef, el = P.feats_erp(enc, rgb, lab)
        (etf if sp == "tr" else evf).append(ef); (etl if sp == "tr" else evl).append(el)
    Etf, Etl = torch.cat(etf), torch.cat(etl)
    Evf, Evl = torch.cat(evf), torch.cat(evl)
    n_tr, n_va = Etf.shape[0], Evf.shape[0]
    erp = head(Etf, Etl, Evf, Evl)

    ptf, ptl, pvf, pvl = [], [], [], []
    for sp, (rgb, lab) in cache:
        pf, pl = e2p_feats(enc, rgb, lab, pmax)
        (ptf if sp == "tr" else pvf).append(pf); (ptl if sp == "tr" else pvl).append(pl)
    Tf, Tl = P.subsample(torch.cat(ptf), torch.cat(ptl), n_tr, SEED)
    Vf, Vl = P.subsample(torch.cat(pvf), torch.cat(pvl), n_va, SEED)
    e2p = head(Tf, Tl, Vf, Vl)
    return p, erp, e2p


def main():
    P.configure(DATASET)
    print(f"dataset={DATASET} N_CLASS={P.N_CLASS} E2P_FOV={FOV} seed={SEED}\n"
          f"{'backbone':14s} {'patch':>5} {'ERP':>7} {'E2P':>7} {'dE2P':>7}", flush=True)
    rows = []
    for name, model in MODELS:
        try:
            p, erp, e2p = run_backbone(model)
            rows.append((name, p, erp, e2p))
            print(f"{name:14s} {p:5d} {erp:7.3f} {e2p:7.3f} {e2p - erp:+7.3f}", flush=True)
        except Exception as ex:
            print(f"{name:14s}  FAIL {type(ex).__name__}: {str(ex)[:70]}", flush=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(rows)); w = 0.35
    ax.bar(x - w / 2, [r[2] for r in rows], w, label="ERP-direct", color="#c44")
    ax.bar(x + w / 2, [r[3] for r in rows], w, label=f"E2P@{FOV:.0f}deg", color="#48c")
    for i, r in enumerate(rows):
        ax.text(i - w / 2, r[2] + .005, f"{r[2]:.3f}", ha="center", fontsize=8)
        ax.text(i + w / 2, r[3] + .005, f"{r[3]:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"{r[0]}\n(patch {r[1]})" for r in rows])
    ax.set_ylabel("seg mIoU"); ax.legend()
    ax.set_title(f"{DATASET}: frozen backbone comparison (linear probe, patch-matched)")
    out = os.path.join(DOCS, f"backbone_compare_{DATASET}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight"); print("saved", out, flush=True)


if __name__ == "__main__":
    main()
