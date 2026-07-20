"""Fair indoor E2P-vs-ERP: full_sphere coverage removes the pole handicap.

Indoor E2P with band(pmax=45) misses the poles (ceiling/floor center, |pitch|>45)
that ERP-direct sees, unfairly handicapping E2P. This re-measures with full_sphere
E2P coverage (incl ±90 pole caps) at the indoor-optimal hfov=65, vs ERP and vs the
handicapped band, all patch-matched + seeded. Saves a bar chart.

Run: CUDA_VISIBLE_DEVICES=1 conda run -n pano python scripts/fair_indoor.py
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

HFOV, OVERLAP, SEED = 65.0, 0.25, 0
DEVICE = P.DEVICE
DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures", "fair_indoor")


def e2p_feats(enc, rgb, lab, mode):
    if mode == "full_sphere":
        plan = P.a2p.plan_tiles("full_sphere", HFOV, HFOV, OVERLAP)
    else:
        plan = P.a2p.plan_tiles("band", HFOV, HFOV, OVERLAP, pmax_deg=45.0)
    fs, ls = [], []
    for tp in plan:
        tile = np.asarray(P.a2p.erp_to_pinhole_tile(rgb, tp.yaw_deg, tp.pitch_deg, HFOV, P.TILE))
        x = torch.from_numpy(tile).float().permute(2, 0, 1)[None] / 255.0
        feat = P.dense(enc, P.normalize_tiles(x.to(DEVICE)))[0]
        d, gh, gw = feat.shape
        gl = P.label_to_grid(P.e2p_label(lab, tp.yaw_deg, tp.pitch_deg, HFOV, P.TILE), gh, gw)
        fs.append(feat.reshape(d, -1).t().cpu()); ls.append(torch.from_numpy(gl.reshape(-1)))
    return torch.cat(fs), torch.cat(ls)


def head(Xtr, ytr, Xva, yva, steps=800):
    torch.manual_seed(SEED)
    return P.linear_probe(Xtr, ytr, Xva, yva, steps=steps)[0]


def collect(enc, kind, cache, n_tr=None, n_va=None):
    tf, tl, vf, vl = [], [], [], []
    for sp, (rgb, lab) in cache:
        f, l = P.feats_erp(enc, rgb, lab) if kind == "erp" else e2p_feats(enc, rgb, lab, kind)
        (tf if sp == "tr" else vf).append(f); (tl if sp == "tr" else vl).append(l)
    Tf, Tl, Vf, Vl = torch.cat(tf), torch.cat(tl), torch.cat(vf), torch.cat(vl)
    if n_tr:
        Tf, Tl = P.subsample(Tf, Tl, n_tr, SEED)
    if n_va:
        Vf, Vl = P.subsample(Vf, Vl, n_va, SEED)
    return Tf, Tl, Vf, Vl


def run(ds, enc):
    P.configure(ds)
    P.TILE = 512
    panos, groups, train = P.grouped()
    cache = [("tr" if g in train else "va", P.load_rgb_label(f)) for g, f in panos]
    Etf, Etl, Evf, Evl = collect(enc, "erp", cache)
    n_tr, n_va = Etf.shape[0], Evf.shape[0]
    erp = head(Etf, Etl, Evf, Evl)
    band = head(*collect(enc, "band", cache, n_tr, n_va))
    full = head(*collect(enc, "full_sphere", cache, n_tr, n_va))
    return erp, band, full


def main():
    enc = PanoEncoder(model_id=P.MODEL, lora_rank=0).to(DEVICE).eval()
    print(f"hfov={HFOV} overlap={OVERLAP} (DINOv3, patch-matched, seeded) seed={SEED}\n"
          f"{'dataset':14s} {'ERP':>7} {'E2P-band':>9} {'E2P-full':>9} {'full-ERP':>9}", flush=True)
    res, dss = {}, ["stanford2d3d", "structured3d"]
    for ds in dss:
        erp, band, full = run(ds, enc)
        res[ds] = (erp, band, full)
        print(f"{ds:14s} {erp:7.3f} {band:9.3f} {full:9.3f} {full - erp:+9.3f}", flush=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(dss)); w = 0.26
    ax.bar(x - w, [res[d][0] for d in dss], w, label="ERP-direct", color="#c44")
    ax.bar(x, [res[d][1] for d in dss], w, label="E2P band (±45, misses poles)", color="#999")
    ax.bar(x + w, [res[d][2] for d in dss], w, label="E2P full_sphere (fair)", color="#48c")
    for i, d in enumerate(dss):
        for k, off in [(0, -w), (1, 0), (2, w)]:
            ax.text(i + off, res[d][k] + .004, f"{res[d][k]:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(dss); ax.set_ylabel("seg mIoU"); ax.legend(fontsize=8)
    ax.set_title(f"Fair indoor coverage: E2P full_sphere vs band vs ERP (hfov={HFOV:.0f}, DINOv3)")
    out = os.path.join(DOCS, "fair_indoor_fullsphere.png")
    fig.savefig(out, dpi=120, bbox_inches="tight"); print("saved", out, flush=True)


if __name__ == "__main__":
    main()
