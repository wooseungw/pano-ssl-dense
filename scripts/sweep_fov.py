"""FOV / tiling sweep for AnyRes-E2P segmentation probe — find the best pinhole FOV.

For a grid of (hfov, overlap) tiling configs, extract DINOv3 features on the E2P
tiles, patch-match to the ERP baseline patch count, train a SEEDED linear head, and
measure mIoU. Same seed + same val split + same patch count across all configs, so
differences reflect tiling geometry (FOV/overlap) only — not init noise or data volume.
Prints a table + saves a heatmap, and reports the best FOV.

Run: CUDA_VISIBLE_DEVICES=1 conda run -n pano python scripts/sweep_fov.py [dataset]
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
HFOVS = [30, 40, 50, 65, 90]
OVERLAPS = [0.25]
SEED = 0
DEVICE = P.DEVICE
DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures", "sweep_fov")


def e2p_feats(enc, rgb, lab, hfov, overlap):
    pmax = 35.0 if DATASET == "densepass" else 45.0     # square tiles cover the content band
    plan = P.a2p.plan_tiles("band", hfov, hfov, overlap, pmax_deg=pmax)
    fs, ls = [], []
    for tp in plan:
        tile = np.asarray(P.a2p.erp_to_pinhole_tile(rgb, tp.yaw_deg, tp.pitch_deg, hfov, P.TILE))
        x = torch.from_numpy(tile).float().permute(2, 0, 1)[None] / 255.0
        feat = P.dense(enc, P.normalize_tiles(x.to(DEVICE)))[0]
        d, gh, gw = feat.shape
        gl = P.label_to_grid(P.e2p_label(lab, tp.yaw_deg, tp.pitch_deg, hfov, P.TILE), gh, gw)
        fs.append(feat.reshape(d, -1).t().cpu())
        ls.append(torch.from_numpy(gl.reshape(-1)))
    return torch.cat(fs), torch.cat(ls), len(plan)


def seeded_head(Xtr, ytr, Xva, yva, steps=800):
    torch.manual_seed(SEED)
    return P.linear_probe(Xtr, ytr, Xva, yva, steps=steps)


def main():
    P.configure(DATASET)
    enc = PanoEncoder(model_id=P.MODEL, lora_rank=0).to(DEVICE).eval()
    panos, groups, train = P.grouped()
    print(f"dataset={DATASET} panos={len(panos)} N_CLASS={P.N_CLASS} seed={SEED} tile={P.TILE}",
          flush=True)
    cache = [("tr" if g in train else "va", P.load_rgb_label(f)) for g, f in panos]

    etf, etl, evf, evl = [], [], [], []
    for sp, (rgb, lab) in cache:
        ef, el = P.feats_erp(enc, rgb, lab)
        (etf if sp == "tr" else evf).append(ef)
        (etl if sp == "tr" else evl).append(el)
    Etf, Etl, Evf, Evl = torch.cat(etf), torch.cat(etl), torch.cat(evf), torch.cat(evl)
    n_tr, n_va = Etf.shape[0], Evf.shape[0]
    erp_miou, erp_acc, _ = seeded_head(Etf, Etl, Evf, Evl)
    print(f"ERP baseline: mIoU {erp_miou:.3f} acc {erp_acc:.3f} (patches tr={n_tr} va={n_va})\n"
          f"{'hfov':>5} {'ovlp':>5} {'tiles':>5} {'mIoU':>7} {'acc':>6} {'dVsERP':>7}", flush=True)

    grid = np.full((len(HFOVS), len(OVERLAPS)), np.nan)
    for i, hf in enumerate(HFOVS):
        for j, ov in enumerate(OVERLAPS):
            tf, tl, vf, vl, nt = [], [], [], [], 0
            for sp, (rgb, lab) in cache:
                pf, pl, nt = e2p_feats(enc, rgb, lab, hf, ov)
                (tf if sp == "tr" else vf).append(pf)
                (tl if sp == "tr" else vl).append(pl)
            Tf, Tl = P.subsample(torch.cat(tf), torch.cat(tl), n_tr, SEED)
            Vf, Vl = P.subsample(torch.cat(vf), torch.cat(vl), n_va, SEED)
            miou, acc, _ = seeded_head(Tf, Tl, Vf, Vl)
            grid[i, j] = miou
            print(f"{hf:5.0f} {ov:5.2f} {nt:5d} {miou:7.3f} {acc:6.3f} {miou - erp_miou:+7.3f}",
                  flush=True)

    bi, bj = np.unravel_index(np.nanargmax(grid), grid.shape)
    print(f"\nBEST: hfov={HFOVS[bi]}deg overlap={OVERLAPS[bj]} mIoU={grid[bi, bj]:.3f} "
          f"(ERP {erp_miou:.3f}, dVsERP {grid[bi, bj] - erp_miou:+.3f})", flush=True)

    pmax = 35 if DATASET == "densepass" else 45
    fig, ax = plt.subplots(figsize=(8, 5))
    for j, ov in enumerate(OVERLAPS):
        ax.plot(HFOVS, grid[:, j], marker="o", lw=2, label=f"E2P (overlap {ov})")
    ax.axhline(erp_miou, ls="--", color="gray", label=f"ERP baseline {erp_miou:.3f}")
    ax.scatter([HFOVS[bi]], [grid[bi, bj]], s=240, facecolor="none", edgecolor="red",
               lw=2.5, zorder=5, label=f"best: {HFOVS[bi]}deg")
    ax.set_xlabel("pinhole hfov (deg)"); ax.set_ylabel("E2P seg mIoU")
    ax.set_title(f"{DATASET}: E2P mIoU vs FOV (fair coverage, band pmax={pmax})\n"
                 f"[seeded, patch-matched; ERP={erp_miou:.3f}]")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    out = os.path.join(DOCS, f"sweep_fov_{DATASET}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight"); print("saved", out, flush=True)


if __name__ == "__main__":
    main()
