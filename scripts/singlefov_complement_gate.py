"""Single-FOV complementarity gate: do overlapping same-FOV tiles carry COMPLEMENTARY info that
the averaging merge throws away?

Fusion/neck tied because the merge AVERAGES overlapping tiles -> keeps only the redundant (shared)
part, discards inter-view variation. Test whether that discarded variation is USEFUL: at each cell
covered by >=2 tiles, compare linear-probe seg of
  mean (current merge, D)  vs  concat[mean,std] (2D)  vs  max (D)  vs  concat[mean,max] (2D).
std/max across covering tiles = the complementary signal averaging destroys.

  concat/max > mean  => complementary info exists -> designing tiles to be complementary has merit
  ~tie               => overlaps are redundant (averaging is the ceiling; consistent with deform tie)

Run: CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n pano python scripts/singlefov_complement_gate.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data  # noqa: E402
import anyres_e2p as a2p  # noqa: E402
import geometry as G  # noqa: E402
import probe_seg_dinov3 as P  # noqa: E402
from viz_merged_field import HF, WF, TILE, PATCH, ERP_W, ERP_H  # noqa: E402
from encoder import PanoEncoder, normalize_tiles  # noqa: E402

DEVICE = "cuda"
SEED = 0
NC = HF * WF
MODEL = os.environ.get("MODEL", "facebook/dinov3-vitb16-pretrain-lvd1689m")
FOV = float(os.environ.get("FOV", 65.0))
TR = int(os.environ.get("TR", 100))
VA = int(os.environ.get("VA", 30))


def cell_ids_fov(plan, gh):
    ids = []
    for tp in plan:
        cm = G.render_coordmap(ERP_H, ERP_W, tp.yaw_deg, tp.pitch_deg, FOV, gh)
        uf = np.clip((cm[..., 0] / ERP_W * WF).astype(int), 0, WF - 1)
        vf = np.clip((cm[..., 1] / ERP_H * HF).astype(int), 0, HF - 1)
        ids.append((vf * WF + uf).reshape(-1))
    return ids


@torch.no_grad()
def build_reps(enc, erp, plan, ids):
    """Per cell, aggregate covering tiles' patch features into mean / std / max. (>=2-cov cells.)"""
    D = enc.dim
    s = np.zeros((NC, D), np.float64); sq = np.zeros((NC, D), np.float64)
    mx = np.full((NC, D), -1e9, np.float32); cnt = np.zeros(NC, np.float64)
    for tp, c in zip(plan, ids):
        t = np.asarray(a2p.erp_to_pinhole_tile(erp, tp.yaw_deg, tp.pitch_deg, FOV, TILE))
        x = normalize_tiles((torch.from_numpy(t).float().permute(2, 0, 1)[None] / 255.0).to(DEVICE))
        fmap = enc(x)[0].permute(1, 2, 0).reshape(-1, D).float().cpu().numpy()
        np.add.at(s, c, fmap); np.add.at(sq, c, fmap ** 2)
        np.maximum.at(mx, c, fmap); np.add.at(cnt, c, 1.0)
    cov2 = cnt >= 2
    cnt_ = np.maximum(cnt, 1)[:, None]
    mean = (s / cnt_).astype(np.float16)
    std = np.sqrt(np.clip(sq / cnt_ - (s / cnt_) ** 2, 0, None)).astype(np.float16)
    return mean, std, mx.astype(np.float16), cov2


def cache(enc, files, plan, ids):
    out = []
    for f in files:
        erp = np.array(Image.open(f).convert("RGB").resize((ERP_W, ERP_H), Image.BILINEAR))
        mean, std, mx, cov2 = build_reps(enc, erp, plan, ids)
        _, lab = P.load_rgb_label(f)
        seg = P.label_to_grid(lab, HF, WF).reshape(-1).astype(np.int64)
        out.append((mean, std, mx, cov2, seg))
    return out


def gather(c, mode):
    Xs, ys = [], []
    for mean, std, mx, cov2, seg in c:
        m = mean[cov2]
        X = {"mean": m, "max": mx[cov2],
             "mean+std": np.concatenate([m, std[cov2]], 1),
             "mean+max": np.concatenate([m, mx[cov2]], 1)}[mode]
        Xs.append(X.astype(np.float32)); ys.append(seg[cov2])
    return np.concatenate(Xs), np.concatenate(ys)


def probe(Xtr, ytr, Xva, yva):
    keep = ytr != P.IGNORE; Xtr, ytr = Xtr[keep], ytr[keep]
    idx = np.random.RandomState(SEED).permutation(len(Xtr))[:300000]
    Xt = torch.from_numpy(Xtr[idx]).to(DEVICE); yt = torch.from_numpy(ytr[idx]).to(DEVICE)
    torch.manual_seed(SEED); clf = nn.Linear(Xt.shape[1], P.N_CLASS).to(DEVICE)
    opt = torch.optim.Adam(clf.parameters(), 1e-3, weight_decay=1e-4)
    for _ in range(800):
        opt.zero_grad(); F.cross_entropy(clf(Xt), yt, ignore_index=P.IGNORE).backward(); opt.step()
    with torch.no_grad():
        pr = clf(torch.from_numpy(Xva).to(DEVICE)).argmax(1).cpu()
    return P.miou_acc(pr, torch.from_numpy(yva))[0]


def main():
    np.random.seed(SEED)
    P.configure("stanford2d3d"); P.TILE = TILE
    enc = PanoEncoder(model_id=MODEL, lora_rank=0).to(DEVICE).eval(); P.enc_patch = enc.patch
    plan = a2p.plan_tiles("full_sphere", FOV, FOV, 0.25); ids = cell_ids_fov(plan, TILE // PATCH)
    files = data.list_erps("stanford2d3d")
    def area(f): return f.split("extracted_data/")[1].split("/")[0]
    tr = [f for f in files if "5" not in area(f)][:TR]; va = [f for f in files if "5" in area(f)][:VA]
    print(f"single-FOV complementarity gate | enc={MODEL.split('/')[-1]} FOV={FOV:.0f} tiles={len(plan)} "
          f"field={HF}×{WF} (>=2-cov cells) tr={len(tr)} va={len(va)}", flush=True)
    ctr = cache(enc, tr, plan, ids); cva = cache(enc, va, plan, ids)
    print(f"  mean >=2-covered frac = {np.mean([c[3].mean() for c in ctr]):.3f}", flush=True)

    res = {}
    for mode in ("mean", "max", "mean+std", "mean+max"):
        Xtr, ytr = gather(ctr, mode); Xva, yva = gather(cva, mode)
        res[mode] = probe(Xtr, ytr, Xva, yva)
        print(f"  {mode:9} (dim {Xtr.shape[1]:4d}) -> mIoU {res[mode]:.3f}", flush=True)
    base = res["mean"]; best = max(res["max"], res["mean+std"], res["mean+max"])
    d = best - base
    gate = "✅ complementary info exists" if d > 0.01 else "❌ overlaps redundant (averaging is ceiling)"
    print(f"\n=== mean(merge) {base:.3f} | best-complementary {best:.3f}  Δ={d:+.3f}  {gate} ===", flush=True)


if __name__ == "__main__":
    main()
