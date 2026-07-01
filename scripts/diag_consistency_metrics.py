"""Beyond cosine: richer cross-tile consistency metrics, frozen vs LoRA-SSL, head-free.

On held-out overlaps we compute, for each encoder/domain:
  corr / rand / lift  - per-correspondence cosine (and negative control)
  ret@1               - greedy nearest-neighbour retrieval (collisions allowed)
  hun@1               - Hungarian optimal 1:1 assignment accuracy (stricter, global)
  CKA (linear)        - pooled centered-kernel-alignment of A-view vs B-view overlap
                        representations (orthogonal/scale invariant => structural, not
                        just a global-rotation cosine bump)

Run: CUDA_VISIBLE_DEVICES=0 conda run -n pano python scripts/diag_consistency_metrics.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data  # noqa: E402
import probe_seg_dinov3 as P  # noqa: E402
import train_ssl as T  # noqa: E402
from encoder import PanoEncoder, normalize_tiles  # noqa: E402

DEVICE = P.DEVICE
TILE, SEED, HUN_CAP = 512, 0, 256
EVAL = {"outdoor": ("densepass", "out"), "indoor": ("stanford2d3d", "in")}


@torch.no_grad()
def tile_feat(enc, rgb, yaw, pitch, hfov):
    tile = np.asarray(P.a2p.erp_to_pinhole_tile(rgb, yaw, pitch, hfov, TILE))
    x = torch.from_numpy(tile).float().permute(2, 0, 1)[None] / 255.0
    f = P.dense(enc, normalize_tiles(x.to(DEVICE)))[0]
    d, gh, gw = f.shape
    return F.normalize(f.permute(1, 2, 0).reshape(-1, d), dim=1).cpu(), (gh, gw)


def true_b_cell(grid, gh, gw):
    bx = np.clip(((grid[:, 0] + 1) / 2 * gw - 0.5).round().astype(int), 0, gw - 1)
    by = np.clip(((grid[:, 1] + 1) / 2 * gh - 0.5).round().astype(int), 0, gh - 1)
    return by * gw + bx


def linear_cka(X, Y):
    X = X - X.mean(0, keepdim=True); Y = Y - Y.mean(0, keepdim=True)
    xy = (X.t() @ Y).pow(2).sum()
    xx = (X.t() @ X).pow(2).sum().sqrt(); yy = (Y.t() @ Y).pow(2).sum().sqrt()
    return (xy / (xx * yy + 1e-9)).item()


def eval_encoder(enc, cache, geom):
    acc = {k: 0.0 for k in ("corr", "rand", "ret", "hun")}
    n_cell, n_hun = 0, 0
    XA, XB = [], []
    rng = np.random.RandomState(SEED)
    for rgb in cache:
        feats = [tile_feat(enc, rgb, y, p, geom["hfov"])[0] for (y, p) in geom["specs"]]
        gh, gw = (int(np.sqrt(feats[0].shape[0])),) * 2
        for (a, b), (grid, valid, weight) in zip(geom["pairs"], geom["warps"]):
            v = valid.cpu().numpy().astype(bool)
            if v.sum() < 8:
                continue
            FA, FB = feats[a], feats[b]
            tb = true_b_cell(grid.cpu().numpy(), gh, gw)
            sims = FA[v] @ FB.t()                                  # (Nv, M) cosine
            tbv = tb[v]; ar = np.arange(tbv.shape[0])
            acc["corr"] += sims[ar, tbv].sum().item(); acc["rand"] += sims.mean(1).sum().item()
            nn = sims.argmax(1).numpy()
            acc["ret"] += within1(nn, tbv, gw).sum(); n_cell += tbv.shape[0]
            # Hungarian on a capped subset (square-ish, fast)
            sel = ar if tbv.shape[0] <= HUN_CAP else rng.choice(tbv.shape[0], HUN_CAP, replace=False)
            r, c = linear_sum_assignment(-sims[sel].numpy())
            acc["hun"] += within1(c, tbv[sel][r], gw).sum(); n_hun += len(r)
            XA.append(FA[v]); XB.append(FB[tbv])
    XA, XB = torch.cat(XA), torch.cat(XB)
    cka = linear_cka(XA, XB)
    return (acc["corr"] / n_cell, acc["rand"] / n_cell, acc["ret"] / n_cell,
            acc["hun"] / max(n_hun, 1), cka)


def within1(pred, true, gw):
    return ((np.abs(pred // gw - true // gw) <= 1) & (np.abs(pred % gw - true % gw) <= 1)).astype(float)


def val_cache(ds, kind):
    P.configure(ds); P.TILE = TILE
    if kind == "out":
        dp = data.list_densepass(); files = dp[int(len(dp) * 0.7):][:30]
    else:
        files = [f for f in data.list_erps("stanford2d3d")
                 if "5" in f.split("extracted_data/")[1].split("/")[0]][:12]
    return [P.load_rgb_label(f)[0] for f in files]


def main():
    frozen = PanoEncoder(model_id=P.MODEL, lora_rank=0).to(DEVICE).eval()
    lora = PanoEncoder(model_id=P.MODEL, adapter_path=T.CKPT).to(DEVICE).eval()
    geoms = {k: T.build_geometry(frozen, hf, pc) for k, (hf, pc) in T.DOMAINS.items()}
    print(f"{'domain':8s} {'enc':7s} {'corr':>6} {'lift':>6} {'ret@1':>6} {'hun@1':>6} {'CKA':>6}", flush=True)
    rows = {}
    for name, (ds, kind) in EVAL.items():
        cache = val_cache(ds, kind); geom = geoms[kind]
        for tag, enc in [("frozen", frozen), ("LoRA", lora)]:
            P.enc_patch = enc.patch
            corr, rand, ret, hun, cka = eval_encoder(enc, cache, geom)
            rows[(name, tag)] = (corr, corr - rand, ret, hun, cka)
            print(f"{name:8s} {tag:7s} {corr:6.3f} {corr-rand:6.3f} {ret:6.3f} {hun:6.3f} {cka:6.3f}", flush=True)
    return rows


if __name__ == "__main__":
    main()
