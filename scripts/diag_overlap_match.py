"""B: head-free overlap-similarity eval (no trained head). For each adjacent tile pair
on held-out panos, on the valid overlap A-cells:
  corr   = cosine to the TRUE geometric correspondent in B
  rand   = mean cosine to ALL B cells (negative control)  -> lift = corr - rand
  ret@1  = is the cosine nearest-neighbor in B the true correspondent? (within<=1 cell)
  sem@1  = does that NN's GT class match the A-cell's GT class? (head-free semantic)
frozen vs LoRA. If LoRA lifts corr WITHOUT inflating rand, the consistency gain is
correspondence-specific (not collapse); ret/sem say whether it's a usable head-free signal.

Run: CUDA_VISIBLE_DEVICES=1 conda run -n pano python scripts/diag_overlap_match.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data  # noqa: E402
import probe_seg_dinov3 as P  # noqa: E402
import train_ssl as T  # noqa: E402
from encoder import PanoEncoder, normalize_tiles  # noqa: E402

DEVICE = P.DEVICE
CKPT = os.environ.get("ADAPTER", T.CKPT)
EVAL = {"outdoor": ("densepass", "out", 50.0), "indoor": ("stanford2d3d", "in", 65.0)}


@torch.no_grad()
def tile_feats_labels(enc, rgb, lab, geom):
    feats, labs = [], []
    for (yaw, pitch) in geom["specs"]:
        tile = np.asarray(P.a2p.erp_to_pinhole_tile(rgb, yaw, pitch, geom["hfov"], P.TILE))
        x = torch.from_numpy(tile).float().permute(2, 0, 1)[None] / 255.0
        f = P.dense(enc, normalize_tiles(x.to(DEVICE)))[0]                 # (D,gh,gw)
        d, gh, gw = f.shape
        feats.append(F.normalize(f.permute(1, 2, 0).reshape(-1, d), dim=1))   # (gh*gw, D) normed
        gl = P.label_to_grid(P.e2p_label(lab, yaw, pitch, geom["hfov"], P.TILE), gh, gw)
        labs.append(torch.from_numpy(gl.reshape(-1)))
    return feats, labs, (gh, gw)


def true_b_cell(grid, gh, gw):
    gx, gy = grid[:, 0], grid[:, 1]
    bx = torch.clamp(((gx + 1) / 2 * gw - 0.5).round().long(), 0, gw - 1)
    by = torch.clamp(((gy + 1) / 2 * gh - 0.5).round().long(), 0, gh - 1)
    return by * gw + bx


def eval_encoder(enc, cache, geom):
    acc = {k: 0.0 for k in ("corr", "rand", "ret", "sem")}
    n_tot, n_sem = 0, 0
    for rgb, lab in cache:
        feats, labs, (gh, gw) = tile_feats_labels(enc, rgb, lab, geom)
        for (a, b), (grid, valid, weight) in zip(geom["pairs"], geom["warps"]):
            v = valid.to(DEVICE).bool()
            if v.sum() < 4:
                continue
            fa, fb = feats[a], feats[b]
            la, lb = labs[a].to(DEVICE), labs[b].to(DEVICE)
            tb = true_b_cell(grid.to(DEVICE), gh, gw)                     # (N,)
            sim = fa[v] @ fb.t()                                         # (Nv, M) cosine
            tbv = tb[v]; idx = torch.arange(tbv.shape[0], device=DEVICE)
            corr = sim[idx, tbv]
            nn = sim.argmax(1)
            # within<=1 cell tolerance in the (gh,gw) grid
            hit = ((nn // gw - tbv // gw).abs() <= 1) & ((nn % gw - tbv % gw).abs() <= 1)
            lav = la[v]; m = lav != P.IGNORE
            sem = (lb[nn][m] == lav[m])
            acc["corr"] += corr.sum().item(); acc["rand"] += sim.mean(1).sum().item()
            acc["ret"] += hit.float().sum().item(); n_tot += tbv.shape[0]
            acc["sem"] += sem.float().sum().item(); n_sem += int(m.sum().item())
    return (acc["corr"] / n_tot, acc["rand"] / n_tot, acc["ret"] / n_tot, acc["sem"] / max(n_sem, 1))


def val_cache(ds, kind):
    P.configure(ds); P.TILE = 512
    if kind == "out":
        dp = data.list_densepass(); files = dp[int(len(dp) * 0.7):]
    else:
        files = [f for f in data.list_erps("stanford2d3d")
                 if "5" in f.split("extracted_data/")[1].split("/")[0]][:40]
    return [P.load_rgb_label(f) for f in files]


def main():
    frozen = PanoEncoder(model_id=P.MODEL, lora_rank=0).to(DEVICE).eval()
    lora = PanoEncoder(model_id=P.MODEL, adapter_path=CKPT).to(DEVICE).eval()
    geoms = {k: T.build_geometry(frozen, hf, pc) for k, (hf, pc) in T.DOMAINS.items()}
    print(f"{'domain':8s} {'enc':7s} {'corr':>6} {'rand':>6} {'lift':>6} {'ret@1':>6} {'sem@1':>6}", flush=True)
    for name, (ds, kind, _) in EVAL.items():
        cache = val_cache(ds, kind)
        if not cache:
            print(f"{name:8s} skipped (no val panos on disk)", flush=True)
            continue
        geom = geoms[kind]
        for tag, enc in [("frozen", frozen), ("LoRA", lora)]:
            P.enc_patch = enc.patch
            corr, rand, ret, sem = eval_encoder(enc, cache, geom)
            print(f"{name:8s} {tag:7s} {corr:6.3f} {rand:6.3f} {corr-rand:6.3f} {ret:6.3f} {sem:6.3f}", flush=True)


if __name__ == "__main__":
    main()
