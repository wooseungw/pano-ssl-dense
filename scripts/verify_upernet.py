"""De-risk the UPerNet finding: is the SSL-encoder accuracy gain (seg +0.024, normal −0.8°)
real or single-split noise? Multi-seed + per-seed random train-subset, PAIRED frozen-vs-SSL
(same seed → same 180-pano subset for both encoders), report mean±std and per-seed Δ sign.

Cache features once per encoder (one at a time → memory), run all seeds, then pair by seed.
UPerNet decoder only. Stanford2D3D area5 val, tile-pixel metrics @128.

Run: CUDA_VISIBLE_DEVICES=0 conda run -n pano python scripts/verify_upernet.py
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
import multitask_eval as M  # noqa: E402
from encoder import PanoEncoder  # noqa: E402

DEVICE = P.DEVICE
SEEDS = int(os.environ.get("SEEDS", 4))
TR_CACHE, SAMPLE, VA = int(os.environ.get("TR_CACHE", 250)), int(os.environ.get("SAMPLE", 180)), 30
EPOCHS = int(os.environ.get("EPOCHS", 15))
TASKS = ("seg", "normal", "depth")


def subset_idx(n, seed):
    return torch.randperm(n, generator=torch.Generator().manual_seed(1000 + seed))[:SAMPLE].tolist()


def train_one(task, dim, sub, cva, seed):
    c = P.N_CLASS if task == "seg" else M.OUT_CH[task]
    torch.manual_seed(seed)
    dec = M.ZOO["UPerNet"](dim, c).to(DEVICE)
    opt = torch.optim.AdamW(dec.parameters(), 1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        for i in torch.randperm(len(sub), generator=g).tolist():
            feats, gts = sub[i]; opt.zero_grad()
            for s in range(0, len(feats), 8):
                fb = torch.stack([f.float() for f in feats[s:s + 8]]).to(DEVICE)
                ls = M.loss_of(task, dec(fb), gts[s:s + 8])
                if ls is not None:
                    (ls * fb.shape[0] / len(feats)).backward()
            opt.step()
    return M.evaluate(task, dec, cva)


def run_encoder(kw, tr, va):
    enc = PanoEncoder(model_id=P.MODEL, **kw).to(DEVICE).eval(); P.enc_patch = enc.patch
    ctr = [M.encode_pano(enc, f) for f in tr]; cva = [M.encode_pano(enc, f) for f in va]
    dim = enc.dim
    out = {t: [] for t in TASKS}
    for seed in range(SEEDS):
        sub = [ctr[i] for i in subset_idx(len(ctr), seed)]
        for t in TASKS:
            out[t].append(train_one(t, dim, sub, cva, seed))
    del enc, ctr, cva; torch.cuda.empty_cache()
    return out


def main():
    P.configure("stanford2d3d"); P.TILE = M.TILE
    P.plan = P.a2p.plan_tiles("band", M.HFOV, M.HFOV, 0.25, pmax_deg=45.0)
    files = data.list_erps("stanford2d3d")
    def area(f): return f.split("extracted_data/")[1].split("/")[0]
    tr = [f for f in files if "5" not in area(f)][:TR_CACHE]
    va = [f for f in files if "5" in area(f)][:VA]
    print(f"UPerNet de-risk: cache={len(tr)} sample={SAMPLE}/seed va={len(va)} seeds={SEEDS} ep={EPOCHS}", flush=True)

    print("caching + training FROZEN ...", flush=True)
    fz = run_encoder(dict(lora_rank=0), tr, va)
    print("caching + training SSL-LoRA ...", flush=True)
    ss = run_encoder(dict(adapter_path=T.CKPT), tr, va)

    def arr(x, idx=None):
        return np.array([v[idx] if idx is not None else v for v in x])

    print("\n=== UPerNet de-risk: frozen vs common SSL encoder (paired by seed) ===")
    for t in TASKS:
        if t == "depth":
            f, s = arr(fz[t], 0), arr(ss[t], 0)                # |Δlog| (lower better)
            d = s - f
            print(f"[depth |Δlog|↓]  frozen {f.mean():.3f}±{f.std():.3f}  SSL {s.mean():.3f}±{s.std():.3f}  "
                  f"Δ {d.mean():+.3f}±{d.std():.3f}  (Δ<0 in {int((d<0).sum())}/{SEEDS} seeds)")
            fd, sd = arr(fz[t], 1), arr(ss[t], 1)              # δ<1.25 (higher better)
            dd = sd - fd
            print(f"[depth δ<1.25↑]  frozen {fd.mean():.3f}±{fd.std():.3f}  SSL {sd.mean():.3f}±{sd.std():.3f}  "
                  f"Δ {dd.mean():+.3f}±{dd.std():.3f}  (Δ>0 in {int((dd>0).sum())}/{SEEDS})")
        else:
            f, s = arr(fz[t]), arr(ss[t]); d = s - f
            better = "Δ>0" if t == "seg" else "Δ<0"           # seg↑, normal↓
            cnt = int((d > 0).sum()) if t == "seg" else int((d < 0).sum())
            unit = "mIoU↑" if t == "seg" else "ang°↓"
            print(f"[{t} {unit}]  frozen {f.mean():.3f}±{f.std():.3f}  SSL {s.mean():.3f}±{s.std():.3f}  "
                  f"Δ {d.mean():+.3f}±{d.std():.3f}  ({better} in {cnt}/{SEEDS} seeds)")


if __name__ == "__main__":
    main()
