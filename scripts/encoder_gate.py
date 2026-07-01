"""Accuracy gate: does a BIGGER encoder (DINOv3 ViT-L/16) beat ViT-B/16?

Accuracy is feature-bound (fusion/decoder all tie) -> the encoder is the only lever that changes
feature CONTENT. Both are patch-16, so the 64x128 obliquity field geometry is IDENTICAL; only the
feature (dim 768 -> 1024, depth 12 -> 24) changes. Linear probe on the naive blend field, best-val
mIoU, same protocol as field_refine_probe's matched L0 baseline (ViT-B reproduces ~0.578).

Gate: ViT-L best - ViT-B best > +0.01 => bigger encoder is a real accuracy lever => worth the FLOPs.

Run: CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n pano python scripts/encoder_gate.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data  # noqa: E402
import probe_seg_dinov3 as P  # noqa: E402
from adaptive_field_deform import naive_field, precompute_geom, encode, dev, gt_field  # noqa: E402
from encoder import PanoEncoder  # noqa: E402

DEVICE = P.DEVICE
SEED = 0
EPOCHS = int(os.environ.get("EPOCHS", 15))
TR = int(os.environ.get("TR", 120))
VA = int(os.environ.get("VA", 40))
MODELS = {
    "ViT-B/16 (768d, 12L)": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "ViT-L/16 (1024d, 24L)": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}


def linear_probe_field(enc, tr, va, scatter):
    """Linear probe on the naive 64x128 obliquity field; report BEST val mIoU over epochs."""
    P.enc_patch = enc.patch
    cache = {"tr": [], "va": []}
    for sp, fl in [("tr", tr), ("va", va)]:
        for f in fl:
            rgb, lab = P.load_rgb_label(f)
            cache[sp].append((encode(enc, rgb, P.plan), gt_field(lab)))

    torch.manual_seed(SEED)
    decd = nn.Linear(enc.dim, P.N_CLASS).to(DEVICE)
    opt = torch.optim.AdamW(decd.parameters(), 1e-3, weight_decay=1e-4)
    lf = nn.CrossEntropyLoss(ignore_index=P.IGNORE)
    g = torch.Generator().manual_seed(SEED)

    def evaluate():
        decd.eval()
        inter = torch.zeros(P.N_CLASS); union = torch.zeros(P.N_CLASS)
        with torch.no_grad():
            for feats, y in cache["va"]:
                nf, cov = naive_field(dev(feats), scatter)
                pred = decd(nf).argmax(1).cpu()
                cm = cov.cpu(); mm = (y != P.IGNORE) & cm
                for c in range(1, P.N_CLASS):
                    inter[c] += ((pred == c) & (y == c) & mm).sum(); union[c] += (((pred == c) | (y == c)) & mm).sum()
        decd.train()
        return float(np.mean([(inter[c] / union[c]).item() for c in range(1, P.N_CLASS) if union[c] > 0]))

    best, traj = 0.0, []
    for _ in range(EPOCHS):
        for i in torch.randperm(len(cache["tr"]), generator=g).tolist():
            feats, y = cache["tr"][i]; yd = y.to(DEVICE)
            nf, cov = naive_field(dev(feats), scatter)
            opt.zero_grad(); lf(decd(nf[cov]), yd[cov]).backward(); opt.step()
        m = evaluate(); traj.append(m); best = max(best, m)
    print(f"    val/ep=[{' '.join(f'{m:.3f}' for m in traj)}] best={best:.3f}", flush=True)
    del cache; torch.cuda.empty_cache()
    return best


def main():
    P.configure("stanford2d3d"); P.TILE = 512
    P.plan = P.a2p.plan_tiles("band", 65.0, 65.0, 0.25, pmax_deg=45.0)
    scatter, _, _ = precompute_geom(P.plan)
    files = data.list_erps("stanford2d3d")
    def area(f): return f.split("extracted_data/")[1].split("/")[0]
    tr = [f for f in files if "5" not in area(f)][:TR]
    va = [f for f in files if "5" in area(f)][:VA]
    print(f"encoder accuracy gate: field 64×128 naive obliquity probe | tiles={len(P.plan)} "
          f"tr={len(tr)} va={len(va)} ep={EPOCHS}\n", flush=True)
    res = {}
    for name, mid in MODELS.items():
        enc = PanoEncoder(model_id=mid, lora_rank=0).to(DEVICE).eval()
        print(f"[{name}] dim={enc.dim} patch={enc.patch}", flush=True)
        res[name] = linear_probe_field(enc, tr, va, scatter)
        del enc; torch.cuda.empty_cache()
    b = res["ViT-B/16 (768d, 12L)"]; l = res["ViT-L/16 (1024d, 24L)"]
    gate = "✅ encoder lever works" if l - b > 0.01 else "❌ no gain (encoder not the lever either)"
    print(f"\n=== ViT-B {b:.3f}  vs  ViT-L {l:.3f}   Δ={l-b:+.3f}   {gate} ===", flush=True)


if __name__ == "__main__":
    main()
