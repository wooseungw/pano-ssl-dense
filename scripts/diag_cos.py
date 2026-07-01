"""Did the SSL adapter improve HELD-OUT overlap feature consistency (the thing it
optimized), even though the downstream probe barely moved? Measures mean obliquity-
weighted overlap cosine F_A(p)~F_B(Hp) on val panos, frozen vs LoRA.

If LoRA cosine >> frozen on val too, the adapter DID learn geometric consistency and
it just doesn't transfer to linear-probe argmax (headroom is laundered/irreducible).
If not, the train-time gain didn't generalize.

Run: CUDA_VISIBLE_DEVICES=0 conda run -n pano python scripts/diag_cos.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data  # noqa: E402
import probe_seg_dinov3 as P  # noqa: E402
import train_ssl as T  # noqa: E402
from encoder import PanoEncoder, normalize_tiles  # noqa: E402
from losses import warp_equivariance_loss  # noqa: E402

DEVICE = P.DEVICE
CKPT = T.CKPT


@torch.no_grad()
def mean_overlap_cos(enc, erp, geom):
    tiles = normalize_tiles(T.render_tiles(erp, geom["specs"], geom["hfov"]).to(DEVICE))
    feat = enc(tiles)
    cs, ws = [], []
    for (a, b), warp in zip(geom["pairs"], geom["warps"]):
        loss = warp_equivariance_loss(feat[a:a + 1], feat[b:b + 1], *warp)   # weighted mean (1-cos)
        w = (warp[1].float() * warp[2]).sum().item()
        cs.append((1.0 - loss.item()) * w); ws.append(w)
    return sum(cs) / max(sum(ws), 1e-9)


def val_files(ds):
    if ds == "out":
        dp = data.list_densepass(); return [(f, "out") for f in dp[int(len(dp) * 0.7):]]
    s2d = [f for f in data.list_erps("stanford2d3d")
           if "5" in f.split("extracted_data/")[1].split("/")[0]]
    return [(f, "in") for f in s2d[:40]]


def main():
    frozen = PanoEncoder(model_id=P.MODEL, lora_rank=0).to(DEVICE).eval()
    lora = PanoEncoder(model_id=P.MODEL, adapter_path=CKPT).to(DEVICE).eval()
    geom = {k: T.build_geometry(frozen, hf, pc) for k, (hf, pc) in T.DOMAINS.items()}
    print(f"{'domain':8s} {'frozen_cos':>11} {'lora_cos':>9} {'Δcos':>7}", flush=True)
    for tag, kind in [("outdoor", "out"), ("indoor", "in")]:
        fs, ls = [], []
        for f, k in val_files(kind):
            try:
                erp = T.load_erp(f, k)
            except Exception:
                continue
            fs.append(mean_overlap_cos(frozen, erp, geom[k]))
            ls.append(mean_overlap_cos(lora, erp, geom[k]))
        fm, lm = float(np.mean(fs)), float(np.mean(ls))
        print(f"{tag:8s} {fm:11.3f} {lm:9.3f} {lm-fm:+7.3f}", flush=True)


if __name__ == "__main__":
    main()
