"""Fan-mismatch diagnostic — does the geometric patch correspondence degrade toward the fan edges,
and would footprint-aware pooling fix it? (step 1 of fixing the overlap matching)

The overlap SSL matches A's patch CENTER to B's same-ray location (exact, no-parallax). But each patch's
sphere FOOTPRINT is fan-shaped: near the tile center it is compact, near the edges (high obliquity) it is
stretched. Two tiles see the same ray at DIFFERENT obliquity, so their patches summarize DIFFERENT solid
angles — the center matches, the footprint does not, worst at the fan edges. The loss CONCEDES this via the
obliquity weight min(cos th_A, cos th_B). This diagnostic measures the break and tests a footprint-pool fix.

Per geometrically-matched overlap patch pair (frozen DINOv3, S2D3D area_1, hfov65 3-ring):
  cos_point  = cos(F_A, grid_sample(F_B, matched-loc))         current point match
  cos_pool   = cos(F_A, grid_sample(avgpool_k(F_B), loc))      footprint-proxy (enlarge B's footprint)
  obliquity  = min(cos th_A, cos th_B)  (low = fan edge)
Binned by obliquity quartile: does cos_point fall at the fan edges, and does cos_pool recover it there?

Run: OPENCV_IO_ENABLE_OPENEXR=1 CUDA_VISIBLE_DEVICES=0 conda run -n pano python scripts/diag_fan_match.py
Knobs: NVA (panos, def 30), POOL (avg-pool kernel for the footprint proxy, def 3).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pointmap_fusion as PF  # noqa: E402
import data  # noqa: E402
from encoder import PanoEncoder  # noqa: E402

DEVICE = PF.DEVICE
TILE = PF.TILE
NVA = int(os.environ.get("NVA", 30))
POOL = int(os.environ.get("POOL", 3))
P = PF.P


def cos_rows(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return (a * b).sum(1)


def main():
    frozen = PanoEncoder(model_id=P.MODEL, lora_rank=0).to(DEVICE).eval()
    P.configure("stanford2d3d"); P.TILE = TILE; P.enc_patch = frozen.patch
    geom = PF.T.build_geometry(frozen, 65.0, (-45.0, 0.0, 45.0))
    s2d = data.list_erps("stanford2d3d")

    def area(f):
        return f.split("extracted_data/")[1].split("/")[0]
    va = [f for f in s2d if area(f) == "area_1"][:NVA]
    print(f"fan-match diag: frozen DINOv3, {len(va)} area_1 panos, tiles/pano={len(geom['specs'])}, "
          f"pool_k={POOL}", flush=True)

    COS_PT, COS_PL, OBL = [], [], []
    dummy = np.zeros((PF.H, PF.W), np.float32)
    for f in va:
        rgb = np.array(Image.open(f).convert("RGB").resize((PF.W, PF.H), Image.BILINEAR))
        tiles = [PF.tile_pack(frozen, rgb, dummy, dummy, y, p, geom["hfov"]) for (y, p) in geom["specs"]]
        feats = [t[0] for t in tiles]                                    # (gh,gw,D) each
        D = feats[0].shape[-1]
        for (a, b), (grid, valid, weight) in zip(geom["pairs"], geom["warps"]):
            v = valid.cpu().numpy().astype(bool)
            if v.sum() < 16:
                continue
            g = grid.cpu().view(1, 1, -1, 2)
            Fb = feats[b].permute(2, 0, 1)[None].float()                 # (1,D,gh,gw)
            Fb_pool = F.avg_pool2d(Fb, POOL, stride=1, padding=POOL // 2)
            fb_pt = F.grid_sample(Fb, g, align_corners=False)[0, :, 0, :].t().numpy()
            fb_pl = F.grid_sample(Fb_pool, g, align_corners=False)[0, :, 0, :].t().numpy()
            fa = feats[a].reshape(-1, D).numpy()
            COS_PT.append(cos_rows(fa, fb_pt)[v])
            COS_PL.append(cos_rows(fa, fb_pl)[v])
            OBL.append(weight.cpu().numpy()[v])
    cpt = np.concatenate(COS_PT); cpl = np.concatenate(COS_PL); ob = np.concatenate(OBL)
    n = len(cpt)

    print(f"\nmatched overlap cells N={n}", flush=True)
    print(f"overall: point cos={cpt.mean():.3f}   footprint-pool cos={cpl.mean():.3f}   "
          f"Δ={cpl.mean()-cpt.mean():+.3f}", flush=True)
    q = np.quantile(ob, [0.0, 0.25, 0.5, 0.75, 1.0])
    print(f"\nby obliquity min(cos) quartile (LOW = fan edge / stretched):", flush=True)
    print(f"{'obliq bin':22s}{'n':>8}{'point cos':>11}{'pool cos':>10}{'pool-point':>11}", flush=True)
    for i in range(4):
        m = (ob >= q[i]) & (ob <= q[i + 1])
        tag = "  (fan EDGE)" if i == 0 else ("  (center)" if i == 3 else "")
        print(f"[{q[i]:.2f},{q[i+1]:.2f}]{tag:12s}{int(m.sum()):>8}{cpt[m].mean():>11.3f}"
              f"{cpl[m].mean():>10.3f}{cpl[m].mean()-cpt[m].mean():>+11.3f}", flush=True)

    edge = ob <= q[1]; ctr = ob >= q[3]
    drop = cpt[ctr].mean() - cpt[edge].mean()
    pool_gain_edge = cpl[edge].mean() - cpt[edge].mean()
    print(f"\ncenter→edge point-match drop = {drop:+.3f}   |   footprint-pool gain AT edge = {pool_gain_edge:+.3f}", flush=True)
    if drop < 0.03:
        v = "MATCH ROBUST — point-match barely degrades toward the fan edge; footprint fix low-value for consistency."
    elif pool_gain_edge > 0.01:
        v = (f"FAN-MISMATCH REAL & POOL HELPS — match drops {drop:.3f} center→edge, and footprint pooling "
             f"recovers {pool_gain_edge:+.3f} at the edge. Build proper footprint-aware (equal-area) matching.")
    else:
        v = (f"FAN-MISMATCH REAL but crude pool does NOT help ({pool_gain_edge:+.3f}) — the edge break is real "
             f"(drop {drop:.3f}); needs a principled equal-area resample, not a blur. Or the loss's obliquity "
             "weight already handles it (it down-weights exactly these low-obliq cells).")
    print(f"\nVERDICT: {v}", flush=True)


if __name__ == "__main__":
    main()
