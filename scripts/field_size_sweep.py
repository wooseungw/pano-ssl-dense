"""What unified field size do the current E2P tiles naturally fill (no holes)?

Coverage = fraction of field cells that receive >=1 patch token, for tile 512/768, plan band
(current, no pole caps) vs full_sphere (with caps), across field resolutions. The natural size
is the finest grid that still stays well-covered (~>=0.8) — finer than that and the field
develops holes (tokens are sparser than cells). Encoder-free (geometry only).

Run: CUDA_VISIBLE_DEVICES= conda run --no-capture-output -n pano python scripts/field_size_sweep.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import anyres_e2p as a2p  # noqa: E402
import geometry as G  # noqa: E402

ERP_W, ERP_H = 2048, 1024            # coverage is a ratio -> absolute ERP size cancels; use 2K for speed
PATCH = 16
HFOV = 65.0
FIELDS = [(32, 64), (48, 96), (64, 128), (80, 160), (96, 192), (112, 224), (128, 256)]


def plans():
    band = a2p.plan_tiles("band", HFOV, HFOV, 0.25, pmax_deg=45.0)
    sphere = a2p.plan_tiles("full_sphere", HFOV, HFOV, 0.25)
    return {"band": band, "full_sphere": sphere}


def coordmaps(plan, tile):
    gh = tile // PATCH
    return [G.render_coordmap(ERP_H, ERP_W, tp.yaw_deg, tp.pitch_deg, HFOV, gh) for tp in plan]


def coverage(cms, HF, WF):
    cov = np.zeros(HF * WF, bool)
    for cm in cms:
        uf = np.clip((cm[..., 0] / ERP_W * WF).astype(int), 0, WF - 1)
        vf = np.clip((cm[..., 1] / ERP_H * HF).astype(int), 0, HF - 1)
        cov[(vf * WF + uf).reshape(-1)] = True
    return cov.mean()


def main():
    P = plans()
    for tile in (512, 768):
        for pname, plan in P.items():
            gh = tile // PATCH
            toks = len(plan) * gh * gh
            cms = coordmaps(plan, tile)
            print(f"\n=== tile {tile}² ({gh}×{gh}/tile) | plan={pname} | {len(plan)} tiles | "
                  f"{toks} tokens ===", flush=True)
            print(f"{'field (H×W)':>12} {'cells':>7} {'tok/cell':>9} {'coverage':>9}", flush=True)
            for HF, WF in FIELDS:
                cov = coverage(cms, HF, WF)
                cells = HF * WF
                tag = "  <- ~2K@p16" if (HF, WF) == (64, 128) else ("  <- ~4K@p16" if (HF, WF) == (128, 256) else "")
                print(f"{HF:5d}×{WF:<6d} {cells:7d} {toks/cells:9.2f} {cov:9.3f}{tag}", flush=True)


if __name__ == "__main__":
    main()
