"""Backbone probe (B): how overlap-consistent are frozen features across encoders?

Compares corresponded-cos / random-cos / top-1 retrieval in the E2P overlap for
DINOv2 vs DINOv2-with-registers vs DINOv3 (registers/DINOv3 are expected to reduce the
position-artifact + high-norm-token problems that the warp loss would otherwise fight).
Uses the verified WarpField correspondence from geometry.py. Gated models are skipped.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import anyres_e2p as a2p  # noqa: E402
import data  # noqa: E402
import geometry as G  # noqa: E402
from encoder import PanoEncoder, normalize_tiles  # noqa: E402

ERP_W, ERP_H = 2048, 1024
HFOV, YAW_A, YAW_B, PITCH = 90.0, 0.0, 60.0, 0.0
N_ERP = 12
MODELS = [
    "facebook/dinov2-base",
    "facebook/dinov2-with-registers-base",
    "facebook/dinov3-vitb16-pretrain-lvd1689m",
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def probe(model_id: str, files):
    enc = PanoEncoder(model_id=model_id, lora_rank=0).to(DEVICE).eval()
    size = (518 // enc.patch) * enc.patch
    ca = G.render_coordmap(ERP_H, ERP_W, YAW_A, PITCH, HFOV, size)
    cb = G.render_coordmap(ERP_H, ERP_W, YAW_B, PITCH, HFOV, size)
    wf = G.warp_field_from_coordmaps(ca, cb, enc.patch, HFOV, erp_w=ERP_W, dst_stride=3)
    grid = torch.from_numpy(wf.grid).to(DEVICE)
    valid = torch.from_numpy(wf.valid).to(DEVICE)
    gh, gw = wf.grid_hw
    # discrete B cell each A cell maps to (for retrieval)
    bx = torch.round((grid[:, 0] + 1) / 2 * gw - 0.5).clamp(0, gw - 1).long()
    by = torch.round((grid[:, 1] + 1) / 2 * gh - 0.5).clamp(0, gh - 1).long()
    b_cell = (by * gw + bx)

    corr, rand, retr = [], [], []
    for f in files:
        erp = np.array(Image.open(f).convert("RGB").resize((ERP_W, ERP_H)))
        ta = torch.from_numpy(np.asarray(a2p.erp_to_pinhole_tile(erp, YAW_A, PITCH, HFOV, size)))
        tb = torch.from_numpy(np.asarray(a2p.erp_to_pinhole_tile(erp, YAW_B, PITCH, HFOV, size)))
        x = torch.stack([ta, tb]).float().permute(0, 3, 1, 2) / 255.0
        feat = enc(normalize_tiles(x.to(DEVICE)))                       # (2,D,gh,gw)
        fa = F.normalize(feat[0].reshape(feat.shape[1], -1).t(), dim=-1)  # (N,D)
        fb = F.normalize(feat[1].reshape(feat.shape[1], -1).t(), dim=-1)
        fb_warp = F.normalize(F.grid_sample(
            feat[1:2], grid.view(1, 1, -1, 2).float(), align_corners=False)[0, :, 0, :].t(), dim=-1)
        idx = valid
        corr.append((fa[idx] * fb_warp[idx]).sum(-1).mean().item())
        perm = torch.randperm(fb.shape[0], device=DEVICE)[: idx.sum()]
        rand.append((fa[idx] * fb[perm]).sum(-1).mean().item())
        sims = fa[idx] @ fb.t()
        retr.append((sims.argmax(1) == b_cell[idx]).float().mean().item())
    return float(np.mean(corr)), float(np.mean(rand)), float(np.mean(retr)), enc.patch


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "stanford2d3d"
    files = data.list_erps(name, N_ERP)
    print(f"device={DEVICE} dataset={name} erps={len(files)} res={ERP_W}x{ERP_H}\n"
          f"{'model':40s} {'patch':>5} {'corr':>6} {'rand':>6} {'retr':>6}")
    for m in MODELS:
        try:
            c, r, rt, p = probe(m, files)
            print(f"{m:40s} {p:5d} {c:6.3f} {r:6.3f} {rt:6.3f}")
        except Exception as e:
            print(f"{m:40s}  SKIP ({type(e).__name__}: {str(e)[:60]})")


if __name__ == "__main__":
    main()
