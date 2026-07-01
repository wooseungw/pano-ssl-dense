"""Visualize the PIFu query grid: sample the SAME panorama's implicit feature field at increasing
query resolutions (16×32 → 128×256) and PCA-3 each (shared basis). Shows two things:
  (1) the decoder output grid is FREE — any HQ×WQ you query.
  (2) beyond ~64×128 (tile feature density ~2°/cell) finer queries add NO detail (just smoother) —
      the detail ceiling. Tile feats encoded ONCE; coordmaps rendered ONCE; only the binning changes.

Run: CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n pano python scripts/viz_query_grid.py
Out: docs/figures/viz_query_grid/query_grid.png
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import anyres_e2p as a2p  # noqa: E402
import geometry as G  # noqa: E402
import data  # noqa: E402
from encoder import PanoEncoder, normalize_tiles  # noqa: E402

DEVICE = "cuda"
ERP_W, ERP_H = 2048, 1024
FOV, TILE, OUT = 65.0, 512, 256
RES = [(16, 32), (32, 64), (64, 128), (128, 256)]
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs/figures/viz_query_grid")


@torch.no_grad()
def encode_tiles(enc, erp, plan):
    feats, cms = [], []
    for tp in plan:
        t = np.asarray(a2p.erp_to_pinhole_tile(erp, tp.yaw_deg, tp.pitch_deg, FOV, TILE))
        x = normalize_tiles((torch.from_numpy(t).float().permute(2, 0, 1)[None] / 255.0).to(DEVICE))
        feats.append(enc(x))                                  # (1,D,gh,gw)
        cms.append(G.render_coordmap(ERP_H, ERP_W, tp.yaw_deg, tp.pitch_deg, FOV, OUT))
    return feats, cms


def sample_at(feats, cms, HQ, WQ, D):
    """PIFu-sample the implicit field at an arbitrary HQ×WQ query grid."""
    rr, cc = np.mgrid[0:OUT, 0:OUT]
    u_all = (cc / (OUT - 1) * 2 - 1).reshape(-1); v_all = (rr / (OUT - 1) * 2 - 1).reshape(-1)
    w_all = G._offaxis_cos(cc.reshape(-1).astype(float), rr.reshape(-1).astype(float), OUT, FOV)
    NC = HQ * WQ
    pf = torch.zeros(NC, D, device=DEVICE); ws = torch.zeros(NC, device=DEVICE)
    for feat, cm in zip(feats, cms):
        uf = np.clip((cm[..., 0] / ERP_W * WQ).astype(int), 0, WQ - 1).reshape(-1)
        vf = np.clip((cm[..., 1] / ERP_H * HQ).astype(int), 0, HQ - 1).reshape(-1)
        cell = vf * WQ + uf
        us = np.zeros(NC); vs = np.zeros(NC); wsum = np.zeros(NC); cnt = np.zeros(NC)
        np.add.at(us, cell, u_all); np.add.at(vs, cell, v_all)
        np.add.at(wsum, cell, w_all); np.add.at(cnt, cell, 1.0)
        cov = cnt > 0; idx = np.where(cov)[0]; cnt_ = np.maximum(cnt[idx], 1)
        uv = np.stack([us[idx] / cnt_, vs[idx] / cnt_], 1)
        g = torch.from_numpy(uv).float().view(1, 1, -1, 2).to(DEVICE)
        samp = F.grid_sample(feat, g, mode="bilinear", align_corners=False)[0, :, 0, :].t()   # (n,D)
        ci = torch.from_numpy(idx).long().to(DEVICE); wt = torch.from_numpy(wsum[idx] / cnt_).float().to(DEVICE)
        pf.index_add_(0, ci, wt[:, None] * samp); ws.index_add_(0, ci, wt)
    cov = (ws > 0).cpu().numpy()
    pf = (pf / ws.clamp_min(1e-6)[:, None]).cpu().numpy()
    return pf.reshape(HQ, WQ, D), cov.reshape(HQ, WQ)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    enc = PanoEncoder(model_id="facebook/dinov3-vitb16-pretrain-lvd1689m", lora_rank=0).to(DEVICE).eval()
    D = enc.dim
    plan = a2p.plan_tiles("full_sphere", FOV, FOV, 0.25)
    f = data.list_erps("stanford2d3d")[0]
    erp = np.array(Image.open(f).convert("RGB").resize((ERP_W, ERP_H), Image.BILINEAR))
    feats, cms = encode_tiles(enc, erp, plan)
    print(f"encoded {len(plan)} tiles; sampling at {RES}", flush=True)

    fields = {(h, w): sample_at(feats, cms, h, w, D) for (h, w) in RES}
    # shared PCA basis from the finest field
    ff, fc = fields[RES[-1]]
    X = ff[fc]; mu = X.mean(0); _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    proj_fin = (X - mu) @ Vt[:3].T; lo = proj_fin.min(0); hi = proj_fin.max(0)

    def pca_img(pf, cov):
        p = (pf[cov] - mu) @ Vt[:3].T; p = (p - lo) / (hi - lo + 1e-6)
        img = np.zeros((*cov.shape, 3)); img[cov] = np.clip(p, 0, 1)
        return img

    fig, ax = plt.subplots(1, len(RES) + 1, figsize=(4 * (len(RES) + 1), 4))
    ax[0].imshow(np.array(Image.fromarray(erp).resize((512, 256)))); ax[0].set_title("source ERP")
    for k, (h, w) in enumerate(RES):
        pf, cov = fields[(h, w)]
        ax[k + 1].imshow(pca_img(pf, cov), interpolation="nearest")
        cap = "  ← sweet spot" if (h, w) == (64, 128) else ("  ← finer ≈ no new detail" if h > 64 else "")
        ax[k + 1].set_title(f"query {h}×{w}  ({h*w} cells){cap}")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("PIFu query grid — same implicit field sampled at any resolution (shared PCA basis); "
                 "detail ceiling ≈ tile feature density (~2°/cell)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(OUTDIR, "query_grid.png")
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"saved -> {out}", flush=True)
    for (h, w), (_, cov) in fields.items():
        print(f"  {h}×{w}: coverage {cov.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
