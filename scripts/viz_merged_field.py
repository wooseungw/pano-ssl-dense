"""Visualize the MERGED feature field (the canonical unified representation): per-tile features
obliquity-weighted-scattered to a 64x128 ERP grid, then PCA-3 -> RGB. ViT-B vs ViT-L.

Size: field = (D, 64, 128) = (768|1024 channels) x 8192 cells = 2K-ERP @ patch16.

Run: CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n pano python scripts/viz_merged_field.py
Out: docs/figures/viz_merged_field/merged_field.png
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import anyres_e2p as a2p  # noqa: E402
import geometry as G  # noqa: E402
import data  # noqa: E402
from encoder import PanoEncoder, normalize_tiles  # noqa: E402

DEVICE = "cuda"
ERP_W, ERP_H = 2048, 1024
PATCH, HFOV, TILE = 16, 65.0, 512
HF, WF = 64, 128
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs/figures/viz_merged_field")
MODELS = {"ViT-B/16 (768d)": "facebook/dinov3-vitb16-pretrain-lvd1689m",
          "ViT-L/16 (1024d)": "facebook/dinov3-vitl16-pretrain-lvd1689m"}


def obliq(gh):
    cy = (np.arange(gh) + 0.5) * TILE / gh
    XX, YY = np.meshgrid(cy, cy)
    return G._offaxis_cos(XX, YY, TILE, HFOV).reshape(-1)


def cell_ids(plan, gh):
    ids = []
    for tp in plan:
        cm = G.render_coordmap(ERP_H, ERP_W, tp.yaw_deg, tp.pitch_deg, HFOV, gh)
        uf = np.clip((cm[..., 0] / ERP_W * WF).astype(int), 0, WF - 1)
        vf = np.clip((cm[..., 1] / ERP_H * HF).astype(int), 0, HF - 1)
        ids.append((vf * WF + uf).reshape(-1))
    return ids


@torch.no_grad()
def build_field(enc, erp_np, plan, ids):
    gh = TILE // enc.patch; w = obliq(gh); D = enc.dim
    fs = np.zeros((HF * WF, D), np.float32); ws = np.zeros(HF * WF, np.float32)
    for tp, c in zip(plan, ids):
        t = np.asarray(a2p.erp_to_pinhole_tile(erp_np, tp.yaw_deg, tp.pitch_deg, HFOV, TILE))
        x = normalize_tiles((torch.from_numpy(t).float().permute(2, 0, 1)[None] / 255.0).to(DEVICE))
        fmap = enc(x)[0].permute(1, 2, 0).reshape(-1, D).float().cpu().numpy()
        np.add.at(fs, c, w[:, None] * fmap); np.add.at(ws, c, w)
    cov = ws > 0; fs[cov] /= ws[cov][:, None]
    return fs, cov


def pca_rgb(fs, cov):
    X = fs[cov]; X = X - X.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    p = X @ Vt[:3].T; p = (p - p.min(0)) / (np.ptp(p, 0) + 1e-6)
    img = np.zeros((HF * WF, 3)); img[cov] = p
    return img.reshape(HF, WF, 3)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    plan = a2p.plan_tiles("full_sphere", HFOV, HFOV, 0.25)        # incl pole caps
    f = data.list_erps("stanford2d3d")[0]
    erp = np.array(Image.open(f).convert("RGB").resize((ERP_W, ERP_H), Image.BILINEAR))
    ids = cell_ids(plan, TILE // PATCH)
    print(f"plan=full_sphere tiles={len(plan)} field={HF}×{WF}={HF*WF} cells", flush=True)

    pca = {}; cov = None
    for name, mid in MODELS.items():
        enc = PanoEncoder(model_id=mid, lora_rank=0).to(DEVICE).eval()
        fs, cov = build_field(enc, erp, plan, ids)
        pca[name] = pca_rgb(fs, cov)
        print(f"  {name}: merged field = ({enc.dim}, {HF}, {WF})  coverage={cov.mean():.3f}", flush=True)
        del enc; torch.cuda.empty_cache()

    fig, ax = plt.subplots(2, 2, figsize=(18, 9))
    ax[0, 0].imshow(np.array(Image.fromarray(erp).resize((1024, 512)))); ax[0, 0].set_title("source ERP (2K)")
    ax[0, 1].imshow(cov.reshape(HF, WF), cmap="Greys_r", vmin=0, vmax=1)
    ax[0, 1].set_title(f"coverage 64×128  ({cov.mean():.0%})")
    names = list(MODELS)
    ax[1, 0].imshow(pca[names[0]]); ax[1, 0].set_title(f"merged field PCA-3 — {names[0]}  →  (768,64,128)")
    ax[1, 1].imshow(pca[names[1]]); ax[1, 1].set_title(f"merged field PCA-3 — {names[1]}  →  (1024,64,128)")
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Merged feature field — per-tile features obliquity-scattered to one 64×128 ERP grid",
                 fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUTDIR, "merged_field.png")
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
