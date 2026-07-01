"""Visualize, on a real 4K (4096x2048) ERP, the unified patch-16 feature field (128x256):
  (A) coverage at tile 512  -> holes (token budget 22x1024 < 32768 cells)
  (B) coverage at tile 768  -> filled (22x2304 > 32768)
  (C) the source ERP
  (D) DINOv3 output ATTENTION map (last-block CLS->patch, per tile) placed into the 128x256 field
  (E) the same attention overlaid on the ERP
  (F) PCA-3 of the placed FEATURE field (do per-tile features stitch into one coherent panorama?)

Run: CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n pano python scripts/viz_field_attention.py
Out: docs/figures/viz_field_attention/field_attention.png
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
from transformers import AutoModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import anyres_e2p as a2p  # noqa: E402
import geometry as G  # noqa: E402
import data  # noqa: E402
from encoder import normalize_tiles  # noqa: E402

MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"
ERP_W, ERP_H = 4096, 2048
PATCH = 16
HF, WF = ERP_H // PATCH, ERP_W // PATCH          # 128 x 256 = 32768 cells
HFOV = 65.0
DEVICE = "cuda"
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs/figures/viz_field_attention")


def indoor_plan():
    return a2p.plan_tiles("band", HFOV, HFOV, 0.25, pmax_deg=45.0)


def cell_ids(plan, tile):
    """Per-tile field-cell id for each patch center (coordmap at patch-grid res), + coverage."""
    gh = tile // PATCH
    ids, cov = [], np.zeros(HF * WF, bool)
    for tp in plan:
        cm = G.render_coordmap(ERP_H, ERP_W, tp.yaw_deg, tp.pitch_deg, HFOV, gh)  # (gh,gh,2) erp (x,y)
        uf = np.clip((cm[..., 0] / ERP_W * WF).astype(int), 0, WF - 1)
        vf = np.clip((cm[..., 1] / ERP_H * HF).astype(int), 0, HF - 1)
        c = (vf * WF + uf).reshape(-1)
        ids.append(c); cov[c] = True
    return ids, cov


@torch.no_grad()
def tile_forward(bk, tile_np, tile):
    x = normalize_tiles((torch.from_numpy(tile_np).float().permute(2, 0, 1)[None] / 255.0).to(DEVICE))
    gh = tile // PATCH; n = gh * gh
    try:
        out = bk(pixel_values=x, interpolate_pos_encoding=True, output_attentions=True)
        attn = out.attentions[-1]                       # (1, heads, T, T)
        a = attn[0, :, 0, -n:].mean(0)                  # CLS->patch, mean over heads
        mode = "DINOv3 CLS attention (last block)"
    except Exception:
        out = bk(pixel_values=x, interpolate_pos_encoding=True)
        a = None; mode = "DINOv3 CLS-cosine saliency (attn unavailable)"
    feat = out.last_hidden_state[0]                      # (T, D)
    patch = feat[-n:]
    if a is None:
        a = F.cosine_similarity(patch, feat[0][None], dim=1).clamp_min(0)
    a = a.float().cpu().numpy()
    a = (a - a.min()) / (np.ptp(a) + 1e-6)               # per-tile [0,1] so tiles are comparable
    return patch.float().cpu().numpy(), a, mode


def build_fields(bk, erp_np, plan, tile):
    ids, _ = cell_ids(plan, tile)
    fa = np.zeros(HF * WF); wa = np.zeros(HF * WF)
    ff = None; wf = np.zeros(HF * WF); mode = None
    for tp, c in zip(plan, ids):
        t = np.asarray(a2p.erp_to_pinhole_tile(erp_np, tp.yaw_deg, tp.pitch_deg, HFOV, tile))
        patch, amap, mode = tile_forward(bk, t, tile)
        if ff is None:
            ff = np.zeros((HF * WF, patch.shape[1]))
        np.add.at(fa, c, amap); np.add.at(wa, c, 1.0)
        np.add.at(ff, c, patch); np.add.at(wf, c, 1.0)
    cov = wa > 0
    fa[cov] /= wa[cov]; ff[cov] /= wf[cov][:, None]
    return fa.reshape(HF, WF), cov.reshape(HF, WF), ff, cov, mode


def pca_rgb(ff, cov):
    X = ff[cov]; X = X - X.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    p = X @ Vt[:3].T
    p = (p - p.min(0)) / (np.ptp(p, 0) + 1e-6)
    img = np.zeros((HF * WF, 3)); img[cov.reshape(-1)] = p
    return img.reshape(HF, WF, 3)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    plan = indoor_plan()
    f = data.list_erps("stanford2d3d")[0]
    erp_np = np.array(Image.open(f).convert("RGB").resize((ERP_W, ERP_H), Image.BILINEAR))
    erp_disp = np.array(Image.fromarray(erp_np).resize((1024, 512)))

    cov512 = cell_ids(plan, 512)[1].reshape(HF, WF); c512 = cov512.mean()
    cov768 = cell_ids(plan, 768)[1].reshape(HF, WF); c768 = cov768.mean()
    print(f"plan tiles={len(plan)}  field={HF}x{WF}={HF*WF} cells", flush=True)
    print(f"  tile512: tokens={len(plan)*(512//PATCH)**2} coverage={c512:.3f}", flush=True)
    print(f"  tile768: tokens={len(plan)*(768//PATCH)**2} coverage={c768:.3f}", flush=True)

    try:
        bk = AutoModel.from_pretrained(MODEL, attn_implementation="eager")   # needed for real attn weights
    except Exception:
        bk = AutoModel.from_pretrained(MODEL)
    bk = bk.to(DEVICE).eval()
    attn, covA, ff, covflat, mode = build_fields(bk, erp_np, plan, 768)
    pca = pca_rgb(ff, covflat)
    print(f"  attention mode: {mode}", flush=True)

    fig, ax = plt.subplots(2, 3, figsize=(22, 9))
    ax[0, 0].imshow(cov512, cmap="Greys_r", vmin=0, vmax=1)
    ax[0, 0].set_title(f"(A) coverage @128×256, tile 512²  →  {c512:.0%} (holes)")
    ax[0, 1].imshow(cov768, cmap="Greys_r", vmin=0, vmax=1)
    ax[0, 1].set_title(f"(B) coverage @128×256, tile 768²  →  {c768:.0%} (filled)")
    ax[0, 2].imshow(erp_disp)
    ax[0, 2].set_title("(C) source 4K ERP (4096×2048)")
    ax[1, 0].imshow(np.where(covA, attn, np.nan), cmap="inferno")
    ax[1, 0].set_title(f"(D) output attention field @128×256\n{mode}")
    ax[1, 1].imshow(erp_disp)
    ah = np.array(Image.fromarray((plt.cm.inferno(np.nan_to_num(np.where(covA, attn, 0)))[..., :3] * 255).astype(np.uint8)).resize((1024, 512)))
    ax[1, 1].imshow(ah, alpha=0.55)
    ax[1, 1].set_title("(E) attention overlaid on ERP")
    ax[1, 2].imshow(pca)
    ax[1, 2].set_title("(F) PCA-3 of placed FEATURE field (768²)")
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Unified patch-16 feature field on 4K ERP — {HF}×{WF}={HF*WF} cells, {len(plan)} E2P tiles",
                 fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(OUTDIR, "field_attention.png")
    fig.savefig(out, dpi=105, bbox_inches="tight"); plt.close(fig)
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
