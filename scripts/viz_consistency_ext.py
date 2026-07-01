"""Enhancement visualizations for the consistency story (all requested):
  viz_cosine_heatmap_{outdoor,indoor}.png - per-patch overlap corr-cosine heatmap (where consistency rose)
  viz_feature_pca_indoor.png              - indoor full-pano PCA-3 seam coherence (more dramatic than the strip)
  viz_normal_consistency.png              - predicted-normal RGB + cross-tile angular-disagreement heatmap

Run: CUDA_VISIBLE_DEVICES=1 conda run -n pano python scripts/viz_consistency_ext.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data  # noqa: E402
import probe_seg_dinov3 as P  # noqa: E402
import diag_seam as D  # noqa: E402
import train_ssl as T  # noqa: E402
import viz_consistency as V  # noqa: E402
import probe_normal as PN  # noqa: E402
from encoder import PanoEncoder  # noqa: E402

DEVICE = P.DEVICE
DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures", "viz_consistency_ext")


def equator_hpair(geom):
    for i, (a, b) in enumerate(geom["pairs"]):
        if abs(geom["specs"][a][1]) < 1e-6 and abs(geom["specs"][b][1]) < 1e-6:
            return i
    return 0


def fig_cosine_heatmap(encs, rgb, geom, pair_idx, out, name):
    a, b = geom["pairs"][pair_idx]
    grid, valid, _ = geom["warps"][pair_idx]
    ya, pa = geom["specs"][a]; yb, pb = geom["specs"][b]
    imA = V.tile_rgb(rgb, ya, pa, geom["hfov"]); imB = V.tile_rgb(rgb, yb, pb, geom["hfov"])
    g = grid.cpu().numpy(); v = valid.cpu().numpy().astype(bool)
    fig, axes = plt.subplots(1, 1 + len(encs), figsize=(5 * (1 + len(encs)), 5))
    axes[0].imshow(imA); axes[0].set_title("tile A"); axes[0].axis("off")
    for ax, (tag, enc) in zip(axes[1:], encs):
        fa = V.tile_feat(enc, imA); fb = V.tile_feat(enc, imB)
        d, gh, gw = fa.shape
        FA = fa.reshape(d, -1).t(); FB = fb.reshape(d, -1).t()
        tb = V.true_b_cell(g, gh, gw)
        corr = (FA * FB[tb]).sum(1).numpy()
        m = np.full(gh * gw, np.nan); m[v] = corr[v]
        im = ax.imshow(m.reshape(gh, gw), cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"{tag}: overlap corr cosine (μ={np.nanmean(m):.2f})"); ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{name}: per-patch cross-tile cosine on the overlap (LoRA brighter = more consistent)",
                 y=1.02, fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print("saved", out)


def fig_normal_consistency(encs_clf, rgb, nrm, val, geom, pair_idx, out):
    a, b = geom["pairs"][pair_idx]
    grid, valid, _ = geom["warps"][pair_idx]
    v = valid.cpu().bool()
    fig, axes = plt.subplots(len(encs_clf), 3, figsize=(13, 4.4 * len(encs_clf)))
    for row, (tag, enc, clf) in enumerate(encs_clf):
        P.enc_patch = enc.patch
        packs, _ = PN.pano_tiles(enc, rgb, nrm, val, geom)
        preds = []
        for f, _, _ in packs:
            gh, gw, d = f.shape
            with torch.no_grad():
                p = F.normalize(clf(f.reshape(-1, d).to(DEVICE).float()), dim=1)
            preds.append(p.reshape(gh, gw, 3))
        na = ((preds[a].cpu().numpy() + 1) / 2).clip(0, 1)
        nb = ((preds[b].cpu().numpy() + 1) / 2).clip(0, 1)
        pa = preds[a].permute(2, 0, 1)[None]
        g = grid.to(DEVICE).view(1, 1, -1, 2)
        pbw = F.grid_sample(preds[b].permute(2, 0, 1)[None], g, align_corners=False)[0, :, 0, :].t()
        paf = pa[0].permute(1, 2, 0).reshape(-1, 3)
        cos = (F.normalize(paf, dim=1) * F.normalize(pbw, dim=1)).sum(1).clamp(-1, 1)
        ang = torch.rad2deg(torch.arccos(cos)).cpu().numpy()
        hm = np.full(gh * gw, np.nan); hm[v.cpu().numpy()] = ang[v.cpu().numpy()]
        mdis = np.nanmean(hm)
        axes[row, 0].imshow(na); axes[row, 0].set_title(f"{tag}: pred normal (tile A)"); axes[row, 0].axis("off")
        axes[row, 1].imshow(nb); axes[row, 1].set_title(f"{tag}: pred normal (tile B)"); axes[row, 1].axis("off")
        im = axes[row, 2].imshow(hm.reshape(gh, gw), cmap="inferno_r", vmin=0, vmax=60)
        axes[row, 2].set_title(f"{tag}: cross-tile angular disagree (μ={mdis:.1f}°)"); axes[row, 2].axis("off")
        plt.colorbar(im, ax=axes[row, 2], fraction=0.046, pad=0.04)
    fig.suptitle("Predicted-normal consistency on an overlapping pair (LoRA: lower angular disagreement)",
                 y=1.0, fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print("saved", out)


def main():
    frozen = PanoEncoder(model_id=P.MODEL, lora_rank=0).to(DEVICE).eval()
    lora = PanoEncoder(model_id=P.MODEL, adapter_path=T.CKPT).to(DEVICE).eval()
    encs = [("frozen", frozen), ("LoRA", lora)]

    # 1a. outdoor cosine heatmap
    P.configure("densepass"); D.DATASET = "densepass"; D.HFOV = 50.0; P.enc_patch = frozen.patch
    geom_out = T.build_geometry(frozen, 50.0, (0.0,))
    dp = data.list_densepass(); rgb_o = P.load_rgb_label(dp[75])[0]
    try:
        fig_cosine_heatmap(encs, rgb_o, geom_out, 0, os.path.join(DOCS, "viz_cosine_heatmap_outdoor.png"), "DensePASS")
    except Exception as ex:
        print("cos-heat outdoor FAIL", type(ex).__name__, str(ex)[:110])

    # indoor pano + geometry
    P.configure("stanford2d3d"); D.DATASET = "stanford2d3d"; D.HFOV = 65.0
    geom_in = T.build_geometry(frozen, 65.0, (-45.0, 0.0, 45.0))

    def area(f): return f.split("extracted_data/")[1].split("/")[0]
    s2d = data.list_erps("stanford2d3d")
    val_f = [f for f in s2d if "5" in area(f)]
    rgb_i = P.load_rgb_label(val_f[0])[0]
    hp = equator_hpair(geom_in)

    # 1b. indoor cosine heatmap
    try:
        fig_cosine_heatmap(encs, rgb_i, geom_in, hp, os.path.join(DOCS, "viz_cosine_heatmap_indoor.png"), "Stanford2D3D")
    except Exception as ex:
        print("cos-heat indoor FAIL", type(ex).__name__, str(ex)[:110])

    # 2. indoor full-pano PCA seam
    try:
        plan_in = D.tile_plan()
        V.fig_pca(encs, rgb_i, plan_in, os.path.join(DOCS, "viz_feature_pca_indoor.png"))
    except Exception as ex:
        print("pca indoor FAIL", type(ex).__name__, str(ex)[:110])

    # 3. normal-prediction consistency (train a linear normal probe per encoder)
    try:
        tr_f = [f for f in s2d if "5" not in area(f)][:60]
        clfs = {}
        for tag, enc in encs:
            P.enc_patch = enc.patch
            packs = [PN.pano_tiles(enc, *PN.load_rgb_normal(f), geom_in)[0] for f in tr_f]
            clfs[tag] = PN.train_probe(packs)
        rgbn, nrm, valn = PN.load_rgb_normal(val_f[0])
        fig_normal_consistency([(t, e, clfs[t]) for t, e in encs], rgbn, nrm, valn, geom_in, hp,
                               os.path.join(DOCS, "viz_normal_consistency.png"))
    except Exception as ex:
        print("normal-consistency FAIL", type(ex).__name__, str(ex)[:110])


if __name__ == "__main__":
    main()
