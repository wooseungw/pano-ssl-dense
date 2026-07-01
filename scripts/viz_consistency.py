"""Visualize + verify the CONSISTENCY metrics (frozen vs LoRA-SSL), encoder-only.

Produces three figures in docs/:
  viz_correspondence.png  - feature-NN match lines on an overlapping tile pair (retrieval)
  viz_feature_pca.png     - PCA-3 feature panorama stitched from the tile ring (seam coherence)
  viz_cosine_dist.png     - corr vs rand overlap-cosine histograms (lift)

Run: CUDA_VISIBLE_DEVICES=0 conda run -n pano python scripts/viz_consistency.py
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
from encoder import PanoEncoder, normalize_tiles  # noqa: E402

DEVICE = P.DEVICE
DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures", "viz_consistency")
TILE = 512


def tile_rgb(rgb, yaw, pitch, hfov):
    return np.asarray(P.a2p.erp_to_pinhole_tile(rgb, yaw, pitch, hfov, TILE))


@torch.no_grad()
def tile_feat(enc, img):
    x = torch.from_numpy(img).float().permute(2, 0, 1)[None] / 255.0
    f = P.dense(enc, normalize_tiles(x.to(DEVICE)))[0]          # (D,gh,gw)
    return F.normalize(f, dim=0).cpu()                          # normalized


def true_b_cell(grid, gh, gw):
    bx = np.clip(((grid[:, 0] + 1) / 2 * gw - 0.5).round().astype(int), 0, gw - 1)
    by = np.clip(((grid[:, 1] + 1) / 2 * gh - 0.5).round().astype(int), 0, gh - 1)
    return by * gw + bx


# ---------------------------------------------------------------- figure 1
def fig_correspondence(encs, rgb, geom, pair_idx, out, n_q=14):
    (a, b) = geom["pairs"][pair_idx]
    grid, valid, weight = geom["warps"][pair_idx]
    ya, pa = geom["specs"][a]; yb, pb = geom["specs"][b]
    imA = tile_rgb(rgb, ya, pa, geom["hfov"]); imB = tile_rgb(rgb, yb, pb, geom["hfov"])
    grid = grid.cpu().numpy(); valid = valid.cpu().numpy().astype(bool)
    fig, axes = plt.subplots(len(encs), 1, figsize=(11, 5.2 * len(encs)))
    if len(encs) == 1:
        axes = [axes]
    gap = 40
    for ax, (tag, enc) in zip(axes, encs):
        fa = tile_feat(enc, imA); fb = tile_feat(enc, imB)
        d, gh, gw = fa.shape
        FA = fa.reshape(d, -1).t(); FB = fb.reshape(d, -1).t()
        tb = true_b_cell(grid, gh, gw)
        vcells = np.where(valid)[0]
        rng = np.random.RandomState(0)
        qs = rng.choice(vcells, size=min(n_q, len(vcells)), replace=False)
        sims = FA[qs] @ FB.t()
        nn = sims.argmax(1).numpy()
        canvas = np.concatenate([imA, np.full((TILE, gap, 3), 255, np.uint8), imB], 1)
        ax.imshow(canvas); ax.set_xticks([]); ax.set_yticks([])
        patch = TILE // gh
        hit = 0
        for q, n in zip(qs, nn):
            ay, ax_ = (q // gw) * patch + patch // 2, (q % gw) * patch + patch // 2
            ny, nx_ = (n // gw) * patch + patch // 2, (n % gw) * patch + patch // 2 + TILE + gap
            ok = (abs(n // gw - tb[q] // gw) <= 1) and (abs(n % gw - tb[q] % gw) <= 1)
            hit += ok
            c = "#1a1" if ok else "#d11"
            ax.plot([ax_, nx_], [ay, ny], "-", color=c, lw=1.4, alpha=0.85)
            ax.plot(ax_, ay, "o", color=c, ms=4)
            ax.plot(nx_, ny, "x", color=c, ms=6)
        ax.set_title(f"{tag}: feature-NN correspondence  (retrieval {hit}/{len(qs)} correct)  "
                     f"green=correct red=wrong   [tile A | tile B]", fontsize=11)
    fig.suptitle("Cross-tile correspondence by feature similarity — frozen vs LoRA-SSL", y=1.0, fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print("saved", out)


# ---------------------------------------------------------------- figure 2
def feature_erp(enc, rgb, plan, hf, wf):
    D_ = enc.dim
    fsum = torch.zeros(hf * wf, D_); cov = np.zeros(hf * wf, int)
    for tp in plan:
        img = tile_rgb(rgb, tp.yaw_deg, tp.pitch_deg, D.HFOV)
        f = tile_feat(enc, img); d, gh, gw = f.shape
        cid, _ = D.coord_grid(rgb.shape[:2], tp, gh, gw)
        fm = f.reshape(d, -1).t()
        for k, c in enumerate(cid.reshape(-1)):
            fsum[c] += fm[k]; cov[c] += 1
    m = cov >= 1
    feat = (fsum[m] / torch.from_numpy(cov[m]).float()[:, None]).numpy()
    return feat, m, (hf, wf)


def fig_pca(encs, rgb, plan, out):
    hf, wf = rgb.shape[0] // P.enc_patch, rgb.shape[1] // P.enc_patch
    fig, axes = plt.subplots(len(encs) + 1, 1, figsize=(11, 2.6 * (len(encs) + 1)))
    axes[0].imshow(rgb); axes[0].set_title("ERP RGB", fontsize=10)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    for ax, (tag, enc) in zip(axes[1:], encs):
        P.enc_patch = enc.patch
        feat, m, (hf, wf) = feature_erp(enc, rgb, plan, hf, wf)
        f = feat - feat.mean(0)
        u, s, vt = np.linalg.svd(f, full_matrices=False)
        rgb3 = (u[:, :3] * s[:3])
        rgb3 = (rgb3 - rgb3.min(0)) / (np.ptp(rgb3, 0) + 1e-6)
        canvas = np.zeros((hf * wf, 3)); canvas[m] = rgb3
        ax.imshow(canvas.reshape(hf, wf, 3)); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{tag}: PCA-3 feature panorama (look at tile seams)", fontsize=10)
    fig.suptitle("Stitched feature panorama — seam coherence (LoRA smoother across tile boundaries)",
                 y=1.0, fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print("saved", out)


# ---------------------------------------------------------------- figure 3
def fig_cosine_dist(encs, panos_geom, out):
    fig, axes = plt.subplots(1, len(panos_geom), figsize=(7 * len(panos_geom), 4.5))
    if len(panos_geom) == 1:
        axes = [axes]
    for ax, (name, cache, geom) in zip(axes, panos_geom):
        for tag, enc in encs:
            corr, rand = [], []
            for rgb in cache:
                feats = []
                for (yaw, pitch) in geom["specs"]:
                    feats.append(tile_feat(enc, tile_rgb(rgb, yaw, pitch, geom["hfov"])))
                for (a, b), (grid, valid, weight) in zip(geom["pairs"], geom["warps"]):
                    v = valid.cpu().numpy().astype(bool)
                    if v.sum() < 4:
                        continue
                    d, gh, gw = feats[a].shape
                    FA = feats[a].reshape(d, -1).t(); FB = feats[b].reshape(d, -1).t()
                    tb = true_b_cell(grid.cpu().numpy(), gh, gw)
                    sims = FA[v] @ FB.t()
                    corr.append(sims.numpy()[np.arange(v.sum()), tb[v]])
                    rand.append(sims.mean(1).numpy())
            corr = np.concatenate(corr); rand = np.concatenate(rand)
            ls = "-" if tag == "LoRA" else "--"
            ax.hist(corr, bins=40, range=(-.2, 1), histtype="step", lw=2, ls=ls,
                    color="#48c", label=f"{tag} corr (μ={corr.mean():.2f})")
            ax.hist(rand, bins=40, range=(-.2, 1), histtype="step", lw=2, ls=ls,
                    color="#c44", label=f"{tag} rand (μ={rand.mean():.2f})")
        ax.set_title(f"{name}: overlap cosine (corr vs rand)"); ax.set_xlabel("cosine"); ax.legend(fontsize=8)
    fig.suptitle("Overlap feature cosine — LoRA shifts CORRESPONDENT pairs right without inflating RANDOM (lift↑)",
                 y=1.02, fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print("saved", out)


def main():
    frozen = PanoEncoder(model_id=P.MODEL, lora_rank=0).to(DEVICE).eval()
    lora = PanoEncoder(model_id=P.MODEL, adapter_path=T.CKPT).to(DEVICE).eval()
    encs = [("frozen", frozen), ("LoRA", lora)]
    P.enc_patch = frozen.patch

    # outdoor pano for correspondence + PCA
    P.configure("densepass"); D.DATASET = "densepass"; D.HFOV = 50.0
    geom_out = T.build_geometry(frozen, 50.0, (0.0,))
    dp = data.list_densepass()
    rgb_o, _ = P.load_rgb_label(dp[75])
    try:
        fig_correspondence(encs, rgb_o, geom_out, 0, os.path.join(DOCS, "viz_correspondence.png"))
    except Exception as ex:
        print("correspondence FAIL", type(ex).__name__, str(ex)[:120])
    try:
        plan_o = D.tile_plan()
        fig_pca(encs, rgb_o, plan_o, os.path.join(DOCS, "viz_feature_pca.png"))
    except Exception as ex:
        print("pca FAIL", type(ex).__name__, str(ex)[:120])

    # histograms over a few val panos, outdoor + indoor
    P.configure("stanford2d3d")
    geom_in = T.build_geometry(frozen, 65.0, (-45.0, 0.0, 45.0))
    s2d = [f for f in data.list_erps("stanford2d3d")
           if "5" in f.split("extracted_data/")[1].split("/")[0]][:6]
    cache_in = [P.load_rgb_label(f)[0] for f in s2d]
    P.configure("densepass"); D.HFOV = 50.0
    cache_out = [P.load_rgb_label(f)[0] for f in dp[70:78]]
    try:
        fig_cosine_dist(encs, [("DensePASS", cache_out, geom_out), ("Stanford2D3D", cache_in, geom_in)],
                        os.path.join(DOCS, "viz_cosine_dist.png"))
    except Exception as ex:
        print("cosine_dist FAIL", type(ex).__name__, str(ex)[:120])


if __name__ == "__main__":
    main()
