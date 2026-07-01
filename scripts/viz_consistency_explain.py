"""Self-explanatory consistency figures: show the INPUT tiles (A=reference, B=compared),
WHERE they come from on the ERP, WHERE the overlap is (highlighted on both tiles), and the
frozen-vs-LoRA consistency metric overlaid ON the overlap region.

Overwrites:
  viz_correspondence.png       - ERP context + A|B tiles, overlap shaded, feature-NN match lines
  viz_cosine_heatmap_{out,in}.png - ERP + tile A/B (overlap boxed) + cosine overlaid on overlap
  viz_normal_consistency.png   - ERP + tile A/B + per-encoder normal maps + angular-disagree overlay

Run: CUDA_VISIBLE_DEVICES=1 conda run -n pano python scripts/viz_consistency_explain.py
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
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data  # noqa: E402
import probe_seg_dinov3 as P  # noqa: E402
import diag_seam as D  # noqa: E402
import train_ssl as T  # noqa: E402
import viz_consistency as V  # noqa: E402
import probe_normal as PN  # noqa: E402
from encoder import PanoEncoder  # noqa: E402

DEVICE = P.DEVICE
DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures", "viz_consistency_explain")
TILE = 512


def equator_hpair(geom):
    for i, (a, b) in enumerate(geom["pairs"]):
        if abs(geom["specs"][a][1]) < 1e-6 and abs(geom["specs"][b][1]) < 1e-6:
            return i
    return 0


def draw_erp(ax, erp, specs_ab, hfov):
    H, W = erp.shape[:2]
    ax.imshow(erp)
    for (yaw, pitch), lab, col in zip(specs_ab, ["A (reference)", "B (compared)"], ["#0cf", "#fc0"]):
        cx = (yaw + 180) / 360 * W; cy = (90 - pitch) / 180 * H
        hw = hfov / 2 / 360 * W; hh = hfov / 2 / 180 * H
        ax.add_patch(Rectangle((cx - hw, cy - hh), 2 * hw, 2 * hh, fill=False, ec=col, lw=2.5))
        ax.text(cx, cy - hh - 8, lab, color=col, ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("input ERP panorama — perspective tiles A & B are cut from the boxed regions", fontsize=10)
    ax.axis("off")


def shade_overlap(ax, img, cells, gh, gw, color, x0=0):
    patch = TILE // gh
    m = np.kron(cells.reshape(gh, gw).astype(float), np.ones((patch, patch)))
    ov = np.zeros((TILE, TILE, 4)); ov[..., 0], ov[..., 1], ov[..., 2] = color; ov[..., 3] = m * 0.4
    ax.imshow(img, extent=(x0, x0 + TILE, TILE, 0))
    ax.imshow(ov, extent=(x0, x0 + TILE, TILE, 0))
    ys, xs = np.where(cells.reshape(gh, gw) > 0)
    if len(ys):
        ax.add_patch(Rectangle((x0 + xs.min() * patch, ys.min() * patch),
                                (xs.max() - xs.min() + 1) * patch, (ys.max() - ys.min() + 1) * patch,
                                fill=False, ec="yellow", lw=2))


def heat_on_tile(ax, img, vals, valid, gh, gw, title, cmap, vmin, vmax):
    patch = TILE // gh
    Hh = np.full(gh * gw, np.nan); Hh[valid] = vals[valid]
    Hu = np.kron(Hh.reshape(gh, gw), np.ones((patch, patch)))
    cm = plt.get_cmap(cmap).copy(); cm.set_bad(alpha=0)
    ax.imshow(img); im = ax.imshow(np.ma.masked_invalid(Hu), cmap=cm, vmin=vmin, vmax=vmax, alpha=0.9)
    ys, xs = np.where(valid.reshape(gh, gw))
    if len(ys):
        ax.add_patch(Rectangle((xs.min() * patch, ys.min() * patch),
                               (xs.max() - xs.min() + 1) * patch, (ys.max() - ys.min() + 1) * patch,
                               fill=False, ec="yellow", lw=2))
    ax.set_title(title, fontsize=10); ax.axis("off")
    return im


# ------------------------------------------------------------------ cosine
def fig_cosine(encs, erp, geom, pi, out, name):
    a, b = geom["pairs"][pi]; grid, valid, _ = geom["warps"][pi]
    ya, pa = geom["specs"][a]; yb, pb = geom["specs"][b]
    imA = V.tile_rgb(erp, ya, pa, geom["hfov"]); imB = V.tile_rgb(erp, yb, pb, geom["hfov"])
    g = grid.cpu().numpy(); v = valid.cpu().numpy().astype(bool)
    fig = plt.figure(figsize=(15, 7.5))
    gs = GridSpec(2, 4, height_ratios=[1.05, 1.4], figure=fig)
    draw_erp(fig.add_subplot(gs[0, :]), erp, [(ya, pa), (yb, pb)], geom["hfov"])
    axA = fig.add_subplot(gs[1, 0]); axB = fig.add_subplot(gs[1, 1])
    gh = gw = int(np.sqrt(v.shape[0]))
    tb = V.true_b_cell(g, gh, gw)
    bcells = np.zeros(gh * gw, bool); bcells[tb[v]] = True
    shade_overlap(axA, imA, v.astype(float), gh, gw, (0, 0.8, 1)); axA.set_title("tile A (overlap shaded)", fontsize=10); axA.axis("off")
    shade_overlap(axB, imB, bcells.astype(float), gh, gw, (1, 0.8, 0)); axB.set_title("tile B (overlap shaded)", fontsize=10); axB.axis("off")
    for ax, (tag, enc) in zip([fig.add_subplot(gs[1, 2]), fig.add_subplot(gs[1, 3])], encs):
        fa = V.tile_feat(enc, imA); fb = V.tile_feat(enc, imB)
        d = fa.shape[0]; FA = fa.reshape(d, -1).t(); FB = fb.reshape(d, -1).t()
        corr = (FA * FB[tb]).sum(1).numpy()
        im = heat_on_tile(ax, imA, corr, v, gh, gw, f"{tag}: cross-tile cosine on overlap (μ={corr[v].mean():.2f})",
                          "viridis", 0, 1)
    fig.colorbar(im, ax=fig.axes[-1], fraction=0.046, pad=0.04)
    fig.suptitle(f"{name}: cross-tile feature consistency on the overlap (brighter = A & B agree)", y=1.0, fontsize=13)
    fig.tight_layout(); fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig); print("saved", out)


# ------------------------------------------------------------------ normal
def fig_normal(encs_clf, erp, nrm, val, geom, pi, out):
    a, b = geom["pairs"][pi]; grid, valid, _ = geom["warps"][pi]
    ya, pa = geom["specs"][a]; yb, pb = geom["specs"][b]
    imA = V.tile_rgb(erp, ya, pa, geom["hfov"]); imB = V.tile_rgb(erp, yb, pb, geom["hfov"])
    v = valid.cpu().numpy().astype(bool); gh = gw = int(np.sqrt(v.shape[0]))
    tb = V.true_b_cell(grid.cpu().numpy(), gh, gw)
    bcells = np.zeros(gh * gw, bool); bcells[tb[v]] = True
    fig = plt.figure(figsize=(13, 4.3 * (len(encs_clf) + 1)))
    gs = GridSpec(len(encs_clf) + 1, 3, figure=fig)
    axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1]); axE = fig.add_subplot(gs[0, 2])
    shade_overlap(axA, imA, v.astype(float), gh, gw, (0, 0.8, 1)); axA.set_title("input tile A (overlap shaded)", fontsize=10); axA.axis("off")
    shade_overlap(axB, imB, bcells.astype(float), gh, gw, (1, 0.8, 0)); axB.set_title("input tile B (overlap shaded)", fontsize=10); axB.axis("off")
    draw_erp(axE, erp, [(ya, pa), (yb, pb)], geom["hfov"])
    for row, (tag, enc, clf) in enumerate(encs_clf, start=1):
        P.enc_patch = enc.patch
        packs, _ = PN.pano_tiles(enc, erp_rgb_for_normal(erp), nrm, val, geom)
        preds = []
        for f, _, _ in packs:
            with torch.no_grad():
                p = F.normalize(clf(f.reshape(-1, f.shape[-1]).to(DEVICE).float()), dim=1)
            preds.append(p.reshape(gh, gw, 3))
        nA = ((preds[a].cpu().numpy() + 1) / 2).clip(0, 1); nB = ((preds[b].cpu().numpy() + 1) / 2).clip(0, 1)
        g = grid.to(DEVICE).view(1, 1, -1, 2)
        pbw = F.grid_sample(preds[b].permute(2, 0, 1)[None], g, align_corners=False)[0, :, 0, :].t()
        paf = preds[a].reshape(-1, 3)
        cos = (F.normalize(paf, dim=1) * F.normalize(pbw, dim=1)).sum(1).clamp(-1, 1)
        ang = torch.rad2deg(torch.arccos(cos)).cpu().numpy()
        fig.add_subplot(gs[row, 0]).imshow(nA); fig.axes[-1].set_title(f"{tag}: pred normal A", fontsize=10); fig.axes[-1].axis("off")
        fig.add_subplot(gs[row, 1]).imshow(nB); fig.axes[-1].set_title(f"{tag}: pred normal B", fontsize=10); fig.axes[-1].axis("off")
        axh = fig.add_subplot(gs[row, 2])
        im = heat_on_tile(axh, imA, ang, v, gh, gw, f"{tag}: A-vs-B normal disagree on overlap (μ={ang[v].mean():.1f}°)",
                          "inferno_r", 0, 60)
        fig.colorbar(im, ax=axh, fraction=0.046, pad=0.04)
    fig.suptitle("Predicted-normal cross-tile consistency (overlap region; LoRA = lower angular disagreement)", y=1.0, fontsize=13)
    fig.tight_layout(); fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig); print("saved", out)


def erp_rgb_for_normal(erp):
    return erp


# ------------------------------------------------------------------ correspondence
def fig_corr(encs, erp, geom, pi, out, name, n_q=14):
    a, b = geom["pairs"][pi]; grid, valid, _ = geom["warps"][pi]
    ya, pa = geom["specs"][a]; yb, pb = geom["specs"][b]
    imA = V.tile_rgb(erp, ya, pa, geom["hfov"]); imB = V.tile_rgb(erp, yb, pb, geom["hfov"])
    g = grid.cpu().numpy(); v = valid.cpu().numpy().astype(bool)
    gh = gw = int(np.sqrt(v.shape[0])); tb = V.true_b_cell(g, gh, gw)
    bcells = np.zeros(gh * gw, bool); bcells[tb[v]] = True
    gap = 50
    fig = plt.figure(figsize=(12, 4.6 * (len(encs) + 1)))
    gs = GridSpec(len(encs) + 1, 1, height_ratios=[0.8] + [1] * len(encs), figure=fig)
    draw_erp(fig.add_subplot(gs[0]), erp, [(ya, pa), (yb, pb)], geom["hfov"])
    rng = np.random.RandomState(0)
    for r, (tag, enc) in enumerate(encs, start=1):
        ax = fig.add_subplot(gs[r])
        fa = V.tile_feat(enc, imA); fb = V.tile_feat(enc, imB)
        d = fa.shape[0]; FA = fa.reshape(d, -1).t(); FB = fb.reshape(d, -1).t()
        shade_overlap(ax, imA, v.astype(float), gh, gw, (0, 0.8, 1), x0=0)
        shade_overlap(ax, imB, bcells.astype(float), gh, gw, (1, 0.8, 0), x0=TILE + gap)
        ax.set_xlim(0, 2 * TILE + gap); ax.set_ylim(TILE, 0); ax.axis("off")
        vcells = np.where(v)[0]; qs = rng.choice(vcells, min(n_q, len(vcells)), replace=False)
        nn = (FA[qs] @ FB.t()).argmax(1).numpy(); patch = TILE // gh; hit = 0
        for q, n in zip(qs, nn):
            ay, ax_ = (q // gw) * patch + patch // 2, (q % gw) * patch + patch // 2
            ny, nx_ = (n // gw) * patch + patch // 2, (n % gw) * patch + patch // 2 + TILE + gap
            ok = (abs(n // gw - tb[q] // gw) <= 1) and (abs(n % gw - tb[q] % gw) <= 1); hit += ok
            c = "#1a1" if ok else "#e11"
            ax.plot([ax_, nx_], [ay, ny], "-", color=c, lw=1.5, alpha=.9); ax.plot(ax_, ay, "o", color=c, ms=4); ax.plot(nx_, ny, "x", color=c, ms=6)
        ax.set_title(f"{tag}: feature-NN matches  (A→B, {hit}/{len(qs)} correct)   "
                     f"cyan=A overlap, yellow=B overlap, green=correct red=wrong", fontsize=10)
    fig.suptitle(f"{name}: cross-tile correspondence by feature similarity", y=1.0, fontsize=13)
    fig.tight_layout(); fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig); print("saved", out)


def main():
    frozen = PanoEncoder(model_id=P.MODEL, lora_rank=0).to(DEVICE).eval()
    lora = PanoEncoder(model_id=P.MODEL, adapter_path=T.CKPT).to(DEVICE).eval()
    encs = [("frozen", frozen), ("LoRA", lora)]

    P.configure("densepass"); D.DATASET = "densepass"; D.HFOV = 50.0; P.enc_patch = frozen.patch
    geom_o = T.build_geometry(frozen, 50.0, (0.0,))
    erp_o = P.load_rgb_label(data.list_densepass()[75])[0]
    for fn, args in [(fig_corr, (encs, erp_o, geom_o, 0, os.path.join(DOCS, "viz_correspondence.png"), "DensePASS")),
                     (fig_cosine, (encs, erp_o, geom_o, 0, os.path.join(DOCS, "viz_cosine_heatmap_outdoor.png"), "DensePASS"))]:
        try:
            fn(*args)
        except Exception as ex:
            print(fn.__name__, "FAIL", type(ex).__name__, str(ex)[:120])

    P.configure("stanford2d3d"); D.DATASET = "stanford2d3d"; D.HFOV = 65.0
    geom_i = T.build_geometry(frozen, 65.0, (-45.0, 0.0, 45.0))

    def area(f): return f.split("extracted_data/")[1].split("/")[0]
    s2d = data.list_erps("stanford2d3d"); val_f = [f for f in s2d if "5" in area(f)]
    erp_i = P.load_rgb_label(val_f[0])[0]; hp = equator_hpair(geom_i)
    try:
        fig_cosine(encs, erp_i, geom_i, hp, os.path.join(DOCS, "viz_cosine_heatmap_indoor.png"), "Stanford2D3D")
    except Exception as ex:
        print("cosine indoor FAIL", type(ex).__name__, str(ex)[:120])

    try:
        tr_f = [f for f in s2d if "5" not in area(f)][:60]; clfs = {}
        for tag, enc in encs:
            P.enc_patch = enc.patch
            packs = [PN.pano_tiles(enc, *PN.load_rgb_normal(f), geom_i)[0] for f in tr_f]
            clfs[tag] = PN.train_probe(packs)
        _, nrm, valn = PN.load_rgb_normal(val_f[0])
        fig_normal([(t, e, clfs[t]) for t, e in encs], erp_i, nrm, valn, geom_i, hp,
                   os.path.join(DOCS, "viz_normal_consistency.png"))
    except Exception as ex:
        print("normal FAIL", type(ex).__name__, str(ex)[:120])


if __name__ == "__main__":
    main()
