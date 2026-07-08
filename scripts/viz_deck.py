"""Generate the 3 slide-deck figures (P1 method / P2 results / P3 eval) as PNGs.

P1 (CPU): E2P-overlap pipeline — ERP -> tiles -> frozen DINOv3+LoRA -> dense feat, plus a
          real overlap warp-correspondence inset and the 3 loss terms.
P2 (CPU): results — Seg-S2D3D mIoU bars, prototype-purity (erosion) bars, consistency-vs-accuracy.
P3      : reuse the already-rendered stitch integration figure.

Run: conda run -n pano python scripts/viz_deck.py
"""
from __future__ import annotations

import os
import shutil
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import anyres_e2p as a2p          # noqa: E402
import data                       # noqa: E402
import geometry as G              # noqa: E402
import runlog                     # noqa: E402

import matplotlib                 # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

# ---- shared style (picked, hue-biased slate neutrals; single blue accent + good/bad semantics) ----
INK, MUTE, LINE = "#22303a", "#5b6b76", "#d5dce1"
BG, CARD = "#ffffff", "#f4f6f8"
ACCENT, GOOD, BAD, WARN, NEU = "#2f6f9f", "#2e8b57", "#c8455f", "#e08a3c", "#9aa7b0"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "figure.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": LINE, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTE, "ytick.color": MUTE, "axes.titlecolor": INK})
ERP_H, ERP_W, TILE, HFOV, PATCH = 1024, 2048, 512, 65.0, 16


def load_area5_erp():
    files = [f for f in data.list_erps("stanford2d3d") if "area_5" in f]
    f = files[0]
    return np.array(Image.open(f).convert("RGB").resize((ERP_W, ERP_H), Image.BILINEAR))


def box(ax, x, y, w, h, text, fc=CARD, ec=LINE, tc=INK, fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                linewidth=1.3, edgecolor=ec, facecolor=fc, mutation_aspect=0.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=15,
                                 linewidth=1.6, color=MUTE, shrinkA=2, shrinkB=2))


# ============================== P1 — method pipeline ==============================
def make_p1(erp, out):
    yaws = a2p.make_yaw_centers_closed_loop(HFOV, 0.25, start_deg=-180.0)
    # two adjacent equator tiles + coordmaps for a real warp inset
    ya, yb = yaws[3], yaws[4]
    tA = np.asarray(a2p.erp_to_pinhole_tile(erp, ya, 0.0, HFOV, TILE))
    tB = np.asarray(a2p.erp_to_pinhole_tile(erp, yb, 0.0, HFOV, TILE))
    cA = G.render_coordmap(ERP_H, ERP_W, ya, 0.0, HFOV, TILE)
    cB = G.render_coordmap(ERP_H, ERP_W, yb, 0.0, HFOV, TILE)
    wf = G.warp_field_from_coordmaps(cA, cB, PATCH, HFOV, erp_w=ERP_W, dst_stride=3)
    gn = wf.grid_hw[0]
    gi, gj = np.mgrid[0:gn, 0:gn]
    ax_px = (gj * PATCH + PATCH // 2).ravel().astype(float)      # A-cell centers in tile px
    ay_px = (gi * PATCH + PATCH // 2).ravel().astype(float)
    bx_px = (wf.grid[:, 0] + 1) / 2 * TILE
    by_px = (wf.grid[:, 1] + 1) / 2 * TILE
    val = wf.valid
    idx = np.where(val)[0]
    idx = idx[np.linspace(0, len(idx) - 1, 26).astype(int)] if len(idx) else idx

    fig = plt.figure(figsize=(12.8, 7.2))
    bg = fig.add_axes((0, 0, 1, 1)); bg.axis("off"); bg.set_xlim(0, 1); bg.set_ylim(0, 1)
    bg.text(0.035, 0.95, "P1 · Method — E2P-Overlap SSL pipeline", fontsize=17,
            fontweight="bold", color=INK)
    bg.text(0.035, 0.905, "Frozen DINOv3 (ViT-B/16) + LoRA 0.59M · label-free · overlap warp = free geometric self-supervision",
            fontsize=10.5, color=MUTE)

    # top flow: ERP -> tiles -> encoder -> feat
    ax_erp = fig.add_axes((0.035, 0.55, 0.24, 0.28)); ax_erp.imshow(erp); ax_erp.axis("off")
    ax_erp.set_title("input ERP  1024×2048", fontsize=9.5)
    mont = np.concatenate([np.asarray(a2p.erp_to_pinhole_tile(erp, yaws[k], 0.0, HFOV, TILE))
                           for k in (2, 3, 4)], axis=1)
    ax_t = fig.add_axes((0.33, 0.58, 0.20, 0.22)); ax_t.imshow(mont); ax_t.axis("off")
    ax_t.set_title("E2P tiles  (24 × 512²)", fontsize=9.5)
    box(bg, 0.575, 0.60, 0.17, 0.14, "DINOv3 ViT-B/16\n(frozen)\n+ LoRA 0.59M", fc="#eaf1f7",
        ec=ACCENT, fs=10, bold=True)
    box(bg, 0.785, 0.60, 0.18, 0.14, "dense features\n(24, 768, 32, 32)", fc=CARD, fs=10, bold=True)
    arrow(bg, 0.28, 0.69, 0.325, 0.69); arrow(bg, 0.535, 0.69, 0.573, 0.69)
    arrow(bg, 0.747, 0.67, 0.783, 0.67)
    bg.text(0.302, 0.72, "tile", fontsize=8, color=MUTE, ha="center")

    # bottom-left: warp correspondence inset (two adjacent tiles + matched lines)
    gap = 60
    pair = np.full((TILE, TILE * 2 + gap, 3), 255, np.uint8)
    pair[:, :TILE] = tA; pair[:, TILE + gap:] = tB
    ax_w = fig.add_axes((0.035, 0.07, 0.44, 0.40)); ax_w.imshow(pair); ax_w.axis("off")
    for i in idx:
        ax_w.plot([ax_px[i], bx_px[i] + TILE + gap], [ay_px[i], by_px[i]],
                  color=ACCENT, lw=0.6, alpha=0.75)
        ax_w.scatter([ax_px[i]], [ay_px[i]], s=6, color=BAD, zorder=3)
        ax_w.scatter([bx_px[i] + TILE + gap], [by_px[i]], s=6, color=GOOD, zorder=3)
    ax_w.set_title("overlap warp correspondence  (geometry-only, depth-free, exact)", fontsize=9.5)
    ax_w.text(0.25, -0.04, "tile A", transform=ax_w.transAxes, ha="center", fontsize=9, color=MUTE)
    ax_w.text(0.75, -0.04, "tile B", transform=ax_w.transAxes, ha="center", fontsize=9, color=MUTE)

    # bottom-right: 3 loss terms
    bg.text(0.52, 0.45, "Loss  L  =  w₁·warp  +  w₂·distill  +  w₃·VICReg", fontsize=12,
            fontweight="bold", color=INK)
    box(bg, 0.52, 0.32, 0.45, 0.10, "①  warp-equivariance   Fₐ(p) ≈ F_b(Hp)\n     cosine on overlap · obliquity-weighted  → geometry",
        fc="#eaf1f7", ec=ACCENT, fs=9.5)
    box(bg, 0.52, 0.20, 0.45, 0.10, "②  distill   token cosine + relational Gram → frozen teacher\n     preserves planar semantics & inter-region structure",
        fc=CARD, fs=9.5)
    box(bg, 0.52, 0.08, 0.45, 0.10, "③  VICReg   variance + covariance  (γ=0.04)\n     anti-collapse floor · AdamW 1e-4 · warp warm-up",
        fc=CARD, fs=9.5)
    fig.savefig(out, dpi=125); plt.close(fig)


# ============================== P2 — results ==============================
def make_p2(out):
    seg = [("frozen", 57.7, "base"), ("geo", 56.6, "ad"), ("TC3", 56.3, "champ"),
           ("scaled", 55.6, "ad"), ("iBOT", 50.9, "bad")]
    pur = [("TC3", .862, "champ"), ("geo", .854, "ok"), ("frozen", .838, "base"),
           ("F3-EMA", .830, "er"), ("F3-stu", .821, "er"), ("VICReg-d", .753, "er"),
           ("M1", .730, "er"), ("VICReg-n", .728, "er")]
    cmp_labels = ["overlap cos\n(out)", "retrieval@1\n(out)", "CKA\n(indoor)", "single mIoU\n(indoor)"]
    froz = [0.68, 0.21, 0.52, 0.577]; lora = [0.91, 0.86, 0.83, 0.576]

    fig = plt.figure(figsize=(13.2, 5.6))
    fig.text(0.035, 0.93, "P2 · Results — variants & the erosion story", fontsize=17,
             fontweight="bold", color=INK)
    fig.text(0.035, 0.875, "Accuracy: no adapter beats frozen.  Semantic identity: TC3 is the only erosion-free winner.  "
             "Base adapter: consistency ↑↑, accuracy flat.", fontsize=10, color=MUTE)
    gs = fig.add_gridspec(1, 3, left=0.05, right=0.975, top=0.79, bottom=0.13, wspace=0.42)

    # (a) Seg mIoU
    ax = fig.add_subplot(gs[0, 0])
    cmap = {"base": INK, "ad": NEU, "champ": ACCENT, "bad": BAD}
    names = [s[0] for s in seg]; vals = [s[1] for s in seg]
    ax.bar(names, vals, color=[cmap[s[2]] for s in seg], width=0.66, zorder=3)
    ax.axhline(57.7, ls="--", lw=1, color=INK, alpha=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.15, f"{v:.1f}", ha="center", fontsize=9, color=INK, fontweight="bold")
    ax.set_ylim(48, 59); ax.set_ylabel("Seg-S2D3D fold1 mIoU"); ax.set_title("Downstream accuracy", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color=LINE, lw=0.6)
    ax.tick_params(axis="x", labelsize=8.5)

    # (b) purity / erosion
    ax = fig.add_subplot(gs[0, 1])
    base = 0.838
    labs = [p[0] for p in pur]; pv = [p[1] for p in pur]
    cols = [ACCENT if p[2] == "champ" else INK if p[2] == "base" else
            (GOOD if p[1] >= base else BAD) for p in pur]
    ax.barh(range(len(pur)), pv, color=cols, height=0.68, zorder=3)
    ax.axvline(base, ls="--", lw=1, color=INK, alpha=0.6)
    ax.text(base, len(pur) - 0.3, " frozen 0.838", fontsize=8, color=INK, va="center")
    ax.set_yticks(range(len(pur))); ax.set_yticklabels(labs, fontsize=8.5); ax.invert_yaxis()
    ax.set_xlim(0.70, 0.875); ax.set_xlabel("prototype purity  (↓ = erosion)")
    ax.set_title("Semantic identity (erosion)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", color=LINE, lw=0.6)
    for i, v in enumerate(pv):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=7.5, color=MUTE)

    # (c) consistency vs accuracy (base frozen -> LoRA)
    ax = fig.add_subplot(gs[0, 2])
    x = np.arange(len(cmp_labels)); w = 0.38
    ax.bar(x - w / 2, froz, w, label="frozen", color=NEU, zorder=3)
    ax.bar(x + w / 2, lora, w, label="LoRA (SSL)", color=ACCENT, zorder=3)
    for i in range(len(x)):
        dv = lora[i] - froz[i]
        ax.text(x[i], max(froz[i], lora[i]) + 0.03, f"{dv:+.2f}", ha="center", fontsize=8,
                color=GOOD if dv > 0.03 else MUTE, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(cmp_labels, fontsize=8); ax.set_ylim(0, 1.05)
    ax.set_title("Base: consistency ↑, accuracy flat", fontsize=11); ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color=LINE, lw=0.6)
    fig.savefig(out, dpi=125); plt.close(fig)


def main():
    run = runlog.create_run("deck", {"purpose": "3-slide deck figures (method/results/eval)"})
    viz = os.path.join(run, "viz")
    erp = load_area5_erp()
    make_p1(erp, os.path.join(viz, "P1_method_pipeline.png"))
    make_p2(os.path.join(viz, "P2_results.png"))
    src = os.path.join(os.path.dirname(viz), "..", "0705_1202_stitch_demo", "viz",
                       "s2d3d_stitch_integration.png")
    if os.path.exists(src):
        shutil.copy(src, os.path.join(viz, "P3_eval_stitch.png"))
    print(f"saved deck -> {viz}", flush=True)
    for p in sorted(os.listdir(viz)):
        print("  ", p, flush=True)


if __name__ == "__main__":
    main()
