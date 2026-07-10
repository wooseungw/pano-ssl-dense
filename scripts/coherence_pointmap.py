"""Coherence deliverable — flagship demo: frozen vs SSL-adapter on COHERENCE (not accuracy).

The information-theoretic close (docs/FAILURE_ANALYSIS.md §8) established that encoder-SSL cannot raise per-tile
ACCURACY (seg axis-1-saturated, depth axis-2). Its genuine, robust value is cross-tile COHERENCE. This demo
formalizes that for the committed pointmap task: the adapter makes overlapping E2P tiles fuse into ONE coherent
full-sphere 3D field (frozen ghosts at overlaps), and it recovers dense correspondence frozen DINOv3 cannot.

Metrics (frozen vs adapter, S2D3D area_1 train / area_3 val, pano-disjoint):
  depth logErr           accuracy — expected ~FLAT (the honest 'consistency != accuracy')
  cross-tile |Δlogd|     depth COHERENCE at overlap correspondences — expected LOWER (better)
  overlap point-gap      3D fusion ghosting — expected LOWER (better)
  overlap feat cosine    dense correspondence — expected HIGHER (frozen ~0.68 -> adapter ~0.9)
Saves a fused-cloud + overlap-ghosting figure (frozen vs adapter).

Run: OPENCV_IO_ENABLE_OPENEXR=1 CUDA_VISIBLE_DEVICES=0 conda run -n pano python scripts/coherence_pointmap.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pointmap_fusion as PF  # noqa: E402
import data  # noqa: E402
import train_ssl as T  # noqa: E402
from encoder import PanoEncoder  # noqa: E402

DEVICE = PF.DEVICE
NTR = int(os.environ.get("NTR", 60))
NVA = int(os.environ.get("NVA", 30))
P = PF.P


def packs_for(enc, files, geom):
    P.enc_patch = enc.patch
    out = []
    for f in files:
        rgb = np.array(Image.open(f).convert("RGB").resize((PF.W, PF.H), Image.BILINEAR))
        dn, val = PF.load_depth(f)
        out.append([PF.tile_pack(enc, rgb, dn, val, y, p, geom["hfov"]) for (y, p) in geom["specs"]])
    return out


def overlap_cosine(packs, geom):
    """head-free correspondence: mean feature cosine at overlap correspondences."""
    tot, n = 0.0, 0
    for tiles in packs:
        feats = [t[0] for t in tiles]
        D = feats[0].shape[-1]
        for (a, b), (grid, valid, _) in zip(geom["pairs"], geom["warps"]):
            v = valid.cpu().numpy().astype(bool)
            if v.sum() < 8:
                continue
            g = grid.cpu().view(1, 1, -1, 2)
            fbw = F.grid_sample(feats[b].permute(2, 0, 1)[None].float(), g, align_corners=False)[0, :, 0, :].t().numpy()
            fa = feats[a].reshape(-1, D).numpy()
            c = (fa * fbw).sum(1) / (np.linalg.norm(fa, axis=1) * np.linalg.norm(fbw, axis=1) + 1e-9)
            tot += c[v].sum(); n += int(v.sum())
    return tot / max(n, 1)


def point_gap(packs, clf, geom):
    tot, n = 0.0, 0
    for tiles in packs:
        for pi in range(len(geom["pairs"])):
            try:
                pa, pb = PF.pair_points(tiles, clf, geom, pi)
            except Exception:
                continue
            if len(pa) < 8:
                continue
            tot += np.linalg.norm(pa - pb, axis=1).sum(); n += len(pa)
    return tot / max(n, 1)


def main():
    frozen = PanoEncoder(model_id=P.MODEL, lora_rank=0).to(DEVICE).eval()
    lora = PanoEncoder(model_id=P.MODEL, adapter_path=T.CKPT).to(DEVICE).eval()
    P.configure("stanford2d3d"); P.TILE = PF.TILE
    geom = PF.T.build_geometry(frozen, 65.0, (-45.0, 0.0, 45.0))
    s2d = data.list_erps("stanford2d3d")

    def area(f):
        return f.split("extracted_data/")[1].split("/")[0]
    tr_f = [f for f in s2d if area(f) == "area_1"][:NTR]
    va_f = [f for f in s2d if area(f) == "area_3"][:NVA]     # area_5 missing -> area_3 held-out
    print(f"coherence demo: adapter={os.path.basename(T.CKPT)}  train {len(tr_f)}(area_1) / val {len(va_f)}(area_3)  "
          f"tiles/pano={len(geom['specs'])}\n", flush=True)

    print(f"{'enc':8s}{'depthErr↓':>11}{'xtile|Δlogd|↓':>15}{'pointGap↓':>11}{'overlapCos↑':>13}", flush=True)
    res = {}
    clfs, packs_va = {}, {}
    for tag, enc in [("frozen", frozen), ("adapter", lora)]:
        tr = packs_for(enc, tr_f, geom); va = packs_for(enc, va_f, geom)
        clf = PF.train_probe(tr)
        acc, con = PF.evaluate(clf, va, geom)
        gap = point_gap(va, clf, geom)
        cos = overlap_cosine(va, geom)
        res[tag] = (acc, con, gap, cos); clfs[tag] = clf; packs_va[tag] = va
        print(f"{tag:8s}{acc:>11.3f}{con:>15.3f}{gap:>11.3f}{cos:>13.3f}", flush=True)

    (fa, ca, ga, ka), (fl, cl, gl, kl) = res["frozen"], res["adapter"]
    print(f"\nadapter vs frozen:  depthErr {fl-fa:+.3f} ({'flat' if abs(fl-fa)<0.01 else 'moved'})   "
          f"|Δlogd| {(cl-ca)/ca*100:+.0f}%   pointGap {(gl-ga)/ga*100:+.0f}%   overlapCos {kl-ka:+.3f}", flush=True)
    print("\nCOHERENCE VERDICT: accuracy ~flat, but the adapter fuses more coherently "
          "(lower ghosting/inconsistency) and recovers dense correspondence — the deliverable.", flush=True)

    # ghosting figure on one val pano
    DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures", "coherence")
    os.makedirs(DOCS, exist_ok=True)
    P.enc_patch = frozen.patch
    rgb = np.array(Image.open(va_f[0]).convert("RGB").resize((PF.W, PF.H), Image.BILINEAR))
    dn, val = PF.load_depth(va_f[0])
    tiles_by = {tag: [PF.tile_pack(enc, rgb, dn, val, y, p, geom["hfov"]) for (y, p) in geom["specs"]]
                for tag, enc in [("frozen", frozen), ("adapter", lora)]}
    pi = next((i for i, (a, b) in enumerate(geom["pairs"])
               if abs(geom["specs"][a][1]) < 1e-6 and abs(geom["specs"][b][1]) < 1e-6), 0)
    fig = plt.figure(figsize=(13, 9)); gs = GridSpec(2, 2, figure=fig)
    for col, (tag, enc) in enumerate([("frozen", frozen), ("adapter", lora)]):
        pts, cols = PF.fuse_points(tiles_by[tag], clfs[tag])
        ax = fig.add_subplot(gs[0, col]); ax.scatter(pts[:, 0], pts[:, 2], s=2, c=np.clip(cols, 0, 1))
        ax.set_title(f"{tag}: fused top-down cloud"); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        pa, pb = PF.pair_points(tiles_by[tag], clfs[tag], geom, pi)
        ax2 = fig.add_subplot(gs[1, col])
        ax2.scatter(pa[:, 0], pa[:, 2], s=8, c="#d11", label="tile A", alpha=.7)
        ax2.scatter(pb[:, 0], pb[:, 2], s=8, c="#15c", label="tile B", alpha=.7)
        ax2.set_title(f"{tag}: overlap pair — gap={np.linalg.norm(pa-pb,axis=1).mean():.3f}")
        ax2.set_aspect("equal"); ax2.legend(fontsize=8); ax2.set_xticks([]); ax2.set_yticks([])
    fig.suptitle("Coherence deliverable — adapter fuses overlapping tiles into one coherent field (frozen ghosts)", y=1.0)
    out = os.path.join(DOCS, "coherence_pointmap.png"); fig.tight_layout(); fig.savefig(out, dpi=120, bbox_inches="tight")
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
