"""Cross-tile pointmap FUSION demo, frozen vs LoRA-SSL (Stanford2D3D, indoor hfov65 3-ring).

E2P tiles share one optical center (no parallax) => the same surface point is seen along
the SAME ray by both tiles, so fusion ghosting reduces to DEPTH disagreement along that ray.
We predict per-pano-normalized log-depth with a light probe (each encoder its own), back-
project each patch to a 3D point (depth x shared ray dir), fuse all tiles, and ask whether
the LoRA adapter makes the fused cloud COHERENT (overlap points coincide) vs frozen ghosting.

Metrics (held-out): depth accuracy (|log| err) and CROSS-TILE consistency (|log dA - log dB|
at overlap correspondences). Figure: fused top-down cloud + overlap-pair ghosting, frozen vs LoRA.

Run: OPENCV_IO_ENABLE_OPENEXR=1 CUDA_VISIBLE_DEVICES=1 conda run -n pano python scripts/pointmap_fusion.py
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
import data  # noqa: E402
import geometry as G  # noqa: E402
import probe_seg_dinov3 as P  # noqa: E402
import train_ssl as T  # noqa: E402
import probe_normal as PN  # noqa: E402
import viz_consistency as V  # noqa: E402
from encoder import PanoEncoder, normalize_tiles  # noqa: E402

DEVICE = P.DEVICE
TILE, SEED = 512, 0
ERP_H, ERP_W = 1024, 2048
W, H = 1024, 512


def load_depth(f):
    d = np.array(Image.open(data.s2d3d_gt_path(f, "depth")).resize((W, H), Image.NEAREST)).astype(np.float32)
    valid = (d > 0) & (d < 65535)
    med = np.median(d[valid]) if valid.any() else 1.0
    return (d / med), valid.astype(np.float32)            # normalized depth (~1 at median), mask


def ray_dirs(yaw, pitch, hfov, gh, gw):
    cm = G.render_coordmap(ERP_H, ERP_W, yaw, pitch, hfov, TILE)        # (TILE,TILE,2) erp (x,y)
    cy = ((np.arange(gh) + 0.5) * TILE / gh).astype(int)
    cx = ((np.arange(gw) + 0.5) * TILE / gw).astype(int)
    xy = cm[np.ix_(cy, cx)]                                             # (gh,gw,2)
    lon = xy[..., 0] / ERP_W * 2 * np.pi
    lat = (0.5 - xy[..., 1] / ERP_H) * np.pi
    return np.stack([np.cos(lat) * np.cos(lon), np.sin(lat), np.cos(lat) * np.sin(lon)], -1)  # (gh,gw,3)


@torch.no_grad()
def tile_pack(enc, rgb, dn, val, yaw, pitch, hfov):
    tile = np.asarray(P.a2p.erp_to_pinhole_tile(rgb, yaw, pitch, hfov, TILE))
    x = torch.from_numpy(tile).float().permute(2, 0, 1)[None] / 255.0
    f = P.dense(enc, normalize_tiles(x.to(DEVICE)))[0]
    d, gh, gw = f.shape
    gd = PN.warp_to_grid(dn[:, :, None], yaw, pitch, hfov, gh, gw, 1)[:, :, 0]
    gv = PN.warp_to_grid(val[:, :, None], yaw, pitch, hfov, gh, gw, 1)[:, :, 0] > 0.5
    cy = ((np.arange(gh) + 0.5) * TILE / gh).astype(int)
    cx = ((np.arange(gw) + 0.5) * TILE / gw).astype(int)
    rgbp = tile[np.ix_(cy, cx)].astype(np.float32) / 255.0             # per-patch color (subsample, no re-warp)
    return f.permute(1, 2, 0).cpu(), gd, gv, ray_dirs(yaw, pitch, hfov, gh, gw), rgbp


def train_probe(packs):
    Xs, Ys = [], []
    for tiles in packs:
        for f, gd, gv, _, _ in tiles:
            m = gv.reshape(-1) & (gd.reshape(-1) > 1e-3)
            Xs.append(f.reshape(-1, f.shape[-1])[m]); Ys.append(torch.from_numpy(np.log(gd.reshape(-1)[m])))
    X, Y = torch.cat(Xs).float(), torch.cat(Ys).float()
    if X.shape[0] > 80000:
        idx = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(SEED))[:80000]
        X, Y = X[idx], Y[idx]
    torch.manual_seed(SEED)
    clf = torch.nn.Linear(X.shape[1], 1).to(DEVICE)
    opt = torch.optim.Adam(clf.parameters(), 1e-3, weight_decay=1e-4)
    X, Y = X.to(DEVICE), Y.to(DEVICE)
    for _ in range(800):
        opt.zero_grad(); F.l1_loss(clf(X)[:, 0], Y).backward(); opt.step()
    return clf


@torch.no_grad()
def pred_logd(clf, f):
    gh, gw, d = f.shape
    return clf(f.reshape(-1, d).to(DEVICE).float())[:, 0].reshape(gh, gw).cpu().numpy()


def evaluate(clf, packs, geom):
    aerr, n = 0.0, 0
    cdis, nc = 0.0, 0
    for tiles in packs:
        logd = [pred_logd(clf, t[0]) for t in tiles]
        for i, (f, gd, gv, _, _) in enumerate(tiles):
            m = gv & (gd > 1e-3)
            aerr += np.abs(logd[i][m] - np.log(gd[m])).sum(); n += int(m.sum())
        for (a, b), (grid, valid, weight) in zip(geom["pairs"], geom["warps"]):
            v = valid.cpu().numpy().astype(bool)
            if v.sum() < 8:
                continue
            la = logd[a].reshape(-1)
            lb = torch.from_numpy(logd[b])[None, None].float()
            g = grid.cpu().view(1, 1, -1, 2)
            lbw = F.grid_sample(lb, g, align_corners=False)[0, 0, 0].numpy()
            cdis += np.abs(la[v] - lbw[v]).sum(); nc += int(v.sum())
    return aerr / max(n, 1), cdis / max(nc, 1)


def fuse_points(tiles, clf):
    pts, cols = [], []
    for f, gd, gv, dirs, rgbp in tiles:
        ld = pred_logd(clf, f); dn = np.exp(ld)
        p = dn[:, :, None] * dirs                                       # (gh,gw,3)
        m = gv.reshape(-1)
        pts.append(p.reshape(-1, 3)[m]); cols.append(rgbp.reshape(-1, 3)[m])
    return np.concatenate(pts), np.concatenate(cols)


def pair_points(tiles, clf, geom, pi):
    a, b = geom["pairs"][pi]; grid, valid, _ = geom["warps"][pi]
    v = valid.cpu().numpy().astype(bool)
    fa, _, _, da, _ = tiles[a]; fb, _, _, db, _ = tiles[b]
    pa = (np.exp(pred_logd(clf, fa))[:, :, None] * da).reshape(-1, 3)
    pbmap = np.exp(pred_logd(clf, fb))[:, :, None] * db                 # (gh,gw,3)
    pbt = torch.from_numpy(pbmap).permute(2, 0, 1)[None].float()
    g = grid.cpu().view(1, 1, -1, 2)
    pb = F.grid_sample(pbt, g, align_corners=False)[0, :, 0, :].t().numpy()   # B at A's correspondences
    return pa[v], pb[v]


def main():
    frozen = PanoEncoder(model_id=P.MODEL, lora_rank=0).to(DEVICE).eval()
    lora = PanoEncoder(model_id=P.MODEL, adapter_path=T.CKPT).to(DEVICE).eval()
    encs = [("frozen", frozen), ("LoRA", lora)]
    P.configure("stanford2d3d"); P.TILE = TILE
    geom = T.build_geometry(frozen, 65.0, (-45.0, 0.0, 45.0))
    s2d = data.list_erps("stanford2d3d")

    def area(f): return f.split("extracted_data/")[1].split("/")[0]
    tr_f = [f for f in s2d if "5" not in area(f)][:60]
    va_f = [f for f in s2d if "5" in area(f)][:25]

    def packs_for(enc, files):
        P.enc_patch = enc.patch
        out = []
        for f in files:
            rgb = np.array(Image.open(f).convert("RGB").resize((W, H), Image.BILINEAR))
            dn, val = load_depth(f)
            out.append([tile_pack(enc, rgb, dn, val, y, p, geom["hfov"]) for (y, p) in geom["specs"]])
        return out

    print(f"pointmap fusion: tr={len(tr_f)} va={len(va_f)} tiles/pano={len(geom['specs'])}\n"
          f"{'enc':7s} {'logDepthErr↓':>13} {'xtileConsist(|Δlogd|)↓':>23}", flush=True)
    clfs, results = {}, {}
    for tag, enc in encs:
        tr = packs_for(enc, tr_f); va = packs_for(enc, va_f)
        clf = train_probe(tr); clfs[tag] = clf
        acc, con = evaluate(clf, va, geom)
        results[tag] = (acc, con)
        print(f"{tag:7s} {acc:13.3f} {con:23.3f}", flush=True)

    # fusion figure on one val pano
    DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures", "pointmap_fusion")
    P.enc_patch = frozen.patch
    rgb = np.array(Image.open(va_f[0]).convert("RGB").resize((W, H), Image.BILINEAR))
    dn, val = load_depth(va_f[0])
    tiles_by = {tag: [tile_pack(enc, rgb, dn, val, y, p, geom["hfov"]) for (y, p) in geom["specs"]]
                for tag, enc in encs}
    pi = next((i for i, (a, b) in enumerate(geom["pairs"])
               if abs(geom["specs"][a][1]) < 1e-6 and abs(geom["specs"][b][1]) < 1e-6), 0)

    fig = plt.figure(figsize=(13, 9))
    gs = GridSpec(2, 2, figure=fig)
    for col, (tag, enc) in enumerate(encs):
        pts, cols = fuse_points(tiles_by[tag], clfs[tag])
        ax = fig.add_subplot(gs[0, col])
        ax.scatter(pts[:, 0], pts[:, 2], s=2, c=np.clip(cols, 0, 1))
        ax.set_title(f"{tag}: fused top-down (X–Z) cloud"); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        pa, pb = pair_points(tiles_by[tag], clfs[tag], geom, pi)
        ax2 = fig.add_subplot(gs[1, col])
        ax2.scatter(pa[:, 0], pa[:, 2], s=8, c="#d11", label="tile A points", alpha=.7)
        ax2.scatter(pb[:, 0], pb[:, 2], s=8, c="#15c", label="tile B points", alpha=.7)
        gap = np.linalg.norm(pa - pb, axis=1).mean()
        ax2.set_title(f"{tag}: overlap pair (X–Z) — A vs B (mean gap={gap:.3f})")
        ax2.set_aspect("equal"); ax2.legend(fontsize=8); ax2.set_xticks([]); ax2.set_yticks([])
    fig.suptitle("Cross-tile pointmap FUSION — frozen ghosts at overlaps, LoRA fuses coherently", fontsize=13, y=1.0)
    out = os.path.join(DOCS, "pointmap_fusion.png")
    fig.tight_layout(); fig.savefig(out, dpi=120, bbox_inches="tight"); print("saved", out, flush=True)


if __name__ == "__main__":
    main()
