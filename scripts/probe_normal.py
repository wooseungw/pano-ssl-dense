"""A: surface-normal probe, frozen vs LoRA-SSL, on Stanford2D3D (indoor, hfov65 3-ring).

Geometric task where cross-tile CONSISTENCY is the objective and no classification head
launders it. Light linear probe (each encoder its own) maps frozen/LoRA patch features ->
unit normal. Normals are frame-fixed within a pano, so two tiles seeing the same surface
point should predict the SAME normal -> consistency = angular disagreement at overlaps
(the equivariance the SSL targets). Reported alongside accuracy (angular error vs GT).

Run: CUDA_VISIBLE_DEVICES=1 conda run -n pano python scripts/probe_normal.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import geometry  # noqa: E402
import data  # noqa: E402
import probe_seg_dinov3 as P  # noqa: E402
import train_ssl as T  # noqa: E402
from encoder import PanoEncoder, normalize_tiles  # noqa: E402

DEVICE = P.DEVICE
CKPT = T.CKPT
TILE, SEED, N_TR = 512, 0, 60000
W, H = 1024, 512


def load_rgb_normal(f):
    rgb = np.array(Image.open(f).convert("RGB").resize((W, H), Image.BILINEAR))
    n = np.array(Image.open(data.s2d3d_gt_path(f, "normal")).convert("RGB").resize((W, H), Image.NEAREST))
    v = n.astype(np.float32) / 255.0 * 2 - 1
    nrm = np.linalg.norm(v, axis=2, keepdims=True)
    valid = (nrm[:, :, 0] > 0.5).astype(np.float32)
    v = v / np.clip(nrm, 1e-6, None)
    return rgb, v, valid


def warp_to_grid(arr, yaw, pitch, hfov, gh, gw, ch):
    """e2p-warp an (H,W,ch) map to a tile then sample patch centers -> (gh,gw,ch)."""
    return geometry.warp_nearest_centers(arr, yaw, pitch, hfov, TILE, gh, gw)


@torch.no_grad()
def pano_tiles(enc, rgb, nrm, val, geom):
    """-> per spec: feat (gh,gw,D), gtn (gh,gw,3) unit, gvalid (gh,gw)."""
    out = []
    for (yaw, pitch) in geom["specs"]:
        tile = np.asarray(P.a2p.erp_to_pinhole_tile(rgb, yaw, pitch, geom["hfov"], TILE))
        x = torch.from_numpy(tile).float().permute(2, 0, 1)[None] / 255.0
        f = P.dense(enc, normalize_tiles(x.to(DEVICE)))[0]                 # (D,gh,gw)
        d, gh, gw = f.shape
        gtn = warp_to_grid(nrm, yaw, pitch, geom["hfov"], gh, gw, 3)
        gtn = gtn / np.clip(np.linalg.norm(gtn, axis=2, keepdims=True), 1e-6, None)
        gv = warp_to_grid(val, yaw, pitch, geom["hfov"], gh, gw, 1)[:, :, 0] > 0.5
        out.append((f.permute(1, 2, 0).cpu(), torch.from_numpy(gtn).float(), torch.from_numpy(gv)))
    return out, (gh, gw)


def train_probe(packs):
    Xs, Ys = [], []
    for tiles in packs:
        for f, gtn, gv in tiles:
            m = gv.reshape(-1)
            Xs.append(f.reshape(-1, f.shape[-1])[m]); Ys.append(gtn.reshape(-1, 3)[m])
    X, Y = torch.cat(Xs), torch.cat(Ys)
    if X.shape[0] > N_TR:
        idx = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(SEED))[:N_TR]
        X, Y = X[idx], Y[idx]
    torch.manual_seed(SEED)
    clf = torch.nn.Linear(X.shape[1], 3).to(DEVICE)
    opt = torch.optim.Adam(clf.parameters(), 1e-3, weight_decay=1e-4)
    X, Y = X.to(DEVICE).float(), Y.to(DEVICE).float()
    for _ in range(800):
        opt.zero_grad()
        pred = F.normalize(clf(X), dim=1)
        (1 - (pred * Y).sum(1)).mean().backward()
        opt.step()
    return clf


def evaluate(clf, packs, geom):
    ang_err, n_err = 0.0, 0
    ang_con, n_con = 0.0, 0
    for tiles in packs:
        preds = []
        for f, gtn, gv in tiles:
            gh, gw, d = f.shape
            with torch.no_grad():
                p = F.normalize(clf(f.reshape(-1, d).to(DEVICE).float()), dim=1)
            preds.append(p.reshape(gh, gw, 3))
            m = gv.reshape(-1).to(DEVICE)
            cos = (p * gtn.reshape(-1, 3).to(DEVICE)).sum(1).clamp(-1, 1)
            ang_err += torch.rad2deg(torch.arccos(cos[m])).sum().item(); n_err += int(m.sum())
        # cross-tile consistency on overlaps
        for (a, b), (grid, valid, weight) in zip(geom["pairs"], geom["warps"]):
            v = valid.to(DEVICE).bool()
            if v.sum() < 4:
                continue
            pa = preds[a].permute(2, 0, 1)[None]                          # (1,3,gh,gw)
            g = grid.to(DEVICE).view(1, 1, -1, 2)
            pb = F.grid_sample(preds[b].permute(2, 0, 1)[None], g, align_corners=False)[0, :, 0, :].t()
            paf = pa[0].permute(1, 2, 0).reshape(-1, 3)
            cos = (F.normalize(paf, dim=1) * F.normalize(pb, dim=1)).sum(1).clamp(-1, 1)
            ang_con += torch.rad2deg(torch.arccos(cos[v])).sum().item(); n_con += int(v.sum())
    return ang_err / max(n_err, 1), ang_con / max(n_con, 1)


def main():
    P.configure("stanford2d3d"); P.TILE = TILE
    frozen = PanoEncoder(model_id=P.MODEL, lora_rank=0).to(DEVICE).eval()
    lora = PanoEncoder(model_id=P.MODEL, adapter_path=CKPT).to(DEVICE).eval()
    geom = T.build_geometry(frozen, 65.0, (-45.0, 0.0, 45.0))
    files = data.list_erps("stanford2d3d")

    def area(f): return f.split("extracted_data/")[1].split("/")[0]
    tr_f = [f for f in files if "5" not in area(f)][:80]
    va_f = [f for f in files if "5" in area(f)][:30]
    print(f"normal probe: tr={len(tr_f)} va={len(va_f)} tiles/pano={len(geom['specs'])}\n"
          f"{'enc':7s} {'angErr(deg)↓':>13} {'xtileConsist(deg)↓':>19}", flush=True)
    for tag, enc in [("frozen", frozen), ("LoRA", lora)]:
        P.enc_patch = enc.patch
        tr = [pano_tiles(enc, *load_rgb_normal(f), geom)[0] for f in tr_f]
        va = [pano_tiles(enc, *load_rgb_normal(f), geom)[0] for f in va_f]
        clf = train_probe(tr)
        acc, con = evaluate(clf, va, geom)
        print(f"{tag:7s} {acc:13.2f} {con:19.2f}", flush=True)


if __name__ == "__main__":
    main()
