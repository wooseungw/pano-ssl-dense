"""Compare ERP decomposition methods the way they are MEANT to be used (Tangent Images,
Eder CVPR2020 protocol): decompose the panorama -> predict per view with a frozen planar
encoder + linear head -> RESAMPLE predictions back to the sphere -> ERP (sphere) mIoU.
Not pooled-patch; the metric is on the whole sphere, predictions gathered least-oblique.

Methods (all cover the full sphere, frozen DINOv3):
  erp_direct    - no decomposition, predict on the raw ERP
  cube6         - cubemap, 6 faces @90°
  cube_rot      - cubemap rotated 45° yaw
  e2p_full65    - OURS: AnyRes-E2P full-sphere, hfov65 overlap0.25
  tangent_ico20 - Tangent Images: icosahedron level-0, 20 tangent planes @73°
  tangent_ico80 - Tangent Images: icosahedron level-1 (subdiv), 80 tangent planes @40°

Run: CUDA_VISIBLE_DEVICES=1 conda run -n pano python scripts/tiling_compare.py [stanford2d3d|densepass]
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import probe_seg_dinov3 as P  # noqa: E402
from encoder import PanoEncoder, normalize_tiles  # noqa: E402

DEVICE = P.DEVICE
DATASET = sys.argv[1] if len(sys.argv) > 1 else "stanford2d3d"
SEED, TILE = 0, 512
IH, IW = 512, 1024                                          # ERP load / index resolution
HS, WS = 128, 256                                          # sphere-eval grid
N_TR = 40000
SPLIT = os.environ.get("SPLIT", "grouped")                 # "area5" -> train area1,2,3,6 / val area5a,5b
METHODS = [m for m in os.environ.get("METHODS", "").split(",") if m]   # subset of method names, empty=all


def ico_centers(level, hfov):
    """Tangent Images (Eder CVPR2020): tangent planes at subdivided-icosahedron face centers."""
    import itertools
    phi = (1 + 5 ** 0.5) / 2
    raw = [(0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
           (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
           (phi, 0, 1), (-phi, 0, 1), (phi, 0, -1), (-phi, 0, -1)]
    V = [np.array(v, float) / np.linalg.norm(v) for v in raw]
    el = min(np.linalg.norm(V[i] - V[j]) for i, j in itertools.combinations(range(12), 2))
    faces = [t for t in itertools.combinations(range(12), 3)
             if all(abs(np.linalg.norm(V[a] - V[b]) - el) < 1e-3 for a, b in itertools.combinations(t, 2))]
    def nrm(v): return v / np.linalg.norm(v)
    cs = []
    for (a, b, c) in faces:
        if level == 0:
            cs.append(nrm(V[a] + V[b] + V[c]))
        else:
            mab, mbc, mca = nrm(V[a] + V[b]), nrm(V[b] + V[c]), nrm(V[c] + V[a])
            for tri in [(V[a], mab, mca), (V[b], mbc, mab), (V[c], mca, mbc), (mab, mbc, mca)]:
                cs.append(nrm(sum(tri)))
    return [(math.degrees(math.atan2(v[2], v[0])), math.degrees(math.asin(np.clip(v[1], -1, 1))), hfov) for v in cs]


def healpix_centers(nside, hfov):
    """tangent tiles at HEALPix equal-area face centers (nside=1 ->12, 2 ->48 uniform-area views)."""
    import healpy as hp
    ipix = np.arange(hp.nside2npix(nside))
    theta, phi = hp.pix2ang(nside, ipix, nest=True)         # theta colat from +z, phi azimuth
    return [(float(np.degrees(p)), float(90.0 - np.degrees(t)), hfov) for t, p in zip(theta, phi)]


def methods():
    full = P.a2p.plan_tiles("full_sphere", 65.0, 65.0, 0.25)
    return {
        "erp_direct": None,
        "cube6": [(0, 0, 90), (90, 0, 90), (180, 0, 90), (270, 0, 90), (0, 89, 90), (0, -89, 90)],
        "cube_ovl110": [(0, 0, 110), (90, 0, 110), (180, 0, 110), (270, 0, 110), (0, 89, 110), (0, -89, 110)],
        "cube_ovl120": [(0, 0, 120), (90, 0, 120), (180, 0, 120), (270, 0, 120), (0, 89, 120), (0, -89, 120)],
        "cube_rot": [(45, 0, 90), (135, 0, 90), (225, 0, 90), (315, 0, 90), (0, 89, 90), (0, -89, 90)],
        "e2p_full65": [(tp.yaw_deg, tp.pitch_deg, 65.0) for tp in full],
        "hp12_h65": healpix_centers(1, 65.0),
        "hp12_h85": healpix_centers(1, 85.0),
        "hp48_h40": healpix_centers(2, 40.0),
        "tangent_ico20": ico_centers(0, 73.0),
        "tangent_ico80": ico_centers(1, 40.0),
    }


SR = 128                                                  # per-tile scatter resolution (dense -> full coverage)


def tile_cells(yaw, pitch, hfov):
    """per tile PIXEL (SRxSR) -> (sphere cell id at HSxWS, obliquity)."""
    uy = np.broadcast_to(np.arange(IW, dtype=np.float32)[None], (IH, IW))
    vy = np.broadcast_to(np.arange(IH, dtype=np.float32)[:, None], (IH, IW))
    um = P.e2p(uy[:, :, None], hfov, yaw, pitch, out_hw=(SR, SR), mode="nearest")[:, :, 0]
    vm = P.e2p(vy[:, :, None], hfov, yaw, pitch, out_hw=(SR, SR), mode="nearest")[:, :, 0]
    uf = np.clip((um / IW * WS).astype(int), 0, WS - 1)
    vf = np.clip((vm / IH * HS).astype(int), 0, HS - 1)
    ii, jj = np.meshgrid(np.arange(SR), np.arange(SR), indexing="ij")
    r = np.sqrt((ii - (SR - 1) / 2) ** 2 + (jj - (SR - 1) / 2) ** 2)
    return (vf * WS + uf).reshape(-1), r.reshape(-1)


@torch.no_grad()
def tile_feat(enc, rgb, yaw, pitch, hfov):
    tile = np.asarray(P.a2p.erp_to_pinhole_tile(rgb, yaw, pitch, hfov, TILE))
    x = torch.from_numpy(tile).float().permute(2, 0, 1)[None] / 255.0
    f = P.dense(enc, normalize_tiles(x.to(DEVICE)))[0]
    return f                                                # (D,gh,gw)


def train_head(enc, cache_tr, plan):
    Xs, Ys = [], []
    per = max(2000, N_TR * 3 // max(1, len(cache_tr)))      # subsample per pano to bound RAM at full split
    g = torch.Generator().manual_seed(SEED)
    for rgb, lab in cache_tr:
        xs, ys = [], []
        if plan is None:
            f, l = P.feats_erp(enc, rgb, lab); xs.append(f); ys.append(l)
        else:
            for (yaw, pitch, hfov) in plan:
                f = tile_feat(enc, rgb, yaw, pitch, hfov); d, gh, gw = f.shape
                gl = P.label_to_grid(P.e2p_label(lab, yaw, pitch, hfov, TILE), gh, gw)
                xs.append(f.reshape(d, -1).t().cpu()); ys.append(torch.from_numpy(gl.reshape(-1)))
        xp = torch.cat(xs); yp = torch.cat(ys)
        idx = torch.randperm(xp.shape[0], generator=g)[:per]
        Xs.append(xp[idx]); Ys.append(yp[idx])
    X, Y = P.subsample(torch.cat(Xs), torch.cat(Ys), N_TR, SEED)
    torch.manual_seed(SEED); clf = torch.nn.Linear(X.shape[1], P.N_CLASS).to(DEVICE)
    opt = torch.optim.Adam(clf.parameters(), 1e-3, weight_decay=1e-4)
    lf = torch.nn.CrossEntropyLoss(ignore_index=P.IGNORE); X, Y = X.to(DEVICE).float(), Y.to(DEVICE)
    for _ in range(800):
        opt.zero_grad(); lf(clf(X), Y).backward(); opt.step()
    return clf


@torch.no_grad()
def eval_sphere(enc, clf, cache_va, plan):
    inter = torch.zeros(P.N_CLASS); union = torch.zeros(P.N_CLASS); cov_tot = 0.0
    for rgb, lab in cache_va:
        gt = torch.from_numpy(P.label_to_grid(lab, HS, WS).reshape(-1))
        if plan is None:                                    # erp-direct: predict ERP grid, upsample
            ef, _ = P.feats_erp(enc, rgb, lab)              # (N,D), N = (IH/patch)*(IW/patch)
            gh, gw = IH // enc.patch, IW // enc.patch
            pr = clf(ef.to(DEVICE).float()).argmax(1).reshape(1, 1, gh, gw).float()
            pred = F.interpolate(pr, size=(HS, WS), mode="nearest")[0, 0].long().reshape(-1).cpu()
            cov = torch.ones(HS * WS, dtype=torch.bool)
        else:
            cid_all, r_all, pr_all = [], [], []
            for (yaw, pitch, hfov) in plan:
                f = tile_feat(enc, rgb, yaw, pitch, hfov); d, gh, gw = f.shape
                p32 = clf(f.reshape(d, -1).t().float()).argmax(1).reshape(gh, gw).cpu()
                p = F.interpolate(p32[None, None].float(), size=(SR, SR), mode="nearest")[0, 0].long().numpy().reshape(-1)
                cid, r = tile_cells(yaw, pitch, hfov)
                cid_all.append(cid); r_all.append(r); pr_all.append(p)
            cid_all = np.concatenate(cid_all); r_all = np.concatenate(r_all); pr_all = np.concatenate(pr_all)
            pred = np.zeros(HS * WS, np.int64); rbuf = np.full(HS * WS, np.inf)
            order = np.argsort(-r_all)                       # write smallest-obliquity LAST -> wins
            pred[cid_all[order]] = pr_all[order]; rbuf[cid_all[order]] = r_all[order]
            cov = torch.from_numpy(rbuf < np.inf); pred = torch.from_numpy(pred)
        m = (gt != P.IGNORE) & cov
        for c in range(1, P.N_CLASS):
            inter[c] += ((pred == c) & (gt == c) & m).sum(); union[c] += (((pred == c) | (gt == c)) & m).sum()
        cov_tot += cov.float().mean().item()
    miou = float(np.mean([(inter[c] / union[c]).item() for c in range(1, P.N_CLASS) if union[c] > 0]))
    return miou, cov_tot / len(cache_va)


def main():
    P.configure(DATASET); P.TILE = TILE; P.WORK_HW = (IH, IW)
    enc = PanoEncoder(model_id=P.MODEL, lora_rank=0).to(DEVICE).eval(); P.enc_patch = enc.patch
    if SPLIT == "area5":                                    # match SphereUFormer fold: train 1,2,3,6 / val 5a,5b
        import data
        files = data.list_erps(DATASET)
        def _area(f): return f.split("extracted_data/")[1].split("/")[0]
        trn = int(os.environ.get("TRN", "10000"))               # cap train panos (frozen+linear saturates fast)
        tr_files = [f for f in files if _area(f) in ("area_1", "area_2", "area_3", "area_6")][:trn]
        ctr = [P.load_rgb_label(f) for f in tr_files]
        cva = [P.load_rgb_label(f) for f in files if _area(f) in ("area_5a", "area_5b")]
    else:
        panos, groups, train = P.grouped()
        if DATASET == "densepass":
            panos = panos[:60]
        cache = [("tr" if g in train else "va", P.load_rgb_label(f)) for g, f in panos]
        ctr = [rl for sp, rl in cache if sp == "tr"]; cva = [rl for sp, rl in cache if sp == "va"]
    M = methods()
    if METHODS:
        M = {k: v for k, v in M.items() if k in METHODS}
    print(f"ERP decomposition (Tangent-Images-style: predict→resample-to-sphere→sphere mIoU)\n"
          f"dataset={DATASET} frozen DINOv3 | tr={len(ctr)} va={len(cva)} grid={HS}x{WS}\n"
          f"{'method':14s} {'#views':>6} {'sphereMIoU':>10} {'coverage':>9}", flush=True)
    rows = []
    for name, plan in M.items():
        clf = train_head(enc, ctr, plan)
        miou, cov = eval_sphere(enc, clf, cva, plan)
        nv = 1 if plan is None else len(plan)
        rows.append((name, nv, miou, cov))
        print(f"{name:14s} {nv:6d} {miou:10.3f} {cov:9.2f}", flush=True)
    rows.sort(key=lambda r: -r[2])
    print("\n=== ranked by sphere mIoU ===")
    for name, nv, miou, cov in rows:
        tag = "  ⭐OURS(E2P)" if name.startswith("e2p") else ("  📄TangentImages" if name.startswith("tangent") else "")
        print(f"{miou:7.3f}  {name:14s} ({nv} views, cov {cov:.2f}){tag}")


if __name__ == "__main__":
    main()
