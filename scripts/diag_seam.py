"""D1 seam diagnostic (NO TRAINING of the encoder): does frozen-E2P leave a
cross-tile inconsistency for the overlap-SSL to fix?

Pipeline: train ONE shared linear head on pooled E2P tile features (frozen DINOv3),
then for every val pano scatter each tile's per-cell prediction (and feature) back
onto the ERP feature grid via the e2p coord-map trick (two adjacent tiles that hit
the same ERP cell ARE the geometric correspondence the SSL warp loss aligns).

Per ERP cell we record coverage (#tiles), the multiset of tile predictions, the
blended feature (mean over covering tiles), and which contribution is least-oblique
(nearest tile center). We then report, on val:
  - disagreement rate on seam cells (cov>=2)   <- exactly what F_A(p)~F_B(Hp) drives to 0
  - mIoU interior (cov==1) vs seam (cov>=2), blended head
  - mIoU seam blended vs seam single-best  -> isolates inconsistency from raw obliquity

GO (SSL room): disagreement non-trivial AND blended-seam < single-seam (averaging
inconsistent cross-tile features hurts). NO-GO: disagreement ~0, blend ~ single.

Run: CUDA_VISIBLE_DEVICES=1 conda run -n pano python scripts/diag_seam.py [densepass|stanford2d3d|structured3d]
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import probe_seg_dinov3 as P  # noqa: E402
from encoder import PanoEncoder, normalize_tiles  # noqa: E402

DATASET = sys.argv[1] if len(sys.argv) > 1 else "densepass"
HFOV = float(sys.argv[2]) if len(sys.argv) > 2 else 90.0
OVERLAP, TILE, SEED = 0.25, 512, 0
N_TR = 40000
DEVICE = P.DEVICE


def tile_plan():
    if DATASET == "densepass":                       # content is an equator band
        n = P.a2p._ring_yaw_count(HFOV, OVERLAP, 0.0, 90.0)
        return [P.a2p.TilePlan(y, 0.0) for y in P.a2p._ring_yaws(n, 0.0)]
    return P.a2p.plan_tiles("band", HFOV, HFOV, OVERLAP, pmax_deg=P.PMAX)


@torch.no_grad()
def tile_feat_pred(enc, rgb, tp, head=None):
    """-> feat (gh,gw,D), and (if head) pred (gh,gw)."""
    tile = np.asarray(P.a2p.erp_to_pinhole_tile(rgb, tp.yaw_deg, tp.pitch_deg, HFOV, TILE))
    x = torch.from_numpy(tile).float().permute(2, 0, 1)[None] / 255.0
    feat = P.dense(enc, normalize_tiles(x.to(DEVICE)))[0]          # (D,gh,gw)
    d, gh, gw = feat.shape
    fmap = feat.permute(1, 2, 0).cpu()                            # (gh,gw,D)
    if head is None:
        return fmap, None, (gh, gw)
    pred = head(fmap.reshape(-1, d).to(DEVICE)).argmax(1).cpu().reshape(gh, gw)
    return fmap, pred, (gh, gw)


def coord_grid(rgb_hw, tp, gh, gw):
    """For each (gh,gw) tile cell, the source ERP cell (vf,uf) at feature resolution."""
    h, w = rgb_hw
    hf, wf = h // P.enc_patch, w // P.enc_patch
    uy = np.broadcast_to(np.arange(w, dtype=np.float32)[None, :], (h, w))
    vy = np.broadcast_to(np.arange(h, dtype=np.float32)[:, None], (h, w))
    um = P.e2p(uy[:, :, None], HFOV, tp.yaw_deg, tp.pitch_deg, out_hw=(TILE, TILE), mode="nearest")[:, :, 0]
    vm = P.e2p(vy[:, :, None], HFOV, tp.yaw_deg, tp.pitch_deg, out_hw=(TILE, TILE), mode="nearest")[:, :, 0]
    cy = ((np.arange(gh) + 0.5) * TILE / gh).astype(int)
    cx = ((np.arange(gw) + 0.5) * TILE / gw).astype(int)
    us = um[np.ix_(cy, cx)]; vs = vm[np.ix_(cy, cx)]              # (gh,gw) ERP pixel
    uf = np.clip((us / w * wf).astype(int), 0, wf - 1)
    vf = np.clip((vs / h * hf).astype(int), 0, hf - 1)
    return vf * wf + uf, (hf, wf)                                 # flat ERP-cell id


def head_on_tiles(enc, cache, plan):
    Xs, Ys = [], []
    for rgb, lab in cache["tr"]:
        for tp in plan:
            fmap, _, (gh, gw) = tile_feat_pred(enc, rgb, tp)
            gl = P.label_to_grid(P.e2p_label(lab, tp.yaw_deg, tp.pitch_deg, HFOV, TILE), gh, gw)
            Xs.append(fmap.reshape(-1, fmap.shape[-1])); Ys.append(torch.from_numpy(gl.reshape(-1)))
    X, Y = P.subsample(torch.cat(Xs), torch.cat(Ys), N_TR, SEED)
    torch.manual_seed(SEED)
    clf = torch.nn.Linear(X.shape[1], P.N_CLASS).to(DEVICE)
    opt = torch.optim.Adam(clf.parameters(), 1e-3, weight_decay=1e-4)
    lf = torch.nn.CrossEntropyLoss(ignore_index=P.IGNORE)
    X, Y = X.to(DEVICE).float(), Y.to(DEVICE)
    for _ in range(800):
        opt.zero_grad(); lf(clf(X), Y).backward(); opt.step()
    return clf


def scatter_pano(enc, rgb, lab, plan, head):
    h, w = rgb.shape[:2]
    hf, wf = h // P.enc_patch, w // P.enc_patch
    ncell = hf * wf
    D = enc.dim
    fsum = torch.zeros(ncell, D)
    cov = np.zeros(ncell, int)
    best_r = np.full(ncell, 1e9)
    best_pred = np.zeros(ncell, int)
    disagree = np.zeros(ncell, bool)
    first_pred = np.full(ncell, -1, int)
    gt = P.label_to_grid(lab, hf, wf).reshape(-1)
    gh = gw = TILE // P.enc_patch
    ii, jj = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
    r = np.sqrt((ii - (gh - 1) / 2) ** 2 + (jj - (gw - 1) / 2) ** 2).reshape(-1)
    for tp in plan:
        fmap, pred, (gh, gw) = tile_feat_pred(enc, rgb, tp, head)
        cid, _ = coord_grid((h, w), tp, gh, gw)
        cid = cid.reshape(-1); pr = pred.reshape(-1).numpy()
        fm = fmap.reshape(-1, D)
        for k in range(cid.shape[0]):
            c = cid[k]
            fsum[c] += fm[k]; cov[c] += 1
            if first_pred[c] < 0:
                first_pred[c] = pr[k]
            elif pr[k] != first_pred[c]:
                disagree[c] = True
            if r[k] < best_r[c]:
                best_r[c] = r[k]; best_pred[c] = pr[k]
    return fsum, cov, best_pred, disagree, gt


def main():
    P.configure(DATASET)
    enc = PanoEncoder(model_id=P.MODEL, lora_rank=0).to(DEVICE).eval()
    P.enc_patch = enc.patch
    plan = tile_plan()
    panos, groups, train = P.grouped()
    cache = {"tr": [], "va": []}
    for g, f in panos:
        cache["tr" if g in train else "va"].append(P.load_rgb_label(f))
    print(f"dataset={DATASET} N_CLASS={P.N_CLASS} tiles/pano={len(plan)} "
          f"hfov={HFOV} overlap={OVERLAP} | tr_panos={len(cache['tr'])} va_panos={len(cache['va'])}", flush=True)

    head = head_on_tiles(enc, cache, plan)

    pb, ps, gts, covs, dis = [], [], [], [], []
    for rgb, lab in cache["va"]:
        fsum, cov, best_pred, disagree, gt = scatter_pano(enc, rgb, lab, plan, head)
        m = cov >= 1
        blend = fsum[m] / torch.from_numpy(cov[m]).float()[:, None]
        with torch.no_grad():
            pblend = head(blend.to(DEVICE).float()).argmax(1).cpu().numpy()
        pb.append(pblend); ps.append(best_pred[m]); gts.append(gt[m])
        covs.append(cov[m]); dis.append(disagree[m])
    pb = np.concatenate(pb); ps = np.concatenate(ps)
    gts = np.concatenate(gts); covs = np.concatenate(covs); dis = np.concatenate(dis)

    def miou(pred, gt):
        return P.miou_acc(torch.from_numpy(pred), torch.from_numpy(gt))

    inter = covs == 1; seam = covs >= 2
    valid_seam = seam & (gts != P.IGNORE)
    dr = dis[valid_seam].mean() if valid_seam.sum() else 0.0
    mi_all = miou(pb, gts)[0]
    mi_int = miou(pb[inter], gts[inter])[0]
    mi_seam_b = miou(pb[seam], gts[seam])[0]
    mi_seam_s = miou(ps[seam], gts[seam])[0]
    cov_frac = seam.mean()

    print(f"\n=== D1 seam diagnostic: {DATASET} ===")
    print(f"val cells: {len(covs)}  seam(cov>=2) frac = {cov_frac:.3f}")
    print(f"cross-tile DISAGREEMENT rate on seam cells = {dr:.3f}   (SSL drives this ->0)")
    print(f"mIoU  all={mi_all:.3f}  interior(cov1)={mi_int:.3f}  seam(cov>=2)={mi_seam_b:.3f}")
    print(f"      seam-interior gap = {mi_seam_b - mi_int:+.3f}")
    print(f"mIoU seam: blended={mi_seam_b:.3f}  single-best(least oblique)={mi_seam_s:.3f}  "
          f"blend-single = {mi_seam_b - mi_seam_s:+.3f}")
    go = dr > 0.08 and (mi_seam_b < mi_seam_s - 0.003 or mi_seam_b < mi_int - 0.003)
    print(f"\nverdict: {'GO  (frozen-E2P is cross-tile inconsistent -> SSL has room)' if go else 'weak/NO-GO (little seam headroom on this axis)'}")


if __name__ == "__main__":
    main()
