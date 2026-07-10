"""Gate 0.5 (TRAINING-FREE) for open lever #2 — does raw-disagreement inverse-variance depth
fusion beat uniform mean, before building any sigma head? (docs/INVVAR_FUSION_LOG.md)

Advisor triage: at inference the overlapping tiles are already present, so we get sigma from raw
cross-tile disagreement with ZERO training: sigma_i(p) = |logd_i(p) - median_{j!=i} logd_j(p)|
(leave-one-out deviation), closed-form IV-fuse the EXISTING frozen linear-probe depth, compare to
uniform mean / median / trimmed on held-out area-5.

NOTE (coverage): loo-sigma makes IV == uniform at coverage-2 (sigma_A = sigma_B). Raw-sigma IV can
only act at coverage>=3, so we report the coverage split. A shrinkage prior tau (global median sigma)
keeps IV from over-trusting a lucky-agreeing view (the correlated-error failure the advisor flagged).

Triage (NOT a hard kill — a head could denoise raw sigma):
  IV clearly beats uniform (esp. cov>=3 / high-disagreement)  -> strong greenlight (raw-sigma IV may
                                                                  be the deliverable, train-free)
  IV ties uniform / only ties median                          -> the head is the last hope; it must
                                                                  beat THIS raw-sigma ceiling
  IV loses to uniform                                          -> correlated-error over-trust is real
                                                                  -> strong kill

Metric = per-pano-normalized log-depth error vs Stanford2D3D GT (same space as RESULTS §3.6). Frozen
DINOv3 + the pointmap_fusion linear depth probe. This is a single-split TRIAGE, not the Gate 2 de-risk.

Run: OPENCV_IO_ENABLE_OPENEXR=1 CUDA_VISIBLE_DEVICES=<n> conda run -n pano python scripts/diag_invvar_fusion.py
Knobs: NTR (probe-train panos, def 60), NVA (val panos, def 30), FIELD (def 64 => 64x128).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pointmap_fusion as PF  # noqa: E402
import geometry as G  # noqa: E402
from encoder import PanoEncoder  # noqa: E402

DEVICE = PF.DEVICE
ERP_H, ERP_W, TILE = PF.ERP_H, PF.ERP_W, PF.TILE
NTR = int(os.environ.get("NTR", 60))
NVA = int(os.environ.get("NVA", 30))
FH = int(os.environ.get("FIELD", 64))
FW = FH * 2                                                            # 64x128 = §3.8 sweet spot


def tile_cid_obliq(yaw, pitch, hfov, gh, gw):
    """Per tile-cell: shared ERP field-cell id (FH x FW) and obliquity weight cos(off-axis)."""
    cm = G.render_coordmap(ERP_H, ERP_W, yaw, pitch, hfov, TILE)       # (TILE,TILE,2)=(x,y)
    cy = ((np.arange(gh) + 0.5) * TILE / gh).astype(int)
    cx = ((np.arange(gw) + 0.5) * TILE / gw).astype(int)
    xy = cm[np.ix_(cy, cx)]                                            # (gh,gw,2)
    uf = np.clip((xy[..., 0] / ERP_W * FW).astype(int), 0, FW - 1)
    vf = np.clip((xy[..., 1] / ERP_H * FH).astype(int), 0, FH - 1)
    cid = (vf * FW + uf).reshape(-1)
    col = np.broadcast_to(cx[None, :], (gh, gw)).astype(np.float32)
    row = np.broadcast_to(cy[:, None], (gh, gw)).astype(np.float32)
    obl = G._offaxis_cos(col, row, TILE, hfov).reshape(-1)
    return cid, obl


def gt_field(dn, val):
    """ERP normalized depth (H,W) -> (FH*FW,) field GT + valid, center-sampled."""
    h, w = dn.shape
    ys = ((np.arange(FH) + 0.5) * h / FH).astype(int)
    xs = ((np.arange(FW) + 0.5) * w / FW).astype(int)
    return dn[np.ix_(ys, xs)].reshape(-1), val[np.ix_(ys, xs)].reshape(-1) > 0.5


def loo_sigma(vals):
    k = len(vals)
    if k == 1:
        return np.zeros(1)
    sig = np.empty(k)
    for i in range(k):
        sig[i] = abs(vals[i] - np.median(np.delete(vals, i)))
    return sig


def cell_records(clf, packs, specs, hfov):
    """-> list of (vals (k,), gt, cov) for every covered field cell across val panos, + sigma pool."""
    recs, sig_pool = [], []
    for tiles, (dn, val) in packs:
        gt, gv = gt_field(dn, val)
        cid_all, ld_all = [], []
        for (yaw, pitch), t in zip(specs, tiles):
            logd = PF.pred_logd(clf, t[0]).reshape(-1)
            cid, _ = tile_cid_obliq(yaw, pitch, hfov, *t[0].shape[:2])
            cid_all.append(cid); ld_all.append(logd)
        cid = np.concatenate(cid_all); ld = np.concatenate(ld_all)
        order = np.argsort(cid, kind="stable")
        cid_s, ld_s = cid[order], ld[order]
        uniq, start, cnt = np.unique(cid_s, return_index=True, return_counts=True)
        for c, s, k in zip(uniq, start, cnt):
            if not gv[c] or gt[c] <= 1e-3:
                continue
            vals = ld_s[s:s + k]
            recs.append((vals, float(np.log(gt[c])), int(k)))
            sig_pool.append(loo_sigma(vals))
    return recs, np.concatenate(sig_pool) if sig_pool else np.zeros(1)


def fuse_all(vals, tau):
    k = len(vals)
    uni = vals.mean()
    med = np.median(vals)
    trm = np.mean(np.sort(vals)[1:-1]) if k >= 3 else uni
    sig = loo_sigma(vals)
    lam = 1.0 / (sig ** 2 + tau ** 2)
    iv = float((lam * vals).sum() / lam.sum())
    return uni, med, trm, iv


def report(recs, tau):
    err = {m: [] for m in ("uniform", "median", "trimmed", "IV")}
    cov = np.array([r[2] for r in recs])
    std = np.array([r[0].std() if r[2] > 1 else 0.0 for r in recs])
    for vals, lgt, k in recs:
        uni, med, trm, iv = fuse_all(vals, tau)
        err["uniform"].append(abs(uni - lgt)); err["median"].append(abs(med - lgt))
        err["trimmed"].append(abs(trm - lgt)); err["IV"].append(abs(iv - lgt))
    err = {m: np.array(v) for m, v in err.items()}
    hi = std >= np.quantile(std[cov >= 2], 0.70) if (cov >= 2).any() else np.zeros(len(recs), bool)
    masks = [("all covered", cov >= 1), ("cov==2", cov == 2), ("cov>=3", cov >= 3),
             ("hi-disagree(cov>=2,top30%)", (cov >= 2) & hi)]
    print(f"\ncoverage: cov1={np.mean(cov==1):.3f} cov2={np.mean(cov==2):.3f} "
          f"cov>=3={np.mean(cov>=3):.3f}   (shrinkage tau={tau:.4f})", flush=True)
    print(f"\n{'subset':30s}{'n':>8}{'uniform':>9}{'median':>9}{'trimmed':>9}{'IV':>9}"
          f"{'IV-uni':>9}", flush=True)
    for name, m in masks:
        if m.sum() == 0:
            continue
        row = {k: err[k][m].mean() for k in err}
        print(f"{name:30s}{int(m.sum()):>8}{row['uniform']:>9.4f}{row['median']:>9.4f}"
              f"{row['trimmed']:>9.4f}{row['IV']:>9.4f}{row['IV']-row['uniform']:>+9.4f}", flush=True)
    return err, cov, hi


if __name__ == "__main__":
    frozen = PanoEncoder(model_id=PF.P.MODEL, lora_rank=0).to(DEVICE).eval()
    PF.P.configure("stanford2d3d"); PF.P.TILE = TILE
    geom = PF.T.build_geometry(frozen, 65.0, (-45.0, 0.0, 45.0))
    s2d = PF.data.list_erps("stanford2d3d")

    def area(f):
        return f.split("extracted_data/")[1].split("/")[0]
    tr_f = [f for f in s2d if "5" not in area(f)][:NTR]
    va_f = [f for f in s2d if "5" in area(f)][:NVA]
    print(f"Gate 0.5 (train-free IV fusion): frozen DINOv3, tr={len(tr_f)} va={len(va_f)} "
          f"tiles/pano={len(geom['specs'])} field={FH}x{FW}", flush=True)

    def load_packs(files):
        PF.P.enc_patch = frozen.patch
        out = []
        for f in files:
            rgb = np.array(Image.open(f).convert("RGB").resize((PF.W, PF.H), Image.BILINEAR))
            dn, val = PF.load_depth(f)
            tiles = [PF.tile_pack(frozen, rgb, dn, val, y, p, geom["hfov"]) for (y, p) in geom["specs"]]
            out.append((tiles, (dn, val)))
        return out

    tr_packs = load_packs(tr_f)
    clf = PF.train_probe([[t for t in tiles] for tiles, _ in tr_packs])
    va_packs = load_packs(va_f)
    recs, sig_pool = cell_records(clf, va_packs, geom["specs"], geom["hfov"])
    tau = float(np.median(sig_pool))
    err, cov, hi = report(recs, tau)

    d_all = err["IV"].mean() - err["uniform"].mean()
    m3 = cov >= 3
    d_c3 = (err["IV"][m3].mean() - err["uniform"][m3].mean()) if m3.any() else 0.0
    d_med = err["IV"][m3].mean() - err["median"][m3].mean() if m3.any() else 0.0
    if d_c3 < -0.001:
        v = (f"GREENLIGHT — raw-sigma IV beats uniform at cov>=3 by {-d_c3:.4f}"
             + ("" if d_med < -0.0005 else "; but ~ ties median (generic robustness, not calibrated sigma)")
             + ". Train the sigma head only to beat this train-free ceiling.")
    elif d_c3 > 0.001:
        v = f"KILL — IV LOSES to uniform at cov>=3 ({d_c3:+.4f}); correlated-error over-trust is real."
    else:
        v = (f"TIE at cov>=3 ({d_c3:+.4f}) — raw sigma doesn't cash out; the head is the last hope and "
             "must beat this ceiling to justify itself.")
    print(f"\nVERDICT: {v}", flush=True)
