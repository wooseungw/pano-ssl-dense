"""Gate 0 for the inverse-variance depth-fusion lever (docs/PANO_ADAPT_RECIPE_GATE.md, open lever #2).

EXISTENCE CONDITION for a label-free per-view uncertainty sigma: does cross-tile DISAGREEMENT
predict per-view DEPTH ERROR (vs GT), BEYOND obliquity? The heteroscedastic pairwise-residual NLL
(L = Sum ||f_i-f_j||^2/(s_i^2+s_j^2) + D log(s_i^2+s_j^2)) can only learn a useful sigma if the
residual it is trained on actually tracks task error. If disagreement does not predict error, sigma
has no signal -> the whole fusion is dead BEFORE any training (kill-cheap, minutes, no sigma head).

For each E2P overlap correspondence (A-cell <-> B-cell, shared optical center => same ray => same GT;
frozen DINOv3 + linear log-depth probe):
  errA/errB   = |pred_logd - log GT|            per-view task error vs Stanford2D3D depth
  resid_depth = |pred_logd_A - pred_logd_B|     task-space disagreement (the sigma signal)
  resid_feat  = 1 - cos(f_A, f_B)               feature-space disagreement
  obliq       = min(cos th_A, cos th_B)         the cos-lat shortcut = mode-5 control

Reports Spearman(resid, err_pair), the obliquity-only baseline, and the PARTIAL correlation
controlling for obliquity (= the feature-conditional signal that a cos-lat weight cannot capture).
Verdict: KILL if no correlation; MODE-5 if it is all obliquity; PROCEED (train the sigma head) only
if a partial signal beyond obliquity exists.

Run: OPENCV_IO_ENABLE_OPENEXR=1 CUDA_VISIBLE_DEVICES=<n> conda run -n pano python scripts/diag_disagreement_error.py
Knobs: NTR (probe-train panos, def 60), NVA (val panos, def 30).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pointmap_fusion as PF  # noqa: E402  (reuse tile_pack / train_probe / pred_logd / load_depth)
from encoder import PanoEncoder  # noqa: E402

DEVICE = PF.DEVICE
NTR = int(os.environ.get("NTR", 60))
NVA = int(os.environ.get("NVA", 30))


def _rank(a: np.ndarray) -> np.ndarray:
    order = a.argsort(kind="stable")
    r = np.empty(len(a), dtype=np.float64)
    r[order] = np.arange(len(a), dtype=np.float64)
    return r


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(_rank(a), _rank(b))[0, 1])


def partial_spearman(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Rank partial correlation corr(a, b | c) — signal in a<->b not explained by c."""
    ra, rb, rc = _rank(a), _rank(b), _rank(c)
    rab = np.corrcoef(ra, rb)[0, 1]
    rac = np.corrcoef(ra, rc)[0, 1]
    rbc = np.corrcoef(rb, rc)[0, 1]
    denom = np.sqrt(max(1e-9, (1 - rac ** 2) * (1 - rbc ** 2)))
    return float((rab - rac * rbc) / denom)


def collect(enc, packs, geom):
    """Per-overlap-cell arrays across all val panos: err_pair, err_max, resid_depth, resid_feat, obliq."""
    errp, errmx, rd, rf, ob = [], [], [], [], []
    for tiles in packs:
        logd = [PF.pred_logd(enc_clf, t[0]) for t in tiles]              # each (gh,gw)
        for (a, b), (grid, valid, weight) in zip(geom["pairs"], geom["warps"]):
            v = valid.cpu().numpy().astype(bool)
            fa_t, gd_a, gv_a = tiles[a][0], tiles[a][1], tiles[a][2]
            m = v & gv_a.reshape(-1) & (gd_a.reshape(-1) > 1e-3)
            if m.sum() < 8:
                continue
            g = grid.cpu().view(1, 1, -1, 2)
            la = logd[a].reshape(-1)
            lbw = F.grid_sample(torch.from_numpy(logd[b])[None, None].float(), g,
                                align_corners=False)[0, 0, 0].numpy()    # B depth at A cells
            D = fa_t.shape[-1]
            fb = tiles[b][0].permute(2, 0, 1)[None].float()              # (1,D,gh,gw)
            fbw = F.grid_sample(fb, g, align_corners=False)[0, :, 0, :].t().numpy()   # (N,D) B feat at A
            fa = fa_t.reshape(-1, D).numpy()
            loggt = np.log(gd_a.reshape(-1))
            errA = np.abs(la - loggt)
            errB = np.abs(lbw - loggt)
            cos = (fa * fbw).sum(1) / (np.linalg.norm(fa, axis=1) * np.linalg.norm(fbw, axis=1) + 1e-9)
            errp.append((errA + errB)[m])
            errmx.append(np.maximum(errA, errB)[m])
            rd.append(np.abs(la - lbw)[m])
            rf.append((1.0 - cos)[m])
            ob.append(weight.cpu().numpy()[m])
    return (np.concatenate(errp), np.concatenate(errmx), np.concatenate(rd),
            np.concatenate(rf), np.concatenate(ob))


def packs_for(enc, files, geom):
    PF.P.enc_patch = enc.patch
    out = []
    for f in files:
        rgb = np.array(Image.open(f).convert("RGB").resize((PF.W, PF.H), Image.BILINEAR))
        dn, val = PF.load_depth(f)
        out.append([PF.tile_pack(enc, rgb, dn, val, y, p, geom["hfov"]) for (y, p) in geom["specs"]])
    return out


if __name__ == "__main__":
    frozen = PanoEncoder(model_id=PF.P.MODEL, lora_rank=0).to(DEVICE).eval()
    PF.P.configure("stanford2d3d"); PF.P.TILE = PF.TILE
    geom = PF.T.build_geometry(frozen, 65.0, (-45.0, 0.0, 45.0))
    s2d = PF.data.list_erps("stanford2d3d")

    def area(f):
        return f.split("extracted_data/")[1].split("/")[0]
    tr_f = [f for f in s2d if "5" not in area(f)][:NTR]
    va_f = [f for f in s2d if "5" in area(f)][:NVA]

    print(f"Gate 0 (disagreement -> depth error): frozen DINOv3, tr={len(tr_f)} va={len(va_f)} "
          f"tiles/pano={len(geom['specs'])}", flush=True)
    tr = packs_for(frozen, tr_f, geom)
    enc_clf = PF.train_probe(tr)                                          # frozen log-depth probe
    va = packs_for(frozen, va_f, geom)
    errp, errmx, rd, rf, ob = collect(frozen, va, geom)
    n = len(errp)

    # raw correlations (residual/obliquity vs per-view error)
    s_rd = spearman(rd, errp)
    s_rf = spearman(rf, errp)
    s_ob = spearman(ob, errp)                                             # obliquity-only baseline (expect <0: higher weight=lower err)
    # feature/depth-conditional signal beyond obliquity (mode-5 control)
    p_rd = partial_spearman(rd, errp, ob)
    p_rf = partial_spearman(rf, errp, ob)
    # observation-model check: resid^2 ~ sigma_A^2+sigma_B^2 should track err_A^2+err_B^2
    s_obs = spearman(rd ** 2, errp ** 2)

    print(f"\noverlap cells N={n}", flush=True)
    print(f"{'signal':32s}{'Spearman vs err':>16}", flush=True)
    print(f"{'  resid_depth |dlogd|':32s}{s_rd:>16.3f}", flush=True)
    print(f"{'  resid_feat  1-cos':32s}{s_rf:>16.3f}", flush=True)
    print(f"{'  obliquity   min(cos)':32s}{s_ob:>16.3f}   (baseline / mode-5 signal)", flush=True)
    print(f"{'  resid_dep( . | obliq)':32s}{p_rd:>16.3f}   (partial: beyond obliquity)", flush=True)
    print(f"{'  resid_feat( . | obliq)':32s}{p_rf:>16.3f}   (partial: beyond obliquity)", flush=True)
    print(f"{'  obs-model resid^2~err^2':32s}{s_obs:>16.3f}", flush=True)

    # obliquity-binned per-view error: is depth error heteroscedastic across obliquity?
    q = np.quantile(ob, [0.0, 0.25, 0.5, 0.75, 1.0])
    print("\nheteroscedasticity (mean err_max by obliquity quartile, low weight=more oblique):", flush=True)
    for i in range(4):
        sel = (ob >= q[i]) & (ob <= q[i + 1])
        print(f"  obliq[{q[i]:.2f},{q[i+1]:.2f}]  n={int(sel.sum()):>7}  mean|err|={errmx[sel].mean():.3f}", flush=True)

    strong = max(abs(p_rd), abs(p_rf))
    raw = max(abs(s_rd), abs(s_rf))
    if raw < 0.05:
        verdict = "KILL — disagreement does not predict depth error; sigma has no label-free signal."
    elif strong < 0.05:
        verdict = "MODE-5 — the signal is essentially all obliquity; a learned sigma ~ fixed cos-lat weight (no feature-conditional gain)."
    else:
        verdict = (f"PROCEED — feature-conditional signal beyond obliquity exists (partial up to {strong:.3f}); "
                   "train the sigma head (Gate 1), validate sigma vs held-out GT error AUROC.")
    print(f"\nVERDICT: {verdict}", flush=True)
