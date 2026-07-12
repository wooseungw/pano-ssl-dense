"""GATE B for the distortion/architecture path (P1 axis: obliquity robustness headroom).

Diagnose-before-build (iron law #1). The spherical-RoPE / AdaLN-modulation architecture bet only has
a target if frozen DINOv3's representation quality actually DEGRADES with tile obliquity. The naive
stratification (error by obliquity bin) is CONTENT-CONFOUNDED: oblique tiles at pitch +-45 see
ceilings/floors, which may be intrinsically harder. This gate removes the confound with the overlap
machinery: every E2P overlap cell is the SAME 3D ray (shared optical center) seen by TWO tiles at
DIFFERENT off-axis angles. Pair the two views' depth-probe errors at the same cell and ask whether
the more-oblique observation errs more.

Per overlap cell (frozen DINOv3 + linear log-depth probe, S2D3D GT):
  errA/errB = |pred_logd - log GT|   per-view error at the shared cell
  cosA/cosB = off-axis cos of the cell in each tile (A at patch centers; B at the warped position)
  delta     = err(more-oblique view) - err(more-central view)     (content-controlled by pairing)

Verdict (pre-registered): on the top |cosA-cosB| gap quartile,
  HEADROOM  rel_delta = mean(delta)/mean(err) >= +5%  AND  sign-consistency >= 0.55
  FLAT      rel_delta < +2%  -> E2P tangent tiling already neutralizes distortion; kill/de-prioritize
  MARGINAL  otherwise
Also reports the naive content-confounded stratification for contrast.

Run: OPENCV_IO_ENABLE_OPENEXR=1 CUDA_VISIBLE_DEVICES=<n> conda run -n pano python scripts/diag_obliquity_headroom.py
Knobs: NTR (probe-train panos def 60), NVA (val panos def 30).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pointmap_fusion as PF  # noqa: E402
import geometry as G  # noqa: E402
from encoder import PanoEncoder  # noqa: E402

DEVICE = PF.DEVICE
HFOV = 65.0
NTR = int(os.environ.get("NTR", 60))
NVA = int(os.environ.get("NVA", 30))


def packs_for(enc, files, geom):
    PF.P.enc_patch = enc.patch
    out = []
    for f in files:
        rgb = np.array(Image.open(f).convert("RGB").resize((PF.W, PF.H), Image.BILINEAR))
        dn, val = PF.load_depth(f)
        out.append([PF.tile_pack(enc, rgb, dn, val, y, p, geom["hfov"]) for (y, p) in geom["specs"]])
    return out


def collect(clf, packs, geom, patch):
    """Per-overlap-cell arrays: errA, errB, cosA, cosB (per-VIEW obliquity at the shared cell)."""
    grid_n = PF.TILE // patch
    half = patch // 2
    gi, gj = np.mgrid[0:grid_n, 0:grid_n]
    a_row = (gi * patch + half).ravel().astype(np.float32)
    a_col = (gj * patch + half).ravel().astype(np.float32)
    cosA_all = G._offaxis_cos(a_col, a_row, PF.TILE, HFOV)              # same for every tile
    eA, eB, cA, cB = [], [], [], []
    for tiles in packs:
        logd = [PF.pred_logd(clf, t[0]) for t in tiles]
        for (a, b), (grid, valid, _w) in zip(geom["pairs"], geom["warps"]):
            v = valid.cpu().numpy().astype(bool)
            gd_a, gv_a = tiles[a][1], tiles[a][2]
            m = v & gv_a.reshape(-1) & (gd_a.reshape(-1) > 1e-3)
            if m.sum() < 8:
                continue
            gnp = grid.cpu().numpy()                                     # (N,2) normalized on B
            mb_col = (gnp[:, 0] + 1.0) / 2.0 * PF.TILE - 0.5
            mb_row = (gnp[:, 1] + 1.0) / 2.0 * PF.TILE - 0.5
            cosB_all = G._offaxis_cos(mb_col.astype(np.float32), mb_row.astype(np.float32),
                                      PF.TILE, HFOV)
            g = grid.cpu().view(1, 1, -1, 2)
            la = logd[a].reshape(-1)
            lbw = F.grid_sample(torch.from_numpy(logd[b])[None, None].float(), g,
                                align_corners=False)[0, 0, 0].numpy()   # B's depth at A's cells
            loggt = np.log(gd_a.reshape(-1))
            eA.append(np.abs(la - loggt)[m]); eB.append(np.abs(lbw - loggt)[m])
            cA.append(cosA_all[m]); cB.append(cosB_all[m])
    return (np.concatenate(eA), np.concatenate(eB), np.concatenate(cA), np.concatenate(cB))


def main():
    frozen = PanoEncoder(model_id=PF.P.MODEL, lora_rank=0).to(DEVICE).eval()
    PF.P.configure("stanford2d3d"); PF.P.TILE = PF.TILE
    geom = PF.T.build_geometry(frozen, HFOV, (-45.0, 0.0, 45.0))
    s2d = PF.data.list_erps("stanford2d3d")

    def area(f):
        return f.split("extracted_data/")[1].split("/")[0]
    tr_f = [f for f in s2d if "5" not in area(f)][:NTR]
    va_f = [f for f in s2d if "5" in area(f)][:NVA]
    print(f"Gate B (content-controlled obliquity headroom): frozen DINOv3, tr={len(tr_f)} "
          f"va={len(va_f)} tiles/pano={len(geom['specs'])}", flush=True)

    tr = packs_for(frozen, tr_f, geom)
    clf = PF.train_probe(tr)
    va = packs_for(frozen, va_f, geom)
    eA, eB, cA, cB = collect(clf, va, geom, frozen.patch)
    n = len(eA)
    mean_err = float(0.5 * (eA + eB).mean())
    gap = cA - cB                                                        # >0: A more central
    err_obl = np.where(gap > 0, eB, eA)
    err_ctr = np.where(gap > 0, eA, eB)
    delta = err_obl - err_ctr
    agap = np.abs(gap)
    print(f"\noverlap cells N={n}  mean|err|={mean_err:.3f}  obliquity-gap |cosA-cosB| "
          f"median={np.median(agap):.3f}", flush=True)

    # naive (content-confounded) stratification for contrast
    cv = np.concatenate([cA, cB]); ev = np.concatenate([eA, eB])
    q = np.quantile(cv, [0.0, 0.25, 0.5, 0.75, 1.0])
    print("naive per-view err by obliquity quartile (content-CONFOUNDED, most->least oblique):", flush=True)
    for i in range(4):
        s = (cv >= q[i]) & (cv <= q[i + 1])
        print(f"  cos[{q[i]:.2f},{q[i+1]:.2f}]  n={int(s.sum()):>8}  mean|err|={ev[s].mean():.3f}", flush=True)

    # content-CONTROLLED paired deltas by gap quartile
    gq = np.quantile(agap, [0.0, 0.25, 0.5, 0.75, 1.0])
    print("\npaired same-cell delta err(oblique)-err(central) by |gap| quartile (content-CONTROLLED):",
          flush=True)
    top_rel, top_sign = 0.0, 0.5
    for i in range(4):
        s = (agap >= gq[i]) & (agap <= gq[i + 1])
        d = delta[s]
        rel = float(d.mean()) / max(mean_err, 1e-9)
        sign = float((d > 0).mean() + 0.5 * (d == 0).mean())
        print(f"  gap[{gq[i]:.3f},{gq[i+1]:.3f}]  n={int(s.sum()):>8}  mean_delta={d.mean():+.4f}  "
              f"rel={rel:+.1%}  sign={sign:.3f}", flush=True)
        if i == 3:
            top_rel, top_sign = rel, sign

    if top_rel >= 0.05 and top_sign >= 0.55:
        v = (f"HEADROOM — at the same 3D point the more-oblique view errs {top_rel:+.1%} more "
             f"(sign {top_sign:.2f}) with content controlled. Distortion genuinely costs frozen "
             "representation quality -> the spherical-RoPE / AdaLN architecture lever has a real "
             "target (evaluate as ARCHITECTURE, never as SSL — PANO_ADAPT_RECIPE_GATE.md).")
    elif top_rel < 0.02:
        v = (f"FLAT — same-cell oblique-vs-central delta is {top_rel:+.1%}; E2P tangent tiling "
             "already neutralizes distortion at the representation level. The naive obliquity "
             "stratification was content confound. Kill/de-prioritize the P1 architecture bet.")
    else:
        v = (f"MARGINAL — delta {top_rel:+.1%}, sign {top_sign:.2f}; a small real effect. "
             "Architecture bet is low-EV; only worth bundling with an independently-justified change.")
    print(f"\nVERDICT: {v}", flush=True)


if __name__ == "__main__":
    main()
