"""P1 FLOOR-CHECK (N2 upper bound) — is MoGe-2's single-view error single-view-LEARNABLE? (docs/MOGE_PARALLAX_LOG.md)

NOT a greenlight for lever C — a KILL gate (advisor 2026-07-09). Logic: oracle correspondence + exact rays +
true poses ⇒ triangulation → GT, so GT-correction learnability UPPER-BOUNDS parallax-correction learnability;
if the GT correction is NOT single-view-learnable, no parallax approximation is either → C dead. A PASS only
licenses the (deferred) real parallax-delta test; it does NOT greenlight the build (GT−MoGe is dominated by the
import-A "MoGe ≫ DINOv3-depth" gap, not the parallax delta).

Per E2P tile (S2D3D area_1, pano-disjoint split): target = per-tile scale correction log s_i = log median(d_gt/d_sv)
(the dominant P0 headroom). Ridge-regress it from three sources, held-out R²:
  - geom (pitch, |pitch|, mean obliquity)  -> R2_geom = E2P-vs-pinhole tiling artifact / per-crop calibration.
    If HIGH: the error is a fixed per-tile calibration, NOT something a displaced optical center fixes (parallax
    is the wrong tool -> framing A/tokenization).
  - MoGe-2's own DINOv2 features (pooled)  -> systematic-in-principle content signal.
  - frozen DINOv3 features (pooled, the DEPLOYMENT encoder).
Verdict: CLEAN KILL if NEITHER MoGe nor DINOv3 predicts beyond geom. DINOv3-BOTTLENECK if MoGe does but DINOv3
doesn't. FLOOR-PASS (import-A, not parallax; licenses only the delta test) if features predict beyond geom.

Run: CUDA_VISIBLE_DEVICES=0 OPENCV_IO_ENABLE_OPENEXR=1 conda run -n pano python scripts/diag_p1_floor.py
Knobs: NTR (train panos def 80), NTE (test panos def 40).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pointmap_fusion as PF  # noqa: E402
import data  # noqa: E402
import geometry as G  # noqa: E402
from encoder import PanoEncoder, normalize_tiles  # noqa: E402
from moge.model.v2 import MoGeModel  # noqa: E402

DEVICE = "cuda"
TILE = PF.TILE
ERP_H, ERP_W = 1024, 2048
DEPTH_SCALE, HFOV = 512.0, 65.0
NTR = int(os.environ.get("NTR", 80))
NTE = int(os.environ.get("NTE", 40))
P = PF.P


def erp_depth_m(f):
    D = cv2.imread(data.s2d3d_gt_path(f, "depth"), cv2.IMREAD_UNCHANGED).astype(np.float32)
    D = cv2.resize(D, (ERP_W, ERP_H), interpolation=cv2.INTER_NEAREST)
    v = (D > 0) & (D < 65535)
    dm = D / DEPTH_SCALE; dm[~v] = 0.0
    return dm


def tile_gt_range(dm, yaw, pitch):
    cm = G.render_coordmap(ERP_H, ERP_W, yaw, pitch, HFOV, TILE)
    xi = np.clip(np.round(cm[..., 0]).astype(int), 0, ERP_W - 1)
    yi = np.clip(np.round(cm[..., 1]).astype(int), 0, ERP_H - 1)
    return dm[yi, xi]


def ridge_r2(Xtr, ytr, Xte, yte, alphas=(1.0, 10.0, 100.0, 1000.0, 1e4)):
    """Standardized ridge; return best held-out R2 over the alpha grid (same grid for all feature sets)."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    ym = ytr.mean(); yc = ytr - ym
    d = Xtr.shape[1]
    A = Xtr.T @ Xtr
    XtY = Xtr.T @ yc
    sstot = ((yte - yte.mean()) ** 2).sum()
    best = -1e9
    for a in alphas:
        W = np.linalg.solve(A + a * np.eye(d), XtY)
        pred = Xte @ W + ym
        r2 = 1.0 - ((yte - pred) ** 2).sum() / max(sstot, 1e-9)
        best = max(best, r2)
    return best


def main():
    moge = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl").to(DEVICE).eval()
    enc_mod = getattr(moge, "encoder", None)
    if enc_mod is None:
        cand = [m for n, m in moge.named_modules() if n.endswith("encoder")]
        enc_mod = cand[0]
    cap = {}

    def hook(mod, inp, out):
        fo = out[0] if isinstance(out, (tuple, list)) else out
        cap["f"] = fo.detach().float().mean(dim=(-2, -1))[0].cpu().numpy()  # pooled (D,)
    enc_mod.register_forward_hook(hook)

    frozen = PanoEncoder(model_id=P.MODEL, lora_rank=0).to(DEVICE).eval()
    P.configure("stanford2d3d"); P.TILE = TILE; P.enc_patch = frozen.patch
    geom = PF.T.build_geometry(frozen, HFOV, (-45.0, 0.0, 45.0))
    specs = geom["specs"]
    s2d = data.list_erps("stanford2d3d")

    def area(f):
        return f.split("extracted_data/")[1].split("/")[0]
    a1 = [f for f in s2d if area(f) == "area_1"]
    tr_f, te_f = a1[:NTR], a1[NTR:NTR + NTE]
    print(f"P1 floor-check: train {len(tr_f)} / test {len(te_f)} area_1 panos, tiles/pano={len(specs)}", flush=True)

    @torch.no_grad()
    def rows(files):
        Y, Fm, Fd, Gg = [], [], [], []
        for f in files:
            rgb = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (ERP_W, ERP_H), interpolation=cv2.INTER_AREA)
            dm = erp_depth_m(f)
            for (yaw, pitch) in specs:
                tile = np.asarray(P.a2p.erp_to_pinhole_tile(rgb, yaw, pitch, HFOV, TILE))
                img = torch.from_numpy(tile).float().permute(2, 0, 1).to(DEVICE) / 255.0
                out = moge.infer(img, apply_mask=True, fov_x=HFOV)
                rng = np.linalg.norm(out["points"].detach().float().cpu().numpy(), axis=-1)
                msk = out["mask"].detach().cpu().numpy().astype(bool)
                fmoge = cap.get("f")
                gt = tile_gt_range(dm, yaw, pitch)
                m = msk & (gt > 1e-3) & np.isfinite(rng) & (rng > 1e-3)
                if m.sum() < 64 or fmoge is None:
                    continue
                s = np.median(gt[m] / rng[m])
                fd = frozen(normalize_tiles(img[None]))[0].mean(dim=(-2, -1)).cpu().numpy()  # DINOv3 pooled
                col = ((np.arange(TILE)[None, :]).repeat(TILE, 0)).astype(np.float32)
                row = ((np.arange(TILE)[:, None]).repeat(TILE, 1)).astype(np.float32)
                obl = G._offaxis_cos(col, row, TILE, HFOV).mean()
                Y.append(np.log(max(s, 1e-6)))
                Fm.append(fmoge); Fd.append(fd)
                Gg.append([pitch, abs(pitch), float(obl)])
        return np.array(Y), np.array(Fm), np.array(Fd), np.array(Gg)

    Ytr, Fmtr, Fdtr, Gtr = rows(tr_f)
    Yte, Fmte, Fdte, Gte = rows(te_f)
    print(f"tiles: train={len(Ytr)} test={len(Yte)} | target log s: train std={Ytr.std():.3f} "
          f"(scale spread ~{np.exp(Ytr.std()):.2f}x)", flush=True)

    r2_geom = ridge_r2(Gtr, Ytr, Gte, Yte)
    r2_moge = ridge_r2(Fmtr, Ytr, Fmte, Yte)
    r2_dino = ridge_r2(Fdtr, Ytr, Fdte, Yte)
    r2_gm = ridge_r2(np.hstack([Gtr, Fmtr]), Ytr, np.hstack([Gte, Fmte]), Yte)
    r2_gd = ridge_r2(np.hstack([Gtr, Fdtr]), Ytr, np.hstack([Gte, Fdte]), Yte)

    print(f"\nheld-out R2 predicting per-tile log-scale-correction:", flush=True)
    print(f"  geometry (pitch,obliq)      {r2_geom:+.3f}   <- tiling artifact / per-crop calibration", flush=True)
    print(f"  MoGe DINOv2 feat            {r2_moge:+.3f}", flush=True)
    print(f"  DINOv3 feat (deploy enc)    {r2_dino:+.3f}", flush=True)
    print(f"  geom + MoGe                 {r2_gm:+.3f}   (beyond-geom {r2_gm - r2_geom:+.3f})", flush=True)
    print(f"  geom + DINOv3               {r2_gd:+.3f}   (beyond-geom {r2_gd - r2_geom:+.3f})", flush=True)

    THR = 0.05
    moge_beyond = r2_gm - r2_geom
    dino_beyond = r2_gd - r2_geom
    if max(r2_moge, moge_beyond) < THR and max(r2_dino, dino_beyond) < THR:
        v = ("CLEAN KILL — neither MoGe nor DINOv3 features predict the correction beyond geometry. The "
             "MoGe error is aleatoric / not single-view-learnable -> no parallax label can raise single-view "
             "accuracy -> C is B-in-costume. Take framing A.")
    elif r2_geom > 0.5 and max(moge_beyond, dino_beyond) < THR:
        v = (f"WRONG-TOOL — the scale drift is mostly geometry-predictable (R2_geom={r2_geom:.2f}) = a fixed "
             "per-tile/per-ring calibration or full-ERP-MoGe fix, NOT something inter-pano parallax resolves. "
             "That is framing A/tokenization, not C.")
    elif moge_beyond >= THR and dino_beyond < THR:
        v = (f"DINOv3-BOTTLENECK — MoGe features predict the correction beyond geom ({moge_beyond:+.3f}) but "
             f"DINOv3 (the deployment encoder) does not ({dino_beyond:+.3f}). Systematic in principle, but "
             "distilling into DINOv3+LoRA is substrate-limited. Informative, not a clean pass.")
    else:
        v = (f"FLOOR-PASS (import-A, NOT parallax) — features predict the correction beyond geom "
             f"(MoGe {moge_beyond:+.3f}, DINOv3 {dino_beyond:+.3f}). MoGe left single-view-learnable error on "
             "the table. This LICENSES ONLY the real parallax-delta test (target = parallax-refined − single "
             "MoGe, needs cross-pano); it does NOT greenlight the build.")
    print(f"\nVERDICT: {v}", flush=True)


if __name__ == "__main__":
    main()
