"""P0 — is single-view MoGe-2 metric depth already near S2D3D GT? (docs/MOGE_PARALLAX_LOG.md)

The inter-pano-parallax lever (option C) is worth building ONLY if single-view monocular geometry
has meaningful error that multi-view could fix. P0 is the cheap first cut: if MoGe-2 single-view is
already near-GT, C collapses into A ('just use MoGe-2'). It also answers framing-A viability.

Per E2P tile (S2D3D area_1, indoor hfov65 3-ring), run MoGe-2 (fov_x=65) and compare its metric
RANGE (=‖points‖) to GT range (depth.png/512 m, sampled to the tile via render_coordmap). Report
AbsRel + delta<1.25 both RAW (absolute metric) and per-tile MEDIAN-SCALE-ALIGNED. The raw-vs-aligned
gap separates monocular SCALE error (parallax-correctable) from STRUCTURE error (less so). Prints
MoGe/GT medians so the depth.png/512 scale is self-validated.

Run: CUDA_VISIBLE_DEVICES=0 OPENCV_IO_ENABLE_OPENEXR=1 conda run -n pano python scripts/diag_moge_p0.py
Knobs: NVA (area_1 panos, def 20).
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pointmap_fusion as PF  # geom, P.a2p, data, ERP dims  # noqa: E402
import data  # noqa: E402
import geometry as G  # noqa: E402
from encoder import PanoEncoder  # noqa: E402
from moge.model.v2 import MoGeModel  # noqa: E402

DEVICE = "cuda"
TILE = PF.TILE
ERP_H, ERP_W = 1024, 2048
DEPTH_SCALE = 512.0                     # 2D3D-S: meters = uint16 / 512 (65535 = invalid)
HFOV = 65.0
NVA = int(os.environ.get("NVA", 20))
P = PF.P


def erp_depth_m(f):
    D = cv2.imread(data.s2d3d_gt_path(f, "depth"), cv2.IMREAD_UNCHANGED).astype(np.float32)
    D = cv2.resize(D, (ERP_W, ERP_H), interpolation=cv2.INTER_NEAREST)
    valid = (D > 0) & (D < 65535)
    dm = D / DEPTH_SCALE
    dm[~valid] = 0.0
    return dm                            # (ERP_H, ERP_W) meters, 0 = invalid


def tile_gt_range(dm, yaw, pitch):
    cm = G.render_coordmap(ERP_H, ERP_W, yaw, pitch, HFOV, TILE)      # (TILE,TILE,2)=(x,y)
    xi = np.clip(np.round(cm[..., 0]).astype(int), 0, ERP_W - 1)
    yi = np.clip(np.round(cm[..., 1]).astype(int), 0, ERP_H - 1)
    return dm[yi, xi]                    # (TILE,TILE) GT range at tile pixels


@torch.no_grad()
def moge_range(model, rgb_erp, yaw, pitch):
    tile = np.asarray(P.a2p.erp_to_pinhole_tile(rgb_erp, yaw, pitch, HFOV, TILE))  # (TILE,TILE,3) uint8
    img = torch.from_numpy(tile).float().permute(2, 0, 1).to(DEVICE) / 255.0
    out = model.infer(img, apply_mask=True, fov_x=HFOV)
    pts = out["points"].detach().float().cpu().numpy()               # (TILE,TILE,3) metric
    msk = out["mask"].detach().cpu().numpy().astype(bool)
    return np.linalg.norm(pts, axis=-1), msk


def main():
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl").to(DEVICE).eval()
    frozen = PanoEncoder(model_id=P.MODEL, lora_rank=0).to(DEVICE).eval()
    P.configure("stanford2d3d"); P.TILE = TILE; P.enc_patch = frozen.patch
    geom = PF.T.build_geometry(frozen, HFOV, (-45.0, 0.0, 45.0))
    specs = geom["specs"]
    s2d = data.list_erps("stanford2d3d")

    def area(f):
        return f.split("extracted_data/")[1].split("/")[0]
    va = [f for f in s2d if area(f) == "area_1"][:NVA]
    print(f"P0 MoGe-2 vs S2D3D GT: {len(va)} area_1 panos, tiles/pano={len(specs)}, fov={HFOV}", flush=True)

    absrel_raw, absrel_al, d125_raw, d125_al = [], [], [], []
    med_moge, med_gt = [], []
    for f in va:
        rgb = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (ERP_W, ERP_H), interpolation=cv2.INTER_AREA)
        dm = erp_depth_m(f)
        for (yaw, pitch) in specs:
            gt = tile_gt_range(dm, yaw, pitch)
            rng, msk = moge_range(model, rgb, yaw, pitch)
            m = msk & (gt > 1e-3) & np.isfinite(rng) & (rng > 1e-3)
            if m.sum() < 64:
                continue
            g, r = gt[m], rng[m]
            med_moge.append(np.median(r)); med_gt.append(np.median(g))
            absrel_raw.append(np.mean(np.abs(r - g) / g))
            d125_raw.append(np.mean(np.maximum(r / g, g / r) < 1.25))
            s = np.median(g / r)                                     # per-tile median scale align
            ra = r * s
            absrel_al.append(np.mean(np.abs(ra - g) / g))
            d125_al.append(np.mean(np.maximum(ra / g, g / ra) < 1.25))

    n = len(absrel_raw)
    print(f"\nvalid tiles N={n}", flush=True)
    print(f"scale check:  median MoGe range={np.mean(med_moge):.2f} m   median GT range={np.mean(med_gt):.2f} m"
          f"   (ratio GT/MoGe={np.mean(med_gt)/max(np.mean(med_moge),1e-6):.3f})", flush=True)
    print(f"\n{'metric':22s}{'RAW (absolute)':>16}{'median-aligned':>16}", flush=True)
    print(f"{'AbsRel  (lower=better)':22s}{np.mean(absrel_raw):>16.3f}{np.mean(absrel_al):>16.3f}", flush=True)
    print(f"{'delta<1.25 (higher=b)':22s}{np.mean(d125_raw):>16.3f}{np.mean(d125_al):>16.3f}", flush=True)

    al = np.mean(absrel_al)
    scale_err = abs(np.mean(med_gt) / max(np.mean(med_moge), 1e-6) - 1.0)
    if al < 0.05:
        v = (f"MoGe-2 is NEAR-GT in structure (aligned AbsRel {al:.3f}<0.05). Parallax structure-headroom is "
             f"small -> C largely COLLAPSES to A. Scale error {scale_err:.2f} is the only clear lever "
             f"(monocular scale, parallax-fixable) -> a light metric-scale calibration, not a big build.")
    elif al < 0.12:
        v = (f"MODERATE structure error (aligned AbsRel {al:.3f}). Some parallax headroom may exist; "
             f"proceed to P1 to test the parallax-correctable fraction. Scale error {scale_err:.2f}.")
    else:
        v = (f"LARGE structure error (aligned AbsRel {al:.3f}) -> real single-view headroom; proceed to P1 "
             f"(is it parallax-correctable, or monocular-ambiguous both views get wrong?). Scale error {scale_err:.2f}.")
    print(f"\nVERDICT: {v}", flush=True)


if __name__ == "__main__":
    main()
