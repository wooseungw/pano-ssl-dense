"""Adaptive ERP feature-field fusion: fuse overlapping E2P tile patch-features into ONE
dense panorama feature field, filling each ERP cell by an ADAPTIVE (obliquity-weighted)
combination of its covering tile patches — vs naive uniform averaging. Then decode the
field ONCE (linear probe) -> seg mIoU. frozen vs LoRA, naive vs adaptive.

Rationale: tile-edge patches are maximally oblique/distorted (worst features); weight them
down. Also enables a single decode pass over the field instead of per-tile (FLOPs).

Run: CUDA_VISIBLE_DEVICES=0 conda run -n pano python scripts/adaptive_field.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data  # noqa: E402
import geometry as G  # noqa: E402
import probe_seg_dinov3 as P  # noqa: E402
import diag_seam as D  # noqa: E402
import train_ssl as T  # noqa: E402
from encoder import PanoEncoder, normalize_tiles  # noqa: E402

DEVICE = P.DEVICE
TILE, HFOV, SEED = 512, 65.0, 0
_OBL = {}


def obliquity(gh, gw):
    if (gh, gw) not in _OBL:
        cy = (np.arange(gh) + 0.5) * TILE / gh
        cx = (np.arange(gw) + 0.5) * TILE / gw
        XX, YY = np.meshgrid(cx, cy)
        _OBL[(gh, gw)] = torch.from_numpy(G._offaxis_cos(XX, YY, TILE, HFOV).reshape(-1)).float()
    return _OBL[(gh, gw)]


@torch.no_grad()
def build_fields(enc, rgb, plan):
    """-> {'naive':field, 'adaptive':field} (each (Ncov, D)), mask, (Hf,Wf)."""
    P.enc_patch = enc.patch
    h, w = rgb.shape[:2]; hf, wf = h // enc.patch, w // enc.patch
    fs_n = torch.zeros(hf * wf, enc.dim); ws_n = torch.zeros(hf * wf)
    fs_a = torch.zeros(hf * wf, enc.dim); ws_a = torch.zeros(hf * wf)
    for tp in plan:
        tile = np.asarray(P.a2p.erp_to_pinhole_tile(rgb, tp.yaw_deg, tp.pitch_deg, HFOV, TILE))
        x = torch.from_numpy(tile).float().permute(2, 0, 1)[None] / 255.0
        feat = P.dense(enc, normalize_tiles(x.to(DEVICE)))[0]
        d, gh, gw = feat.shape
        fmap = feat.permute(1, 2, 0).reshape(-1, d).cpu()
        cid = torch.from_numpy(D.coord_grid((h, w), tp, gh, gw)[0].reshape(-1))
        ones = torch.ones(gh * gw); obl = obliquity(gh, gw)
        fs_n.index_add_(0, cid, fmap); ws_n.index_add_(0, cid, ones)
        fs_a.index_add_(0, cid, obl[:, None] * fmap); ws_a.index_add_(0, cid, obl)
    m = ws_n > 0
    return ({"naive": fs_n[m] / ws_n[m][:, None], "adaptive": fs_a[m] / ws_a[m][:, None]},
            m.numpy(), (hf, wf))


def gt_field(lab, hf, wf, m):
    return P.label_to_grid(lab, hf, wf).reshape(-1)[m]


def probe(Xtr, ytr, Xva, yva):
    torch.manual_seed(SEED)
    return P.linear_probe(Xtr, ytr, Xva, yva)[0]


def main():
    P.configure("stanford2d3d"); P.TILE = TILE
    plan = P.a2p.plan_tiles("band", HFOV, HFOV, 0.25, pmax_deg=45.0)
    files = data.list_erps("stanford2d3d")
    def area(f): return f.split("extracted_data/")[1].split("/")[0]
    tr_f = [f for f in files if "5" not in area(f)][:70]
    va_f = [f for f in files if "5" in area(f)][:30]
    print(f"adaptive ERP feature-field: tiles/pano={len(plan)} tr={len(tr_f)} va={len(va_f)} seed={SEED}\n"
          f"{'encoder':8s} {'naive-field':>12} {'adaptive-field':>15} {'Δ':>7}", flush=True)

    for tag, kw in [("frozen", dict(lora_rank=0)), ("LoRA", dict(adapter_path=T.CKPT))]:
        enc = PanoEncoder(model_id=P.MODEL, **kw).to(DEVICE).eval()
        bank = {"tr": [], "va": []}
        for sp, fl in [("tr", tr_f), ("va", va_f)]:
            for f in fl:
                rgb, lab = P.load_rgb_label(f)
                fields, m, (hf, wf) = build_fields(enc, rgb, plan)
                y = torch.from_numpy(gt_field(lab, hf, wf, m))
                bank[sp].append((fields, y))
        res = {}
        for kind in ("naive", "adaptive"):
            Xtr = torch.cat([fl[kind] for fl, _ in bank["tr"]]); ytr = torch.cat([y for _, y in bank["tr"]])
            Xva = torch.cat([fl[kind] for fl, _ in bank["va"]]); yva = torch.cat([y for _, y in bank["va"]])
            res[kind] = probe(Xtr, ytr, Xva, yva)
        print(f"{tag:8s} {res['naive']:12.3f} {res['adaptive']:15.3f} {res['adaptive']-res['naive']:+7.3f}", flush=True)
        del enc; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
