"""HEAL-SWIN (Carlsson et al., CVPR2024) from-scratch on Stanford2D3D seg — head-to-head vs our transfer.
We EXTRACT the pure-torch SwinHPTransformerSys from the HEAL-SWIN repo (Lightning/MLflow/WoodScape infra
dropped) and train it in our pano env (torch2.8) on our area5-fold, scored on the HEALPix grid (equal-area
== sphere-uniform mIoU). nside64 (~49k pix) ≈ SphereUFormer rank-6 (40,962 vert) for a resolution-matched
from-scratch-spherical comparison.

Run: CUDA_VISIBLE_DEVICES=1 conda run -n pano python scripts/healswin_train.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import healpy as hp

HEALSWIN = "/data/1_personal/4_SWWOO/HEAL-SWIN"
sys.path.insert(0, HEALSWIN)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from heal_swin.models_torch.swin_hp_transformer import SwinHPTransformerConfig, SwinHPTransformerSys  # noqa: E402
from heal_swin.data.segmentation.data_spec import DataSpec  # noqa: E402
import data  # noqa: E402
import probe_seg_dinov3 as P  # noqa: E402

DEVICE = "cuda"
NSIDE = int(os.environ.get("NSIDE", 64)); NPIX = hp.nside2npix(NSIDE)
IH, IW = 1024, 2048
EPOCHS = int(os.environ.get("EPOCHS", 120))
BS = int(os.environ.get("BS", 4))
SEED = 0

# ERP -> HEALPix nested sampling map
_theta, _phi = hp.pix2ang(NSIDE, np.arange(NPIX), nest=True)
_EV = np.clip((_theta / np.pi * IH).astype(int), 0, IH - 1)
_EU = np.clip((_phi / (2 * np.pi) * IW).astype(int), 0, IW - 1)


def to_hp(rgb, lab):
    return (rgb[_EV, _EU].astype(np.float32) / 255.0), lab[_EV, _EU]   # [npix,3], [npix]


def cache(fs):
    X, Y = [], []
    for f in fs:
        rgb, lab = P.load_rgb_label(f); x, y = to_hp(rgb, lab)
        X.append(torch.from_numpy(x).permute(1, 0)); Y.append(torch.from_numpy(y).long())  # [3,npix],[npix]
    return torch.stack(X), torch.stack(Y)


@torch.no_grad()
def miou(model, Xva, Yva):
    model.eval(); inter = torch.zeros(P.N_CLASS); union = torch.zeros(P.N_CLASS)
    for i in range(0, len(Xva), BS):
        o = model(Xva[i:i + BS].to(DEVICE)).argmax(1).cpu()
        g = Yva[i:i + BS]; m = g != P.IGNORE
        for c in range(1, P.N_CLASS):
            inter[c] += ((o == c) & (g == c) & m).sum(); union[c] += (((o == c) | (g == c)) & m).sum()
    return float(np.mean([(inter[c] / union[c]).item() for c in range(1, P.N_CLASS) if union[c] > 0]))


def main():
    P.configure("stanford2d3d"); P.WORK_HW = (IH, IW)
    files = data.list_erps("stanford2d3d")
    def area(f): return f.split("extracted_data/")[1].split("/")[0]
    tr_f = [f for f in files if area(f) in ("area_1", "area_2", "area_3", "area_6")]
    va_f = [f for f in files if area(f) in ("area_5a", "area_5b")]
    print(f"HEAL-SWIN from-scratch | nside={NSIDE} npix={NPIX} | tr={len(tr_f)} va={len(va_f)} classes={P.N_CLASS}", flush=True)
    Xtr, Ytr = cache(tr_f); Xva, Yva = cache(va_f)

    ds = DataSpec(dim_in=NPIX, f_in=3, f_out=P.N_CLASS, base_pix=12, class_names=[str(i) for i in range(P.N_CLASS)])
    torch.manual_seed(SEED)
    model = SwinHPTransformerSys(SwinHPTransformerConfig(), ds).to(DEVICE)
    n_param = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=0.05)
    lossf = torch.nn.CrossEntropyLoss(ignore_index=P.IGNORE)
    print(f"params={n_param:.2f}M  start training {EPOCHS} epochs bs={BS}", flush=True)

    g = torch.Generator().manual_seed(SEED); best = 0.0
    for ep in range(EPOCHS):
        model.train()
        for idx in torch.randperm(len(Xtr), generator=g).split(BS):
            xb = Xtr[idx].to(DEVICE); yb = Ytr[idx].to(DEVICE)
            opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
        m = miou(model, Xva, Yva); best = max(best, m)
        print(f"ep{ep:3d}  sphere(HEALPix) mIoU {m:.4f}  best {best:.4f}", flush=True)
    print(f"\nHEAL-SWIN from-scratch BEST sphere mIoU = {best:.4f}  (nside{NSIDE}, {n_param:.1f}M params)")


if __name__ == "__main__":
    main()
