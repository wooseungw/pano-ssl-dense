"""HEALPix single-patch decomposition × DINOv3 (no overlap, equal-area).
Sample the panorama onto a HEALPix grid, lay the 12 equal-area faces out as ONE mosaic image,
run frozen DINOv3 ONCE, linear-probe seg, score mIoU per HEALPix pixel. Because HEALPix pixels
are equal-area, per-pixel mIoU == sphere-uniform mIoU. Same split/probe as tiling_compare so the
number is directly comparable to erp_direct / cube* / e2p / tangent.

  --check : just dump a mosaic PNG + round-trip assertions (no GPU), to verify the HEALPix layout.

Run: CUDA_VISIBLE_DEVICES=1 conda run -n pano python scripts/healpix_seg.py [stanford2d3d]
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import healpy as hp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import probe_seg_dinov3 as P  # noqa: E402
from encoder import PanoEncoder, normalize_tiles  # noqa: E402

DEVICE = P.DEVICE
ARGS = [a for a in sys.argv[1:] if not a.startswith("-")]
DATASET = ARGS[0] if ARGS else "stanford2d3d"
CHECK = "--check" in sys.argv
NSIDE = int(os.environ.get("NSIDE", 128))                  # face = NSIDE x NSIDE px
IH, IW = 1024, 2048                                        # ERP sampling resolution
SEED, N_TR = 0, 40000
GR, GC = 3, 4                                              # 12 faces -> 3 rows x 4 cols mosaic


def nest_face_xy(nside, ipix):
    """nested ipix -> (face, x, y); x,y are the bit-deinterleaved face-local coords (HEALPix nested = Z-order)."""
    npf = nside * nside
    face = (ipix // npf).astype(np.int64)
    p = (ipix % npf).astype(np.int64)
    x = np.zeros_like(p); y = np.zeros_like(p)
    for i in range(int(np.log2(nside))):
        x |= ((p >> (2 * i)) & 1) << i
        y |= ((p >> (2 * i + 1)) & 1) << i
    return face, x, y


# --- precompute HEALPix-pixel <-> mosaic-pixel maps and ERP sample coords ---
NPIX = hp.nside2npix(NSIDE)
_ipix = np.arange(NPIX)
_face, _fx, _fy = nest_face_xy(NSIDE, _ipix)
_my = (_face // GC) * NSIDE + _fy
_mx = (_face % GC) * NSIDE + _fx
MH, MW = GR * NSIDE, GC * NSIDE
_theta, _phi = hp.pix2ang(NSIDE, _ipix, nest=True)          # theta[0,pi] colatitude, phi[0,2pi]
_ev = np.clip((_theta / np.pi * IH).astype(int), 0, IH - 1)
_eu = np.clip((_phi / (2 * np.pi) * IW).astype(int), 0, IW - 1)


def erp_to_mosaic_rgb(rgb):
    mos = np.zeros((MH, MW, 3), np.uint8)
    mos[_my, _mx] = rgb[_ev, _eu]
    return mos


def healpix_label(lab):
    return lab[_ev, _eu]                                    # (NPIX,)


@torch.no_grad()
def mosaic_feat(enc, rgb):
    x = torch.from_numpy(erp_to_mosaic_rgb(rgb)).float().permute(2, 0, 1)[None] / 255.0
    f = P.dense(enc, normalize_tiles(x.to(DEVICE)))[0]       # (D,gh,gw)
    f = F.interpolate(f[None], size=(MH, MW), mode="nearest")[0]
    return f[:, _my, _mx].t().contiguous()                  # (NPIX, D)


def train_head(enc, ctr):
    Xs, Ys = [], []
    per = max(4000, N_TR * 3 // max(1, len(ctr)))           # subsample pixels per pano (NPIX too big to keep all)
    g = torch.Generator().manual_seed(SEED)
    for rgb, lab in ctr:
        f = mosaic_feat(enc, rgb).cpu(); lb = torch.from_numpy(healpix_label(lab))
        idx = torch.randperm(NPIX, generator=g)[:per]
        Xs.append(f[idx]); Ys.append(lb[idx])
    X, Y = P.subsample(torch.cat(Xs), torch.cat(Ys), N_TR, SEED)
    torch.manual_seed(SEED); clf = torch.nn.Linear(X.shape[1], P.N_CLASS).to(DEVICE)
    opt = torch.optim.Adam(clf.parameters(), 1e-3, weight_decay=1e-4)
    lf = torch.nn.CrossEntropyLoss(ignore_index=P.IGNORE); X, Y = X.to(DEVICE).float(), Y.to(DEVICE)
    for _ in range(800):
        opt.zero_grad(); lf(clf(X), Y).backward(); opt.step()
    return clf


@torch.no_grad()
def eval_sphere(enc, clf, cva):
    inter = torch.zeros(P.N_CLASS); union = torch.zeros(P.N_CLASS)
    for rgb, lab in cva:
        pred = clf(mosaic_feat(enc, rgb).float()).argmax(1).cpu()
        gt = torch.from_numpy(healpix_label(lab)).long(); m = gt != P.IGNORE
        for c in range(1, P.N_CLASS):
            inter[c] += ((pred == c) & (gt == c) & m).sum(); union[c] += (((pred == c) | (gt == c)) & m).sum()
    return float(np.mean([(inter[c] / union[c]).item() for c in range(1, P.N_CLASS) if union[c] > 0]))


def main():
    P.configure(DATASET); P.WORK_HW = (IH, IW)
    print(f"HEALPix nside={NSIDE} npix={NPIX} mosaic={MH}x{MW} | frozen DINOv3 | dataset={DATASET}", flush=True)
    if CHECK:                                               # layout sanity: round-trip + dump a mosaic
        assert (_my * MW + _mx).size == np.unique(_my * MW + _mx).size, "mosaic pixel collision!"
        os.makedirs("docs/figures/healpix_seg", exist_ok=True)
        rgb = P.load_rgb_label(P.grouped()[0][0][1])[0]
        from PIL import Image
        Image.fromarray(erp_to_mosaic_rgb(rgb)).save("docs/figures/healpix_seg/mosaic_sample.png")
        print("OK: no mosaic collisions; saved docs/figures/healpix_seg/mosaic_sample.png"); return
    enc = PanoEncoder(model_id=P.MODEL, lora_rank=0).to(DEVICE).eval()
    panos, groups, train = P.grouped()
    if DATASET == "densepass":
        panos = panos[:60]
    cache = [("tr" if g in train else "va", P.load_rgb_label(f)) for g, f in panos]
    ctr = [rl for sp, rl in cache if sp == "tr"]; cva = [rl for sp, rl in cache if sp == "va"]
    clf = train_head(enc, ctr)
    miou = eval_sphere(enc, clf, cva)
    print(f"\nhealpix_single  nside={NSIDE}  sphere(HEALPix-pixel) mIoU = {miou:.3f}   (tr={len(ctr)} va={len(cva)})")


if __name__ == "__main__":
    main()
