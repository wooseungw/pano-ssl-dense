"""Axis-1 vs Axis-2 probe — is frozen DINOv3's depth weakness a MISSING-INFO (encoder) problem
or an EXTRACTABILITY (decoder) problem? (docs/FAILURE_ANALYSIS.md two-axes)

Information theory: raising I(Z;Y) needs NEW info F lacks (axis 1). But our depth linear-probe weakness
(0.13-0.19 log-err) could instead be axis 2 — F HAS the info, a weak (linear) decoder just misses it, and a
strong decoder re-derives it (§3.9 laundering). MoGe (~0.069 AbsRel, ~frozen DINOv2 + strong decoder + sup
data) hints axis-2. This probe settles it for DEPTH: train linear / MLP / conv heads on FROZEN DINOv3 features
(supervised on GT), compare to MoGe on the same metric.

  strong head closes most of linear->MoGe gap  => AXIS-2 (F has the geometry; decoder was the bottleneck)
                                                    => encoder-side injection (physical prior / parallax) has
                                                       little room; the answer is a decoder, not SSL.
  strong head stays far below MoGe             => AXIS-1 (F genuinely lacks it) => encoder injection has room.

Per E2P tile (S2D3D area_1, pano-disjoint), predict log metric depth at the 32x32 feature grid; eval AbsRel +
delta<1.25 (raw & per-tile scale-aligned) vs GT (depth.png/512), alongside MoGe downsampled to the same grid.

Run: CUDA_VISIBLE_DEVICES=0 OPENCV_IO_ENABLE_OPENEXR=1 conda run -n pano python scripts/diag_axis12.py
Knobs: NTR (train panos def 60), NTE (test panos def 30), EP (head epochs def 300).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pointmap_fusion as PF  # noqa: E402
import data  # noqa: E402
import geometry as G  # noqa: E402
from encoder import PanoEncoder, normalize_tiles  # noqa: E402
from moge.model.v2 import MoGeModel  # noqa: E402

DEVICE = "cuda"
TILE, ERP_H, ERP_W = PF.TILE, 1024, 2048
DEPTH_SCALE, HFOV = 512.0, 65.0
NTR = int(os.environ.get("NTR", 60))
NTE = int(os.environ.get("NTE", 30))
EP = int(os.environ.get("EP", 300))
P = PF.P


def erp_depth_m(f):
    D = cv2.imread(data.s2d3d_gt_path(f, "depth"), cv2.IMREAD_UNCHANGED).astype(np.float32)
    D = cv2.resize(D, (ERP_W, ERP_H), interpolation=cv2.INTER_NEAREST)
    v = (D > 0) & (D < 65535)
    dm = D / DEPTH_SCALE; dm[~v] = 0.0
    return dm


def tile_gt_grid(dm, yaw, pitch, gh):
    """GT metric range sampled to the gh x gh feature grid (patch centers)."""
    cm = G.render_coordmap(ERP_H, ERP_W, yaw, pitch, HFOV, TILE)
    cy = ((np.arange(gh) + 0.5) * TILE / gh).astype(int)
    cx = ((np.arange(gh) + 0.5) * TILE / gh).astype(int)
    xy = cm[np.ix_(cy, cx)]
    xi = np.clip(np.round(xy[..., 0]).astype(int), 0, ERP_W - 1)
    yi = np.clip(np.round(xy[..., 1]).astype(int), 0, ERP_H - 1)
    return dm[yi, xi]                                    # (gh,gh) metric range


class MLP(nn.Module):
    def __init__(self, d, h=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, x):                                # x (N,D)
        return self.net(x)[:, 0]


class ConvHead(nn.Module):
    def __init__(self, d, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(d, h, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(h, h, 3, padding=1), nn.GELU(), nn.Conv2d(h, 1, 1))

    def forward(self, x):                                # x (B,D,gh,gw)
        return self.net(x)[:, 0]


def metrics(pred, gt):
    m = gt > 1e-3
    p, g = pred[m], gt[m]
    absrel = float(np.mean(np.abs(p - g) / g))
    d125 = float(np.mean(np.maximum(p / g, g / p) < 1.25))
    s = np.median(g / p)
    pa = p * s
    return absrel, d125, float(np.mean(np.abs(pa - g) / g)), float(np.mean(np.maximum(pa / g, g / pa) < 1.25))


@torch.no_grad()
def collect(frozen, moge, files, geom):
    """-> per-tile: feat grid (D,gh,gw), gt grid (gh,gw), moge grid (gh,gw)."""
    feats, gts, moges = [], [], []
    for f in files:
        rgb = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (ERP_W, ERP_H), interpolation=cv2.INTER_AREA)
        dm = erp_depth_m(f)
        for (yaw, pitch) in geom["specs"]:
            tile = np.asarray(P.a2p.erp_to_pinhole_tile(rgb, yaw, pitch, HFOV, TILE))
            x = torch.from_numpy(tile).float().permute(2, 0, 1)[None].to(DEVICE) / 255.0
            fmap = frozen(normalize_tiles(x))[0]                          # (D,gh,gw)
            gh = fmap.shape[-1]
            gt = tile_gt_grid(dm, yaw, pitch, gh)
            out = moge.infer(x[0], apply_mask=True, fov_x=HFOV)
            mrng = np.linalg.norm(out["points"].detach().float().cpu().numpy(), axis=-1)
            msk = out["mask"].detach().cpu().numpy().astype(bool)
            mrng = np.where(msk & np.isfinite(mrng), mrng, 0.0)          # kill MoGe nan/invalid
            mg = cv2.resize(mrng, (gh, gh), interpolation=cv2.INTER_NEAREST)
            feats.append(fmap.cpu()); gts.append(gt.astype(np.float32)); moges.append(mg.astype(np.float32))
    return feats, gts, moges


def train_head(head, feats, gts, spatial, ep):
    head = head.to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), 1e-3, weight_decay=1e-4)
    D = feats[0].shape[0]
    if spatial:
        Xs = torch.stack(feats).to(DEVICE)                               # (N,D,gh,gw)
        Ys = torch.from_numpy(np.stack([np.log(np.clip(g, 1e-3, None)) for g in gts])).float().to(DEVICE)
        Ms = (Xs.new_tensor(np.stack(gts)) > 1e-3)
        for _ in range(ep):
            opt.zero_grad(); pr = head(Xs)
            loss = (F.l1_loss(pr[Ms], Ys[Ms])); loss.backward(); opt.step()
    else:
        X = torch.cat([f.permute(1, 2, 0).reshape(-1, D) for f in feats]).to(DEVICE)
        Y = torch.from_numpy(np.concatenate([np.log(np.clip(g.reshape(-1), 1e-3, None)) for g in gts])).float().to(DEVICE)
        M = torch.from_numpy(np.concatenate([g.reshape(-1) > 1e-3 for g in gts])).to(DEVICE)
        X, Y = X[M], Y[M]
        for _ in range(ep):
            opt.zero_grad(); loss = F.l1_loss(head(X), Y); loss.backward(); opt.step()
    return head


@torch.no_grad()
def eval_head(head, feats, gts, spatial):
    P_, G_ = [], []
    D = feats[0].shape[0]
    for f, g in zip(feats, gts):
        if spatial:
            pr = torch.exp(head(f[None].to(DEVICE))[0]).cpu().numpy()
        else:
            pr = torch.exp(head(f.permute(1, 2, 0).reshape(-1, D).to(DEVICE))).cpu().numpy().reshape(g.shape)
        P_.append(pr.reshape(-1)); G_.append(g.reshape(-1))
    return metrics(np.concatenate(P_), np.concatenate(G_))


if __name__ == "__main__":
    frozen = PanoEncoder(model_id=P.MODEL, lora_rank=0).to(DEVICE).eval()
    moge = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl").to(DEVICE).eval()
    P.configure("stanford2d3d"); P.TILE = TILE; P.enc_patch = frozen.patch
    geom = PF.T.build_geometry(frozen, HFOV, (-45.0, 0.0, 45.0))
    s2d = data.list_erps("stanford2d3d")

    def area(f):
        return f.split("extracted_data/")[1].split("/")[0]
    a1 = [f for f in s2d if area(f) == "area_1"]
    trf, tef = a1[:NTR], a1[NTR:NTR + NTE]
    print(f"axis1/2 probe: train {len(trf)} / test {len(tef)} area_1 panos, tiles/pano={len(geom['specs'])}", flush=True)

    ftr, gtr, _ = collect(frozen, moge, trf, geom)
    fte, gte, mte = collect(frozen, moge, tef, geom)
    D = ftr[0].shape[0]

    # MoGe reference on the same grid/metric
    mg_flat = np.concatenate([m.reshape(-1) for m in mte]); gt_flat = np.concatenate([g.reshape(-1) for g in gte])
    mm = (gt_flat > 1e-3) & (mg_flat > 1e-3) & np.isfinite(mg_flat)
    moge_absrel = float(np.mean(np.abs(mg_flat[mm] - gt_flat[mm]) / gt_flat[mm]))
    s = np.median(gt_flat[mm] / mg_flat[mm]); moge_al = float(np.mean(np.abs(mg_flat[mm] * s - gt_flat[mm]) / gt_flat[mm]))
    print(f"(MoGe valid-cell frac = {mm.mean():.3f})", flush=True)

    print(f"\n{'depth head on FROZEN DINOv3':30s}{'AbsRel':>9}{'AbsRel-al':>11}{'d<1.25':>9}", flush=True)
    res = {}
    for name, mk, sp in [("linear", lambda: nn.Linear(D, 1), False),
                         ("MLP", lambda: MLP(D), False),
                         ("conv (mini-decoder)", lambda: ConvHead(D), True)]:
        h = mk()
        if name == "linear":
            class Lin(nn.Module):
                def __init__(s2): super().__init__(); s2.l = nn.Linear(D, 1)
                def forward(s2, x): return s2.l(x)[:, 0]
            h = Lin()
        h = train_head(h, ftr, gtr, sp, EP)
        ar, d1, aral, d1al = eval_head(h, fte, gte, sp)
        res[name] = ar
        print(f"{name:30s}{ar:>9.3f}{aral:>11.3f}{d1:>9.3f}", flush=True)
    print(f"{'MoGe-2 (reference)':30s}{moge_absrel:>9.3f}{moge_al:>11.3f}{'-':>9}", flush=True)

    lin, best_head = res["linear"], min(res["MLP"], res["conv (mini-decoder)"])
    gap = lin - moge_absrel
    closed = (lin - best_head) / gap if gap > 1e-6 else 0.0
    print(f"\nlinear AbsRel {lin:.3f} -> best frozen head {best_head:.3f} -> MoGe {moge_absrel:.3f}", flush=True)
    print(f"strong head closes {closed*100:.0f}% of the linear->MoGe gap", flush=True)
    if closed > 0.6:
        v = ("AXIS-2 dominates — a strong decoder on FROZEN DINOv3 recovers most of MoGe's depth. F HAS the "
             "geometry; the linear probe was the bottleneck. => encoder-side injection (physical prior / "
             "parallax) has little room for depth; the honest lever is a stronger decoder, not encoder-SSL.")
    elif closed < 0.3:
        v = ("AXIS-1 dominates — even a strong frozen-DINOv3 decoder stays far below MoGe. F genuinely LACKS "
             "the geometry => encoder-side injection has real room. Proceed to the physical-prior I(T;Y|F) "
             "probe (metric-scale / shape-from-shading).")
    else:
        v = (f"MIXED ({closed*100:.0f}% closed) — partial extractability gap and partial missing-info. Both a "
             "stronger decoder AND an encoder injection could help; measure the physical-prior I(T;Y|F) next.")
    print(f"\nVERDICT: {v}", flush=True)
