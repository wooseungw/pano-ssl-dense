"""Depth head-to-head: our frozen DINOv3 + e2p PUP depth decoder vs SphereUFormer (a1 0.739).
Same metric (a1/a2/a3 scale-invariant; mae/mre on depth/MAXD), area5-fold, our native ERP128x256 grid.
Depth is radial (per-ray) so e2p tiles assemble seam-consistently with no z->radial conversion.
Run: CUDA_VISIBLE_DEVICES=1 conda run -n pano python scripts/depth_headtohead.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "scripts")
import probe_seg_dinov3 as P  # noqa: E402
import tiling_compare as TC  # noqa: E402
import probe_normal as PN  # noqa: E402
import multitask_eval as M  # noqa: E402
import data  # noqa: E402
from encoder import PanoEncoder, normalize_tiles  # noqa: E402

DEVICE = P.DEVICE; MAXD = 5120.0
TILE, IH, IW, SR, HS, WS = TC.TILE, TC.IH, TC.IW, TC.SR, TC.HS, TC.WS
TRN = int(os.environ.get("TRN", 150)); EPOCHS = int(os.environ.get("EPOCHS", 12))


def _r1(a, h, w):
    return np.array(Image.fromarray(a.astype(np.float32)).resize((w, h), Image.NEAREST))


def load_depth_n(f):
    d = np.array(Image.open(data.s2d3d_gt_path(f, "depth")).resize((IW, IH), Image.NEAREST)).astype(np.float32)
    valid = (d > 0) & (d <= MAXD)
    return (np.clip(d, 0, MAXD) / MAXD), valid.astype(np.float32)


P.configure("stanford2d3d"); P.TILE = TILE; P.WORK_HW = (IH, IW)
plan = TC.methods()["e2p_full65"]
cells = [TC.tile_cells(y, p, h) for (y, p, h) in plan]
enc = PanoEncoder(model_id=P.MODEL, lora_rank=0).to(DEVICE).eval(); P.enc_patch = enc.patch


@torch.no_grad()
def cache(files):
    out = []
    for f in files:
        rgb = np.array(Image.open(f).convert("RGB").resize((IW, IH), Image.BILINEAR))
        dn, dval = load_depth_n(f)
        feats, tgts = [], []
        for (yaw, pitch, hfov) in plan:
            t = np.asarray(P.a2p.erp_to_pinhole_tile(rgb, yaw, pitch, hfov, TILE))
            feat = P.dense(enc, normalize_tiles(torch.from_numpy(t).float().permute(2, 0, 1)[None].to(DEVICE) / 255.0))[0]
            dg = PN.warp_to_grid(dn[:, :, None], yaw, pitch, hfov, SR, SR, 1)[:, :, 0]
            dv = PN.warp_to_grid(dval[:, :, None], yaw, pitch, hfov, SR, SR, 1)[:, :, 0] > 0.5
            feats.append(feat.half().cpu()); tgts.append((torch.from_numpy(dg).float(), torch.from_numpy(dv)))
        sgt = (torch.from_numpy(_r1(dn, HS, WS)).float(), torch.from_numpy(_r1(dval, HS, WS) > 0.5))
        out.append((feats, tgts, sgt))
    return out


def train_dec(ctr, dim):
    torch.manual_seed(0); dec = M.PUP(dim, 1).to(DEVICE)
    opt = torch.optim.AdamW(dec.parameters(), 1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(0)
    for _ in range(EPOCHS):
        for i in torch.randperm(len(ctr), generator=g).tolist():
            feats, tgts, _ = ctr[i]; opt.zero_grad()
            for s in range(0, len(feats), 8):
                fb = torch.stack([f.float() for f in feats[s:s + 8]]).to(DEVICE)
                pr = dec(fb)                                          # [B,1,SR,SR]
                loss = 0.0; n = 0
                for j, (dg, dv) in enumerate(tgts[s:s + 8]):
                    p = F.interpolate(pr[j:j + 1], size=(SR, SR), mode="bilinear", align_corners=False)[0, 0]
                    m = dv.to(DEVICE)
                    if m.any():
                        loss = loss + (p[m] - dg.to(DEVICE)[m]).abs().mean(); n += 1
                if n > 0:
                    (loss / n * fb.shape[0] / len(feats)).backward()
            opt.step()
    return dec


@torch.no_grad()
def eval_dep(dec, cva):
    se = sd = a1 = a2 = a3 = 0.0; sn = 0
    for feats, _, sgt in cva:
        num = np.zeros(HS * WS, np.float32); rbuf = np.full(HS * WS, np.inf)
        cid_a, r_a, v_a = [], [], []
        for f, (cid, r) in zip(feats, cells):
            o = dec(f.float()[None].to(DEVICE))[0, 0]
            v = F.interpolate(o[None, None], size=(SR, SR), mode="bilinear", align_corners=False)[0, 0].cpu().numpy().reshape(-1)
            cid_a.append(cid); r_a.append(r); v_a.append(v)
        cid_a = np.concatenate(cid_a); r_a = np.concatenate(r_a); v_a = np.concatenate(v_a)
        order = np.argsort(-r_a); num[cid_a[order]] = v_a[order]; rbuf[cid_a[order]] = r_a[order]
        cov = torch.from_numpy(rbuf < np.inf)
        pred = torch.from_numpy(num); gt = sgt[0].reshape(-1)
        val = sgt[1].reshape(-1) & cov & (gt > 1e-3)
        p = pred[val].clamp(min=1e-3); gg = gt[val].clamp(min=1e-3)
        thr = torch.max(gg / p, p / gg)
        a1 += (thr < 1.25).float().sum().item(); a2 += (thr < 1.25 ** 2).float().sum().item(); a3 += (thr < 1.25 ** 3).float().sum().item()
        se += (p - gg).abs().sum().item(); sd += ((p - gg).abs() / gg).sum().item(); sn += int(val.sum())
    return dict(mae=se / sn, mre=sd / sn, a1=a1 / sn, a2=a2 / sn, a3=a3 / sn)


def main():
    files = data.list_erps("stanford2d3d")
    def area(f): return f.split("extracted_data/")[1].split("/")[0]
    tr = [f for f in files if area(f) in ("area_1", "area_2", "area_3", "area_6")][:TRN]
    va = [f for f in files if area(f) in ("area_5a", "area_5b")]
    print(f"depth head-to-head | our e2p+PUP frozen DINOv3 | tr={len(tr)} va={len(va)} grid={HS}x{WS} norm=/{MAXD:.0f}", flush=True)
    ctr = cache(tr); cva = cache(va)
    dec = train_dec(ctr, enc.dim)
    r = eval_dep(dec, cva)
    print("\n=== OURS (e2p + PUP, frozen DINOv3), sphere(ERP 128x256) depth ===")
    print(f"  a1={r['a1']:.4f}  a2={r['a2']:.4f}  a3={r['a3']:.4f}  mae={r['mae']:.4f}  mre={r['mre']:.4f}")
    print(f"\n=== vs SphereUFormer rank6 (from-scratch, icosphere): a1=0.739  mae=0.376 ===")


if __name__ == "__main__":
    main()
