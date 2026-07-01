"""Pointmap estimation on the unified PIFu field — the committed DUSt3R-style task.

E2P is parallax-free (one optical center) -> each ERP cell has a KNOWN ray; the CAMERA-FRAME pointmap
is depth × ray (canonical, gravity-aligned pano frame). World-frame global_xyz is ill-posed from a
single pano (needs pose), so we use the camera-frame pointmap = (per-pano median-normalized depth) ×
ray as GT, and regress it directly (3-channel head) from the PIFu field. Metric = 3D L2 in units of
median depth, + δ thresholds. (Depth itself is already validated |Δlog| .154; pointmap = depth×ray.)

Run: OPENCV_IO_ENABLE_OPENEXR=1 CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n pano python scripts/pointmap_test.py
"""
from __future__ import annotations

import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data  # noqa: E402
import anyres_e2p as a2p  # noqa: E402
import probe_seg_dinov3 as P  # noqa: E402
import multitask_eval as M  # noqa: E402
import pifu_alltasks as PF  # noqa: E402
from encoder import PanoEncoder  # noqa: E402

DEVICE = "cuda"
SEED = 0
MODEL = os.environ.get("MODEL", "facebook/dinov3-vitb16-pretrain-lvd1689m")
ERP_W, ERP_H = 2048, 1024
HQ, WQ = PF.HQ, PF.WQ
TR = int(os.environ.get("TR", 50))
VA = int(os.environ.get("VA", 20))
EPOCHS = int(os.environ.get("EPOCHS", 25))


def make_ray():
    lat = np.pi / 2 - (np.arange(HQ) + 0.5) / HQ * np.pi
    lon = (np.arange(WQ) + 0.5) / WQ * 2 * np.pi - np.pi
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    return np.stack([np.cos(LAT) * np.sin(LON), np.sin(LAT), np.cos(LAT) * np.cos(LON)], -1).astype(np.float32)


def load_pointmap(f, ray):
    dn, dval = M.load_depth(f)                                            # median-normalized depth (512,1024)
    dn = cv2.resize(dn, (WQ, HQ), interpolation=cv2.INTER_NEAREST)
    v = cv2.resize(dval.astype(np.uint8), (WQ, HQ), interpolation=cv2.INTER_NEAREST) > 0
    return (dn[..., None] * ray).astype(np.float32), v                   # camera-frame point = depth×ray


def main():
    torch.manual_seed(SEED)
    P.configure("stanford2d3d"); P.TILE = PF.TILE
    enc = PanoEncoder(model_id=MODEL, lora_rank=0).to(DEVICE).eval(); P.enc_patch = enc.patch
    plan = a2p.plan_tiles("full_sphere", PF.FOV, PF.FOV, 0.25)
    per_tile, anycov, pw = PF.precompute(plan, PF.TILE // enc.patch)
    cov = torch.from_numpy(anycov.reshape(HQ, WQ)); D = enc.dim; ray = make_ray()
    files = data.list_erps("stanford2d3d")
    def area(f): return f.split("extracted_data/")[1].split("/")[0]
    tr = [f for f in files if "5" not in area(f)][:TR]; va = [f for f in files if "5" in area(f)][:VA]
    print(f"pointmap (camera-frame depth×ray) | enc={MODEL.split('/')[-1]} field {HQ}×{WQ} tr={len(tr)} va={len(va)} ep={EPOCHS}", flush=True)

    def cache(fl):
        out = []
        for f in fl:
            erp = np.array(Image.open(f).convert("RGB").resize((ERP_W, ERP_H), Image.BILINEAR))
            pf, _ = PF.build_fields(enc, erp, plan, per_tile, pw)
            pt, v = load_pointmap(f, ray)
            out.append((pf, torch.from_numpy(pt), torch.from_numpy(v)))
        return out
    ctr, cva = cache(tr), cache(va)

    torch.manual_seed(SEED); head = PF.MLP(D, 3).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), 1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(SEED)
    for _ in range(EPOCHS):
        for i in torch.randperm(len(ctr), generator=g).tolist():
            pf, pt, v = ctr[i]; m = (v & cov).to(DEVICE)
            if not m.any():
                continue
            x = torch.from_numpy(pf).permute(2, 0, 1)[None].float().to(DEVICE)
            ls = F.l1_loss(head(x)[0].permute(1, 2, 0)[m], pt.to(DEVICE)[m])
            opt.zero_grad(); ls.backward(); opt.step()

    head.eval(); errs = []
    with torch.no_grad():
        for pf, pt, v in cva:
            m = (v & cov)
            x = torch.from_numpy(pf).permute(2, 0, 1)[None].float().to(DEVICE)
            pred = head(x)[0].permute(1, 2, 0).cpu()
            errs.append(torch.norm(pred[m] - pt[m], dim=-1))
    e = torch.cat(errs)
    print(f"\n  3D point error (unit = median depth):", flush=True)
    print(f"    mean={e.mean():.3f}  median={e.median():.3f}", flush=True)
    print(f"    δ<0.10={(e<0.10).float().mean():.3f}  δ<0.25={(e<0.25).float().mean():.3f}  δ<0.50={(e<0.50).float().mean():.3f}", flush=True)
    print(f"\n=== pointmap estimation: median 3D err {e.median():.3f}× median-depth, "
          f"{(e<0.25).float().mean():.0%} of cells within 0.25 ===", flush=True)


if __name__ == "__main__":
    main()
