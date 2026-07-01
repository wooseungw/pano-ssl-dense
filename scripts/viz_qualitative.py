"""Qualitative seg prediction maps on ERP for a few area5 val panos.
Stage 1: RGB | GT | Ours(e2p, frozen DINOv3 + linear). SphereUFormer/HEAL-SWIN appended later.
Run: CUDA_VISIBLE_DEVICES=1 conda run -n pano python scripts/viz_qualitative.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "scripts")
import probe_seg_dinov3 as P  # noqa: E402
import tiling_compare as TC  # noqa: E402
import data  # noqa: E402
from encoder import PanoEncoder  # noqa: E402

DEVICE = P.DEVICE
OUT = "docs/figures/viz_qualitative"; os.makedirs(OUT, exist_ok=True)
P.configure("stanford2d3d"); P.TILE = TC.TILE; P.WORK_HW = (TC.IH, TC.IW)
NC, HS, WS = P.N_CLASS, TC.HS, TC.WS
CMAP = (np.array([matplotlib.colormaps["tab20"](i % 20)[:3] for i in range(NC)]) * 255).astype(np.uint8)
CMAP[P.IGNORE] = [25, 25, 25]


def colorize(lab):
    return CMAP[np.clip(lab.reshape(HS, WS), 0, NC - 1)]


def area(f):
    return f.split("extracted_data/")[1].split("/")[0]


enc = PanoEncoder(model_id=P.MODEL, lora_rank=0).to(DEVICE).eval(); P.enc_patch = enc.patch
files = data.list_erps("stanford2d3d")
tr = [P.load_rgb_label(f) for f in [g for g in files if area(g) in ("area_1", "area_2", "area_3", "area_6")][:60]]
plan = TC.methods()["e2p_full65"]
clf = TC.train_head(enc, tr, plan)
cells = [TC.tile_cells(y, p, h) for (y, p, h) in plan]


@torch.no_grad()
def e2p_pred(rgb):
    cid_a, r_a, pr_a = [], [], []
    for (yaw, pitch, hfov), (cid, r) in zip(plan, cells):
        feat = TC.tile_feat(enc, rgb, yaw, pitch, hfov); d, gh, gw = feat.shape
        p32 = clf(feat.reshape(d, -1).t().float()).argmax(1).reshape(gh, gw).cpu()
        pp = F.interpolate(p32[None, None].float(), size=(TC.SR, TC.SR), mode="nearest")[0, 0].long().numpy().reshape(-1)
        cid_a.append(cid); r_a.append(r); pr_a.append(pp)
    cid_a = np.concatenate(cid_a); r_a = np.concatenate(r_a); pr_a = np.concatenate(pr_a)
    pred = np.zeros(HS * WS, np.int64); order = np.argsort(-r_a)
    pred[cid_a[order]] = pr_a[order]
    return pred


va = [f for f in files if area(f) in ("area_5a", "area_5b")]
sel = va[:: max(1, len(va) // 4)][:4]
for k, f in enumerate(sel):
    rgb, gtlab = P.load_rgb_label(f)
    gt = P.label_to_grid(gtlab, HS, WS)
    pred = e2p_pred(rgb)
    rgb_s = np.array(Image.fromarray(rgb).resize((WS, HS), Image.BILINEAR))
    sep = np.full((4, WS, 3), 255, np.uint8)
    panel = np.concatenate([rgb_s, sep, colorize(gt), sep, colorize(pred)], axis=0)
    Image.fromarray(panel).save(f"{OUT}/pano{k}_rgb_gt_ours.png")
    print(f"saved pano{k}: {os.path.basename(f)}", flush=True)
print("done", OUT)
