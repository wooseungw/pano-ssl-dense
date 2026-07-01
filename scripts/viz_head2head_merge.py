"""Unified head-to-head qualitative figure: RGB | GT | Ours(e2p) | SphereUFormer, same pano, same colors.
Loads SphereUFormer ERP preds from suf_pano*.npz (saved by sphere_uformer/src/viz_seg_erp.py),
matches the same pano by filename, runs our e2p, and maps sphere-label -> our-label by GT correspondence.
Run: CUDA_VISIBLE_DEVICES=1 conda run -n pano python scripts/viz_head2head_merge.py
"""
import glob
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

OUT = "docs/figures/viz_qualitative"
P.configure("stanford2d3d"); P.TILE = TC.TILE; P.WORK_HW = (TC.IH, TC.IW)
NC, HS, WS = P.N_CLASS, TC.HS, TC.WS
CMAP = (np.array([matplotlib.colormaps["tab20"](i % 20)[:3] for i in range(NC)]) * 255).astype(np.uint8)
CMAP[P.IGNORE] = [25, 25, 25]


def colorize(lab):
    return CMAP[np.clip(lab.reshape(HS, WS), 0, NC - 1)]


def area(f):
    return f.split("extracted_data/")[1].split("/")[0]


enc = PanoEncoder(model_id=P.MODEL, lora_rank=0).to(P.DEVICE).eval(); P.enc_patch = enc.patch
files = data.list_erps("stanford2d3d")
tr = [P.load_rgb_label(f) for f in [g for g in files if area(g) in ("area_1", "area_2", "area_3", "area_6")][:60]]
plan = TC.methods()["e2p_full65"]; clf = TC.train_head(enc, tr, plan)
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
    pred = np.zeros(HS * WS, np.int64); order = np.argsort(-r_a); pred[cid_a[order]] = pr_a[order]
    return pred.reshape(HS, WS)


sep = np.full((4, WS, 3), 255, np.uint8)
for npz in sorted(glob.glob(f"{OUT}/suf_pano*.npz")):
    k = npz.split("suf_pano")[1].split(".")[0]
    d = np.load(npz, allow_pickle=True)
    suf_gt, suf_pred, fpath = d["gt"], d["pred"], str(d["fpath"])
    match = [f for f in files if os.path.basename(f) == os.path.basename(fpath)]
    if not match:
        print("no match for", os.path.basename(fpath), flush=True); continue
    rgb, gtlab = P.load_rgb_label(match[0])
    our_gt = P.label_to_grid(gtlab, HS, WS)
    our_pred = e2p_pred(rgb)
    mapping = np.zeros(NC, int)                       # sphere label -> our label, by GT overlap
    for s in range(1, NC):
        m = suf_gt == s
        if m.sum() > 0:
            vals = our_gt[m]; vals = vals[vals > 0]
            if len(vals):
                mapping[s] = np.bincount(vals, minlength=NC).argmax()
    suf_pred_mapped = mapping[np.clip(suf_pred, 0, NC - 1)]
    rgb_s = np.array(Image.fromarray(rgb).resize((WS, HS), Image.BILINEAR))
    panel = np.concatenate([rgb_s, sep, colorize(our_gt), sep, colorize(our_pred), sep, colorize(suf_pred_mapped)], 0)
    Image.fromarray(panel).save(f"{OUT}/head2head_pano{k}.png")
    print(f"saved head2head_pano{k}  ({os.path.basename(fpath)[:40]})", flush=True)
print("done", flush=True)
