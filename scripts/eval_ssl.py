"""Frozen vs LoRA-SSL eval: does the overlap-SSL adapter lift the cheap single-tile
ERP prediction toward the multi-view ensemble ceiling, and cut cross-tile disagreement?

All metrics are on the ERP-stitched grid (each sphere cell scored once) so single and
blend share one basis — the headroom story:
  single = single-best (each ERP cell from its least-oblique covering tile)  [cheap, 1 pass/cell]
  blend  = overlap ENSEMBLE (mean feature over covering tiles)              [expensive ceiling]
  frozen headroom budget = blend - single  (DP@50 ~+0.12) = what SSL aims to close.
disagree = cross-tile prediction disagreement on overlap cells (SSL should cut it).

Per encoder we train its OWN linear head on its train tiles, then eval on held-out val.
Run: CUDA_VISIBLE_DEVICES=0 conda run -n pano python scripts/eval_ssl.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import probe_seg_dinov3 as P  # noqa: E402
import diag_seam as D  # noqa: E402
from encoder import PanoEncoder  # noqa: E402

CKPT = os.environ.get("ADAPTER", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "ckpt_ssl_lora"))
EVAL = [("densepass", 50.0), ("stanford2d3d", 65.0)]
DEVICE = P.DEVICE


def encoder_metrics(enc, cache_tr, cache_va, plan):
    """-> (single_best_miou, blend_miou, disagree) on the ERP-stitched val grid."""
    P.enc_patch = enc.patch
    head = D.head_on_tiles(enc, {"tr": cache_tr}, plan)
    sb, bl, gts, covs, dis = [], [], [], [], []
    for rgb, lab in cache_va:
        fsum, cov, best_pred, disagree, gt = D.scatter_pano(enc, rgb, lab, plan, head)
        m = cov >= 1
        blend = fsum[m] / torch.from_numpy(cov[m]).float()[:, None]
        with torch.no_grad():
            pbl = head(blend.to(DEVICE).float()).argmax(1).cpu().numpy()
        sb.append(best_pred[m]); bl.append(pbl); gts.append(gt[m])
        covs.append(cov[m]); dis.append(disagree[m])
    sb, bl, gts = np.concatenate(sb), np.concatenate(bl), np.concatenate(gts)
    covs, dis = np.concatenate(covs), np.concatenate(dis)
    seam = (covs >= 2) & (gts != P.IGNORE)
    single = P.miou_acc(torch.from_numpy(sb), torch.from_numpy(gts))[0]
    blend = P.miou_acc(torch.from_numpy(bl), torch.from_numpy(gts))[0]
    disagree = dis[seam].mean() if seam.sum() else 0.0
    return single, blend, disagree


def main():
    frozen = PanoEncoder(model_id=P.MODEL, lora_rank=0).to(DEVICE).eval()
    lora = PanoEncoder(model_id=P.MODEL, adapter_path=CKPT).to(DEVICE).eval()
    print(f"loaded frozen + LoRA adapter ({CKPT})\n", flush=True)
    print(f"{'dataset':13s} {'enc':7s} {'single':>7} {'blend':>7} {'disagree':>9}", flush=True)
    for ds, fov in EVAL:
        P.configure(ds); D.DATASET = ds; D.HFOV = fov
        plan = D.tile_plan()
        panos, _, train = P.grouped()
        cache_tr = [P.load_rgb_label(f) for g, f in panos if g in train]
        cache_va = [P.load_rgb_label(f) for g, f in panos if g not in train]
        if not cache_va:
            print(f"{ds:13s} skipped (no val panos on disk)\n", flush=True)
            continue
        res = {}
        for tag, enc in [("frozen", frozen), ("LoRA", lora)]:
            res[tag] = encoder_metrics(enc, cache_tr, cache_va, plan)
            s, b, d = res[tag]
            print(f"{ds:13s} {tag:7s} {s:7.3f} {b:7.3f} {d:9.3f}", flush=True)
        budget = res["frozen"][1] - res["frozen"][0]
        d_single = res["LoRA"][0] - res["frozen"][0]
        d_dis = res["LoRA"][2] - res["frozen"][2]
        closed = d_single / budget * 100 if budget > 1e-6 else float("nan")
        print(f"{ds:13s} {'Δ':7s} single={d_single:+.3f} (ceiling budget {budget:+.3f}, "
              f"closed {closed:.0f}%)  disagree={d_dis:+.3f} (want <0)\n", flush=True)


if __name__ == "__main__":
    main()
