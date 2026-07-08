"""Visualize the SEG eval INTEGRATION (stitch): per-tile predictions -> overlap-averaged ERP.

Reuses the exact seg_s2d3d_bench machinery (render_pano/encode/coord_map/predict_erp) with the
FROZEN encoder + its trained seghead (runs/..._seg_s2d3d_frozen_f1, mIoU 57.7), so the stitched
panel is identical to the real benchmark. Shows, for one held-out area-5 pano:
  input ERP -> a few INDIVIDUAL tile predictions scattered onto the ERP (partial coverage = 패치별)
  -> overlap count (#tiles voting per cell) -> STITCHED integrated pred (통합) -> GT.

Run: CUDA_VISIBLE_DEVICES=0 conda run -n pano python scripts/viz_stitch_demo.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("ENC_ADAPTER", "")          # frozen encoder (no adapter) — must precede bench import
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data                                          # noqa: E402
import probe_seg_dinov3 as P                         # noqa: E402
import runlog                                        # noqa: E402
import seg_s2d3d_bench as B                          # noqa: E402  (render_pano/encode/coord_map/predict_erp/SegHead)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEGHEAD = os.path.join(ROOT, "runs", "0704_0524_seg_s2d3d_frozen_f1", "weights", "seghead.pt")
DEVICE = P.DEVICE


def scatter_tile_pred(head, feat_ti, cid):
    """One tile's per-pixel argmax scattered onto the (SH,SW) stitch grid; -1 where the tile
    does not cover. Illustrates a single tile's ERP footprint ('패치별')."""
    with torch.no_grad():
        lg = head(feat_ti[None].float().to(DEVICE))                       # (1,C,128,128)
        lg = torch.nn.functional.interpolate(lg, (B.TILE_OUT, B.TILE_OUT),
                                             mode="bilinear", align_corners=False)[0]
    am = lg.argmax(0).reshape(-1).cpu()                                   # (TILE_OUT^2,)
    grid = torch.full((B.SH * B.SW,), -1, dtype=torch.long)
    grid[cid] = am                                                        # last-wins scatter (viz only)
    return grid.reshape(B.SH, B.SW).numpy()


def main():
    P.configure("stanford2d3d"); P.TILE = B.TILE; P.WORK_HW = (B.EH, B.EW)
    enc = B.build_encoder(); P.enc_patch = enc.patch
    B.PLAN = P.a2p.plan_tiles("full_sphere", B.HFOV, B.HFOV, 0.25)

    head = B.SegHead(enc.dim, P.N_CLASS).to(DEVICE)
    head.load_state_dict(torch.load(SEGHEAD, map_location="cpu")); head.eval()

    files = data.list_erps("stanford2d3d")
    va = [f for f in files if B.area_num(f) in B.FOLD_TEST[1]]
    f = va[0]                                                             # same scene as the frozen run's s0
    print(f"pano={f}\ntiles/pano={len(B.PLAN)} stitch={B.SH}x{B.SW} eval={B.EH}x{B.EW}", flush=True)

    tiles, _ = B.render_pano(f)
    feat = B.encode(enc, tiles)                                           # (T,768,32,32)
    cids = [B.coord_map(tp.yaw_deg, tp.pitch_deg) for tp in B.PLAN]

    # official stitched (integrated) prediction — identical to the benchmark
    pred, cov = B.predict_erp(head, feat, cids)                          # (EH,EW), coverage frac
    gt_full = P.load_rgb_label(f)[1]
    gt = np.array(Image.fromarray(gt_full.astype(np.uint8)).resize((B.EW, B.EH), Image.NEAREST)).astype(np.int64)

    # overlap: number of DISTINCT tiles covering each stitch cell (intuitive '겹침')
    cnt = torch.zeros(B.SH * B.SW)
    for cid in cids:
        cnt[cid.unique()] += 1.0
    count = cnt.reshape(B.SH, B.SW).numpy()

    # pick 3 equator tiles at spread yaws to show distinct, partially-overlapping footprints
    eq = [i for i, tp in enumerate(B.PLAN) if abs(tp.pitch_deg) < 1e-3]
    picks = [eq[0], eq[len(eq) // 3], eq[2 * len(eq) // 3]] if len(eq) >= 3 else list(range(3))
    tile_scatters = [(B.PLAN[i], scatter_tile_pred(head, feat[i], cids[i])) for i in picks]

    # ---- compose figure (shared integration layout: runlog.save_integration_figure) ----
    pal = runlog.seg_palette(P.N_CLASS)
    rgb = np.array(Image.open(f).convert("RGB").resize((B.EW, B.EH), Image.BILINEAR)).astype(np.float32) / 255.0

    panels = [("input ERP (equirectangular panorama)", rgb, "rgb")]
    for tp, sc in tile_scatters:
        panels.append((f"single tile only  (yaw={tp.yaw_deg:.0f}°, pitch={tp.pitch_deg:.0f}°)  "
                       f"-> its ERP footprint [per-tile / patch-wise]", runlog.colorize(sc, pal), "field"))
    panels.append((f"overlap  (# distinct tiles covering each cell, max={int(count.max())})", count, "count"))
    panels.append((f"STITCHED = INTEGRATED  (coverage-normalized logit mean -> argmax)  "
                   f"mIoU=57.7, coverage={cov*100:.1f}%", runlog.colorize(pred, pal), "field"))
    panels.append(("GT (13-class)", runlog.colorize(gt, pal), "field"))

    run = runlog.create_run("stitch_demo", {
        "purpose": "visualize seg eval integration (stitch)", "encoder": "frozen DINOv3",
        "seghead": SEGHEAD, "pano": f, "tiles": len(B.PLAN), "picks": picks,
        "stitch_hw": [B.SH, B.SW], "eval_hw": [B.EH, B.EW], "mIoU_run": 0.5768})
    out = runlog.save_integration_figure(
        run, "s2d3d", 0, panels, fname="s2d3d_stitch_integration.png",
        suptitle="SEG eval INTEGRATION: per-tile predictions -> overlap-averaged ERP "
                 "(Stanford2D3D area-5, frozen DINOv3)")
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
