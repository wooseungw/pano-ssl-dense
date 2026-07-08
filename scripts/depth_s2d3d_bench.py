"""Metric 360 depth benchmark on Stanford2D3D — places our frozen-DINOv3 + E2P + conv head
on the published pano-depth scale (UniFuse / SGFormer / Elite360D / PanoFormer / OmniFusion).

Protocol (SOTA_BENCHMARK_PLAN §B): official fold (fold1 = areas 1-4,6 train / area5 test),
512x1024 eval, METRIC meters — GT depth.png is uint16 at 1/512 m per unit, valid where
0<raw<65535, evaluated up to a DEPTH_CAP (~10 m) with the poles falling out via the valid mask.
No median alignment on the headline (metric board); a per-image median-aligned delta1 is reported
as a scale-invariant secondary for comparability with depth_headtohead / SphereUFormer.

FULL-SPHERE E2P tiles -> FROZEN encoder -> conv head (DenseHead, C=1 log-depth) -> per-tile
log-depth -> coverage-mean STITCH to the ERP grid -> exp -> metric depth. Encoder is FROZEN
(a loaded SSL adapter is FIXED, not trained here); the ~2.4M head is the only trained module.

Metrics (dataset-aggregated over valid+covered pixels): AbsRel, SqRel, RMSE, RMSE(log),
delta<1.25^{1,2,3} (all metric), + SI-delta1 (per-image median-aligned).

Run: ENC_ADAPTER= FOLD=1 EPOCHS=20 CUDA_VISIBLE_DEVICES=0 python scripts/depth_s2d3d_bench.py
"""
from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench_common as B  # noqa: E402
import data  # noqa: E402
import runlog  # noqa: E402

DEPTH_SCALE = 512.0                                             # depth.png unit = 1/512 m
DEPTH_CAP = float(os.environ.get("DEPTH_CAP", 10.0))           # eval cap (m); poles fall out via valid
MIN_DEPTH = float(os.environ.get("MIN_DEPTH", 0.1))            # clamp predictions to a sane floor (m)

# Published Stanford2D3D depth SOTA (512x1024, METRIC) — web-scoped from SOTA_BENCHMARK_PLAN §B
# (Cross360 Table I). UNVERIFIED vs primary sources (the plan flags a Cross360-table discrepancy):
# reconfirm each row against its own paper before any published comparison.
# (method, AbsRel, RMSE, delta1%)  lower AbsRel/RMSE better, higher delta1 better.
SOTA = [("SGFormer (2024)", 0.104, 0.341, 90.0),
        ("UniFuse (2021)", 0.112, 0.356, 87.1),
        ("PanoFormer (2022)", 0.112, 0.395, 88.7),
        ("OmniFusion (2022)", 0.115, 0.381, 86.7),
        ("Elite360D (2024)", 0.118, 0.376, 88.7),
        ("EGFormer (2023)", 0.153, 0.497, 81.9)]

STAT_KEYS = ("abs", "sq", "se", "sle", "d1", "d2", "d3", "d1si", "n")


def load_depth_m(f: str, hw: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """ERP depth.png -> (depth_m, valid) at hw=(H,W). Metric meters; valid excludes 0/65535/>CAP."""
    h, w = hw
    raw = np.array(Image.open(data.s2d3d_gt_path(f, "depth")).resize((w, h), Image.NEAREST)).astype(np.float32)
    d_m = raw / DEPTH_SCALE
    valid = (raw > 0) & (raw < 65535) & (d_m <= DEPTH_CAP)
    return d_m, valid.astype(np.float32)


def depth_pixel_stats(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> Dict[str, float]:
    """Per-pixel error SUMS over valid pixels (dataset-aggregatable; sum across panos, then finalize).
    SI-delta1 aligns predictions to GT by this image's median ratio (scale-invariant secondary)."""
    p = np.clip(pred[valid], MIN_DEPTH, DEPTH_CAP)
    g = np.clip(gt[valid], MIN_DEPTH, None)
    if p.size == 0:
        return {k: 0.0 for k in STAT_KEYS}
    thr = np.maximum(p / g, g / p)
    ps = p * (np.median(g) / max(np.median(p), 1e-6))
    return dict(abs=float(np.sum(np.abs(p - g) / g)), sq=float(np.sum((p - g) ** 2 / g)),
                se=float(np.sum((p - g) ** 2)), sle=float(np.sum((np.log(p) - np.log(g)) ** 2)),
                d1=float(np.sum(thr < 1.25)), d2=float(np.sum(thr < 1.25 ** 2)),
                d3=float(np.sum(thr < 1.25 ** 3)),
                d1si=float(np.sum(np.maximum(ps / g, g / ps) < 1.25)), n=float(p.size))


def finalize_depth(s: Dict[str, float]) -> Dict[str, float]:
    """Aggregated pixel sums -> mean metrics."""
    n = max(s["n"], 1.0)
    return dict(AbsRel=s["abs"] / n, SqRel=s["sq"] / n, RMSE=float(np.sqrt(s["se"] / n)),
                RMSE_log=float(np.sqrt(s["sle"] / n)), d1=s["d1"] / n, d2=s["d2"] / n,
                d3=s["d3"] / n, d1_SI=s["d1si"] / n)


def build_cache(enc, files: List[str], plan, want_full: bool) -> List:
    """Encode each pano's tiles once (frozen). Train items carry per-tile (log-depth, valid) at
    HEAD_OUT; val items carry the full-ERP metric GT + valid at (EH,EW) for stitched scoring."""
    cache = []
    for f in files:
        rgb = np.array(Image.open(f).convert("RGB").resize((B.EW, B.EH), Image.BILINEAR))
        feat = B.encode_erp(enc, rgb).half()                      # GPU render + encode
        d_m, valid = load_depth_m(f, (B.EH, B.EW))
        if want_full:
            cache.append((feat, torch.from_numpy(d_m), torch.from_numpy(valid > 0.5)))
        else:
            dg_all = B.warp_gt_gpu(d_m[:, :, None], "nearest").numpy()[:, :, :, 0]     # (T,HO,HO) GPU warp
            mv_all = B.warp_gt_gpu(valid[:, :, None], "nearest").numpy()[:, :, :, 0] > 0.5
            tiles = []
            for ti in range(len(plan)):
                dg = dg_all[ti]
                dlog = np.log(np.clip(dg, MIN_DEPTH, None))
                tiles.append((torch.from_numpy(dlog).float(), torch.from_numpy(mv_all[ti] & (dg > MIN_DEPTH))))
            cache.append((feat, tiles))
    return cache


def train_head(head: nn.Module, ctr: List) -> None:
    """L1 on log-depth over valid tile pixels (metric: head learns absolute scale from train set)."""
    opt = torch.optim.AdamW(head.parameters(), 1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(B.SEED)
    for ep in range(B.EPOCHS):
        head.train()
        tot, nb = 0.0, 0
        for i in torch.randperm(len(ctr), generator=g).tolist():
            feat, tiles = ctr[i]
            opt.zero_grad()
            for s in range(0, feat.shape[0], B.CHUNK):
                sub = tiles[s:s + B.CHUNK]
                y = torch.stack([t[0] for t in sub]).to(B.DEVICE)
                m = torch.stack([t[1] for t in sub]).to(B.DEVICE)
                if not m.any():
                    continue
                # interpolate head output to the GT tile res (patch-size agnostic: a non-patch-16
                # MODEL backbone yields a different feature grid -> different head-output size).
                out = F.interpolate(head(feat[s:s + B.CHUNK].float().to(B.DEVICE)),
                                    (B.HEAD_OUT, B.HEAD_OUT), mode="bilinear", align_corners=False)
                p = out[:, 0]
                loss = F.l1_loss(p[m], y[m]) * (len(sub) / feat.shape[0])
                loss.backward()
                tot += loss.item()
                nb += 1
            opt.step()
        if (ep + 1) % 5 == 0 or ep == B.EPOCHS - 1:
            print(f"  ep{ep+1}/{B.EPOCHS} loss={tot/max(nb,1):.4f}", flush=True)


@torch.no_grad()
def predict_depth(head: nn.Module, feat: torch.Tensor, cids) -> Tuple[np.ndarray, float, np.ndarray]:
    """Stitched metric depth (EH,EW) meters + stitch coverage fraction + (EH,EW) covered mask."""
    field, cov, covered = B.stitch_field(head, feat, cids, 1)        # (1,EH,EW) log-depth
    return field[0].exp().clamp(MIN_DEPTH, DEPTH_CAP).numpy(), cov, covered


def emit_integration(run: str, head: nn.Module, cva: List, cids, plan, va: List[str]) -> None:
    """Write the depth stitch-integration figures (3 spread val samples) into run/viz/:
    input ERP -> single-tile footprints -> overlap -> STITCHED@eval (metric) -> GT, GT alongside."""
    cnt = torch.zeros(B.SH * B.SW)                                    # overlap: # tiles covering each stitch cell
    for cid in cids:
        cnt[cid.unique()] += 1.0
    count = cnt.reshape(B.SH, B.SW).numpy()
    eq = [j for j, tp in enumerate(plan) if abs(tp.pitch_deg) < 1e-3]
    tpick = ([eq[0], eq[len(eq) // 3], eq[2 * len(eq) // 3]] if len(eq) >= 3 else list(range(min(3, len(plan)))))
    lg_min, lg_cap = float(np.log(MIN_DEPTH)), float(np.log(DEPTH_CAP))
    for i in runlog.spread_indices(len(cva), 3):                      # spread picks (not first-3 near-dups)
        feat, gt, valid = cva[i]
        pred, cov, covered = predict_depth(head, feat, cids)
        rgb = np.array(Image.open(va[i]).convert("RGB").resize((B.EW, B.EH), Image.BILINEAR)).astype(np.float32) / 255.0
        gtn = gt.numpy()
        v = valid.numpy() & covered
        gt_log = np.log(np.clip(gtn, MIN_DEPTH, None))
        pred_log = np.log(np.clip(pred, MIN_DEPTH, None))
        lo, hi = (gt_log[v].min(), gt_log[v].max()) if v.any() else (lg_min, lg_cap)
        runlog.save_depth_sample(run, "s2d3d_depth", i, rgb, gt_log, {B.TAG: pred_log}, v, scale=1)

        def col(field_log, mask):                                    # turbo on log-depth, GT-shared scale
            img = runlog._turbo((field_log - lo) / max(hi - lo, 1e-6))
            img[~mask] = 0.15
            return img
        panels = [("input ERP (equirectangular panorama)", rgb, "rgb")]
        for ti in tpick:                                             # a few single-tile footprints (per-tile)
            f1, _, cov1 = B.stitch_field(head, feat[ti:ti + 1], [cids[ti]], 1)
            panels.append((f"single tile only  (yaw={plan[ti].yaw_deg:.0f}°, pitch={plan[ti].pitch_deg:.0f}°) "
                           f" -> its ERP footprint [per-tile]", col(f1[0].numpy(), cov1), "field"))
        panels.append((f"overlap  (# distinct tiles covering each cell, max={int(count.max())})", count, "count"))
        r1 = finalize_depth(depth_pixel_stats(pred, gtn, v))
        panels.append((f"STITCHED = INTEGRATED  (coverage-mean log-depth -> exp)  "
                       f"AbsRel={r1['AbsRel']:.3f}, d1={r1['d1']*100:.1f}%, coverage={cov*100:.1f}%",
                       col(pred_log, v), "field"))
        panels.append(("GT depth (log, metric-valid pixels)", col(gt_log, v), "field"))
        runlog.save_integration_figure(
            run, "s2d3d_depth", i, panels,
            suptitle=f"DEPTH eval INTEGRATION: per-tile -> overlap-averaged ERP @ {B.EH}x{B.EW} "
                     f"(Stanford2D3D fold{B.FOLD}, {'frozen DINOv3' if not B.ADAPTER else B.TAG})")


def main() -> None:
    torch.manual_seed(B.SEED)
    enc = B.build_encoder()
    plan = B.build_plan()
    B.build_sample_grids(plan)                                    # precompute GPU render grids once
    cids = [B.coord_map(tp.yaw_deg, tp.pitch_deg) for tp in plan]
    files = data.list_erps("stanford2d3d")
    tr, va = B.split_files(files, B.FOLD)
    print(f"Depth-S2D3D fold{B.FOLD} enc={B.TAG} model={B.MODEL} tiles/pano={len(plan)} "
          f"tr={len(tr)} va={len(va)} eval={B.EH}x{B.EW} cap={DEPTH_CAP}m ep={B.EPOCHS}", flush=True)

    t0 = time.time()
    ctr = build_cache(enc, tr, plan, want_full=False)
    cva = build_cache(enc, va, plan, want_full=True)
    print(f"encoded {len(tr)}+{len(va)} panos ({time.time()-t0:.0f}s)", flush=True)

    torch.manual_seed(B.SEED)
    head = B.DenseHead(enc.dim, 1).to(B.DEVICE)
    train_head(head, ctr)

    head.eval()
    agg = {k: 0.0 for k in STAT_KEYS}
    cov_sum = 0.0
    for feat, gt, valid in cva:
        pred, cov, covered = predict_depth(head, feat, cids)
        cov_sum += cov
        v = valid.numpy() & covered & np.isfinite(pred)
        s = depth_pixel_stats(pred, gt.numpy(), v)
        for k in STAT_KEYS:
            agg[k] += s[k]

    r = finalize_depth(agg)
    coverage = cov_sum / max(len(cva), 1)
    head_M = sum(p.numel() for p in head.parameters()) / 1e6
    ours = (f"OURS (frozen DINOv3 + {head_M:.2f}M head)" if not B.ADAPTER
            else f"OURS (DINOv3 + SSL[{B.TAG}] + {head_M:.2f}M head)")

    print(f"\n=== Depth-S2D3D fold{B.FOLD} | {ours} | eval {B.EH}x{B.EW} metric<= {DEPTH_CAP}m ===", flush=True)
    print(f"  AbsRel={r['AbsRel']:.4f}  SqRel={r['SqRel']:.4f}  RMSE={r['RMSE']:.4f}  "
          f"RMSE_log={r['RMSE_log']:.4f}", flush=True)
    print(f"  d1={r['d1']*100:.1f}  d2={r['d2']*100:.1f}  d3={r['d3']*100:.1f}  "
          f"(SI-d1={r['d1_SI']*100:.1f})  | stitch coverage {coverage*100:.1f}%", flush=True)
    print(f"\n{'method':22s} {'AbsRel':>7} {'RMSE':>7} {'d1%':>6}", flush=True)
    print(f"{ours[:22]:22s} {r['AbsRel']:7.3f} {r['RMSE']:7.3f} {r['d1']*100:6.1f}", flush=True)
    for name, ar, rm, d1 in SOTA:
        print(f"{name:22s} {ar:7.3f} {rm:7.3f} {d1:6.1f}", flush=True)
    leak = "" if not B.ADAPTER else " · SSL row is TRANSDUCTIVE (adapter pool incl. test areas)"
    print(f"caveats: our eval fold{B.FOLD}, {B.EH}x{B.EW}, head-only on FROZEN features, {len(tr)}-pano "
          f"train / {len(va)} test; SOTA rows web-scoped (Cross360 Table I) — reconfirm vs primary "
          f"source; SOTA = full fine-tune.{leak}", flush=True)

    run = runlog.create_run(f"depth_s2d3d_{B.TAG}_f{B.FOLD}", {
        "benchmark": "Stanford2D3D metric 360 depth", "fold": B.FOLD, "encoder": B.ADAPTER or "frozen",
        "model": B.MODEL, "tag": B.TAG, "eval_hw": [B.EH, B.EW], "stitch_hw": [B.SH, B.SW],
        "depth_cap_m": DEPTH_CAP, "epochs": B.EPOCHS, "tr_panos": len(tr), "va_panos": len(va),
        "metrics": r, "stitch_coverage": coverage, "head_M": head_M, "transductive": bool(B.ADAPTER),
        "sota": [{"method": m, "AbsRel": a, "RMSE": rm, "delta1": d} for m, a, rm, d in SOTA],
        "protocol_caveats": "single fold, head-only on frozen features; SOTA web-scoped/unverified; "
        f"SSL rows transductive={bool(B.ADAPTER)}"})
    torch.save(head.state_dict(), os.path.join(run, "weights", "depthhead.pt"))

    emit_integration(run, head, cva, cids, plan, va)
    print(f"saved -> {run} (config + weights + viz + SOTA table)", flush=True)


if __name__ == "__main__":
    main()
