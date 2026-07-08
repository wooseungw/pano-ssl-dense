"""Surface-normal benchmark on Stanford2D3D — evaluates our frozen-DINOv3 + E2P + conv head
against OTHER MODELS by swapping the encoder: frozen DINOv3 vs SSL LoRA adapters (ENC_ADAPTER)
vs other HF backbones (MODEL). Unlike depth/seg there is NO widely-agreed published pano-normal
leaderboard on Stanford2D3D, so the PRIMARY comparison here is cross-encoder (run once per model,
runlog aggregates); any published row must be web-verified before it goes in SOTA below.

Protocol mirrors depth_s2d3d_bench / seg_s2d3d_bench: official fold, FULL-SPHERE E2P tiles ->
FROZEN encoder -> conv head (DenseHead, C=3) trained with a cosine loss -> per-tile normals ->
coverage-mean STITCH -> renormalize. GT normals.png is a pano/world-frame unit-vector map (valid
where |n|>0.5); warp_to_grid resamples the vectors WITHOUT rotating them, so per-tile predictions
stay in the world frame and overlapping tiles can be averaged directly.

Metrics (dataset-aggregated over valid+covered pixels): mean & median angular error (deg),
and % of pixels within 11.25 / 22.5 / 30 deg (the NYUv2/FrameNet convention).

Run: ENC_ADAPTER= FOLD=1 EPOCHS=20 CUDA_VISIBLE_DEVICES=0 python scripts/normal_s2d3d_bench.py
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

NBINS = 720                                                      # 0.25-deg histogram bins for median
EDGES = np.linspace(0.0, 180.0, NBINS + 1)

# Published Stanford2D3D pano surface-normal SOTA: none reconfirmed. DO NOT fabricate angular-error
# rows — a plausible-looking invented number is worse than an empty table. Add (method, mean_deg,
# median_deg, pct_11.25) rows ONLY after web-verifying against the primary source.
SOTA: List[Tuple[str, float, float, float]] = []

STAT_KEYS = ("sum", "n", "c1125", "c225", "c30")


def load_normal(f: str, hw: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """ERP normals.png -> (unit_normal (H,W,3) world-frame, valid (H,W)) at hw=(H,W)."""
    h, w = hw
    n = np.array(Image.open(data.s2d3d_gt_path(f, "normal")).convert("RGB").resize((w, h), Image.NEAREST))
    v = n.astype(np.float32) / 255.0 * 2 - 1
    nrm = np.linalg.norm(v, axis=2, keepdims=True)
    valid = (nrm[:, :, 0] > 0.5).astype(np.float32)
    return v / np.clip(nrm, 1e-6, None), valid


def normal_pixel_stats(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> Dict[str, object]:
    """Angular-error sums/threshold-counts + a histogram (for median), over valid pixels.
    pred/gt are (H,W,3) unit normals; returns dataset-aggregatable pieces."""
    p = pred[valid]
    g = gt[valid]
    if p.shape[0] == 0:
        return dict(sum=0.0, n=0.0, c1125=0.0, c225=0.0, c30=0.0, hist=np.zeros(NBINS))
    cos = np.clip(np.sum(p * g, axis=1), -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    hist, _ = np.histogram(ang, bins=EDGES)
    return dict(sum=float(ang.sum()), n=float(ang.size), c1125=float(np.sum(ang < 11.25)),
                c225=float(np.sum(ang < 22.5)), c30=float(np.sum(ang < 30.0)), hist=hist)


def finalize_normal(agg: Dict[str, float], hist: np.ndarray) -> Dict[str, float]:
    """Aggregated angular stats -> mean/median error + within-threshold percentages."""
    n = max(agg["n"], 1.0)
    centers = (EDGES[:-1] + EDGES[1:]) / 2
    med_idx = int(np.searchsorted(np.cumsum(hist), n / 2.0))
    median = float(centers[min(med_idx, len(centers) - 1)])
    return dict(mean=agg["sum"] / n, median=median, pct_11=agg["c1125"] / n * 100,
                pct_22=agg["c225"] / n * 100, pct_30=agg["c30"] / n * 100)


def build_cache(enc, files: List[str], plan, want_full: bool) -> List:
    """Encode each pano's tiles once (frozen). Train items carry per-tile (unit-normal, valid) at
    HEAD_OUT; val items carry the full-ERP normal GT + valid at (EH,EW) for stitched scoring."""
    cache = []
    for f in files:
        rgb = np.array(Image.open(f).convert("RGB").resize((B.EW, B.EH), Image.BILINEAR))
        feat = B.encode_erp(enc, rgb).half()                      # GPU render + encode
        nrm, valid = load_normal(f, (B.EH, B.EW))
        if want_full:
            cache.append((feat, torch.from_numpy(nrm).float(), torch.from_numpy(valid > 0.5)))
        else:
            ng_all = B.warp_gt_gpu(nrm, "nearest").numpy()                            # (T,HO,HO,3) GPU warp
            mv_all = B.warp_gt_gpu(valid[:, :, None], "nearest").numpy()[:, :, :, 0] > 0.5
            tiles = []
            for ti in range(len(plan)):
                ng = ng_all[ti]
                ng = ng / np.clip(np.linalg.norm(ng, axis=2, keepdims=True), 1e-6, None)
                tiles.append((torch.from_numpy(ng).float(), torch.from_numpy(mv_all[ti])))
            cache.append((feat, tiles))
    return cache


def train_head(head: nn.Module, ctr: List) -> None:
    """Cosine loss (1 - cos) over valid tile pixels; head predicts world-frame normals."""
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
                y = torch.stack([t[0] for t in sub]).permute(0, 3, 1, 2).to(B.DEVICE)   # (b,3,h,w)
                m = torch.stack([t[1] for t in sub]).to(B.DEVICE)                        # (b,h,w)
                if not m.any():
                    continue
                # interpolate head output to the GT tile res (patch-size agnostic: a non-patch-16
                # MODEL backbone yields a different feature grid -> different head-output size).
                o = F.interpolate(head(feat[s:s + B.CHUNK].float().to(B.DEVICE)),
                                  (B.HEAD_OUT, B.HEAD_OUT), mode="bilinear", align_corners=False)
                out = F.normalize(o, dim=1)
                cos = (out * y).sum(1)                                                   # (b,h,w)
                loss = (1 - cos)[m].mean() * (len(sub) / feat.shape[0])
                loss.backward()
                tot += loss.item()
                nb += 1
            opt.step()
        if (ep + 1) % 5 == 0 or ep == B.EPOCHS - 1:
            print(f"  ep{ep+1}/{B.EPOCHS} loss={tot/max(nb,1):.4f}", flush=True)


@torch.no_grad()
def predict_normal(head: nn.Module, feat: torch.Tensor, cids) -> Tuple[np.ndarray, float, np.ndarray]:
    """Stitched unit normals (EH,EW,3) + stitch coverage fraction + (EH,EW) covered mask."""
    field, cov, covered = B.stitch_field(head, feat, cids, 3)        # (3,EH,EW)
    n = F.normalize(field.permute(1, 2, 0), dim=-1).numpy()          # renormalize averaged vectors
    return n, cov, covered


def emit_integration(run: str, head: nn.Module, cva: List, cids, plan, va: List[str]) -> None:
    """Write the normal stitch-integration figures (3 spread val samples) into run/viz/:
    input ERP -> single-tile footprints -> overlap -> STITCHED@eval (metric) -> GT, GT alongside."""
    cnt = torch.zeros(B.SH * B.SW)                                    # overlap: # tiles covering each stitch cell
    for cid in cids:
        cnt[cid.unique()] += 1.0
    count = cnt.reshape(B.SH, B.SW).numpy()
    eq = [j for j, tp in enumerate(plan) if abs(tp.pitch_deg) < 1e-3]
    tpick = ([eq[0], eq[len(eq) // 3], eq[2 * len(eq) // 3]] if len(eq) >= 3 else list(range(min(3, len(plan)))))

    def col(n_hwc, mask):                                             # unit normals -> (n+1)/2 RGB, gray where invalid
        img = np.clip((n_hwc + 1.0) / 2.0, 0.0, 1.0)
        img[~mask] = 0.15
        return img
    for i in runlog.spread_indices(len(cva), 3):                      # spread picks (not first-3 near-dups)
        feat, gt, valid = cva[i]
        pred, cov, covered = predict_normal(head, feat, cids)
        rgb = np.array(Image.open(va[i]).convert("RGB").resize((B.EW, B.EH), Image.BILINEAR)).astype(np.float32) / 255.0
        gtn = gt.numpy()
        v = valid.numpy() & covered
        runlog.save_normal_sample(run, "s2d3d_normal", i, rgb, gtn, {B.TAG: pred}, v, scale=1)

        panels = [("input ERP (equirectangular panorama)", rgb, "rgb")]
        for ti in tpick:                                             # a few single-tile footprints (per-tile)
            f1, _, cov1 = B.stitch_field(head, feat[ti:ti + 1], [cids[ti]], 3)
            n1 = F.normalize(f1.permute(1, 2, 0), dim=-1).numpy()
            panels.append((f"single tile only  (yaw={plan[ti].yaw_deg:.0f}°, pitch={plan[ti].pitch_deg:.0f}°) "
                           f" -> its ERP footprint [per-tile]", col(n1, cov1), "field"))
        panels.append((f"overlap  (# distinct tiles covering each cell, max={int(count.max())})", count, "count"))
        s = normal_pixel_stats(pred, gtn, v)
        r1 = finalize_normal(s, s["hist"])
        panels.append((f"STITCHED = INTEGRATED  (coverage-mean vectors -> renormalize)  "
                       f"mean={r1['mean']:.1f}°, <11.25={r1['pct_11']:.1f}%, coverage={cov*100:.1f}%",
                       col(pred, v), "field"))
        panels.append(("GT normals (world-frame, metric-valid pixels)", col(gtn, v), "field"))
        runlog.save_integration_figure(
            run, "s2d3d_normal", i, panels,
            suptitle=f"NORMAL eval INTEGRATION: per-tile -> overlap-averaged ERP @ {B.EH}x{B.EW} "
                     f"(Stanford2D3D fold{B.FOLD}, {'frozen DINOv3' if not B.ADAPTER else B.TAG})")


def main() -> None:
    torch.manual_seed(B.SEED)
    enc = B.build_encoder()
    plan = B.build_plan()
    B.build_sample_grids(plan)                                    # precompute GPU render grids once
    cids = [B.coord_map(tp.yaw_deg, tp.pitch_deg) for tp in plan]
    files = data.list_erps("stanford2d3d")
    tr, va = B.split_files(files, B.FOLD)
    print(f"Normal-S2D3D fold{B.FOLD} enc={B.TAG} model={B.MODEL} tiles/pano={len(plan)} "
          f"tr={len(tr)} va={len(va)} eval={B.EH}x{B.EW} ep={B.EPOCHS}", flush=True)

    t0 = time.time()
    ctr = build_cache(enc, tr, plan, want_full=False)
    cva = build_cache(enc, va, plan, want_full=True)
    print(f"encoded {len(tr)}+{len(va)} panos ({time.time()-t0:.0f}s)", flush=True)

    torch.manual_seed(B.SEED)
    head = B.DenseHead(enc.dim, 3).to(B.DEVICE)
    train_head(head, ctr)

    head.eval()
    agg = {k: 0.0 for k in STAT_KEYS}
    hist = np.zeros(NBINS)
    cov_sum = 0.0
    for feat, gt, valid in cva:
        pred, cov, covered = predict_normal(head, feat, cids)
        cov_sum += cov
        v = valid.numpy() & covered
        s = normal_pixel_stats(pred, gt.numpy(), v)
        for k in STAT_KEYS:
            agg[k] += s[k]
        hist += s["hist"]

    r = finalize_normal(agg, hist)
    coverage = cov_sum / max(len(cva), 1)
    head_M = sum(p.numel() for p in head.parameters()) / 1e6
    ours = (f"OURS (frozen DINOv3 + {head_M:.2f}M head)" if not B.ADAPTER
            else f"OURS (DINOv3 + SSL[{B.TAG}] + {head_M:.2f}M head)")

    print(f"\n=== Normal-S2D3D fold{B.FOLD} | {ours} | eval {B.EH}x{B.EW} ===", flush=True)
    print(f"  mean={r['mean']:.2f}deg  median={r['median']:.2f}deg  <11.25={r['pct_11']:.1f}%  "
          f"<22.5={r['pct_22']:.1f}%  <30={r['pct_30']:.1f}%  | stitch coverage {coverage*100:.1f}%", flush=True)
    print(f"\n{'method':30s} {'mean°':>7} {'med°':>7} {'<11.25%':>8}", flush=True)
    print(f"{ours[:30]:30s} {r['mean']:7.2f} {r['median']:7.2f} {r['pct_11']:8.1f}", flush=True)
    if SOTA:
        for name, me, md, p11 in SOTA:
            print(f"{name:30s} {me:7.2f} {md:7.2f} {p11:8.1f}", flush=True)
    else:
        print("(no web-verified pano-normal SOTA on Stanford2D3D — primary comparison is "
              "cross-encoder: rerun with ENC_ADAPTER=<adapter> or MODEL=<backbone>)", flush=True)
    leak = "" if not B.ADAPTER else " · SSL row is TRANSDUCTIVE (adapter pool incl. test areas)"
    print(f"caveats: our eval fold{B.FOLD}, {B.EH}x{B.EW}, head-only on FROZEN features, {len(tr)}-pano "
          f"train / {len(va)} test.{leak}", flush=True)

    run = runlog.create_run(f"normal_s2d3d_{B.TAG}_f{B.FOLD}", {
        "benchmark": "Stanford2D3D surface normal", "fold": B.FOLD, "encoder": B.ADAPTER or "frozen",
        "model": B.MODEL, "tag": B.TAG, "eval_hw": [B.EH, B.EW], "stitch_hw": [B.SH, B.SW],
        "epochs": B.EPOCHS, "tr_panos": len(tr), "va_panos": len(va), "metrics": r,
        "stitch_coverage": coverage, "head_M": head_M, "transductive": bool(B.ADAPTER),
        "sota": [{"method": m, "mean": me, "median": md, "pct_11": p} for m, me, md, p in SOTA],
        "protocol_caveats": "single fold, head-only on frozen features; no verified pano-normal SOTA "
        f"(cross-encoder comparison); SSL rows transductive={bool(B.ADAPTER)}"})
    torch.save(head.state_dict(), os.path.join(run, "weights", "normalhead.pt"))

    emit_integration(run, head, cva, cids, plan, va)
    print(f"saved -> {run} (config + weights + viz + SOTA table)", flush=True)


if __name__ == "__main__":
    main()
