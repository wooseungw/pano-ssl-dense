"""SEG eval INTEGRATION viz across EVERY seg dataset (Stanford2D3D + DensePASS + Structured3D).

Generalizes viz_stitch_demo.py from one S2D3D pano to all three wired seg datasets, 3 spread
val samples each, on the REAL metric path: per-tile logits -> coverage-normalized STITCH at the
coarse (SH,SW) grid -> bilinear-upsample to the EVAL grid -> argmax (seg_s2d3d_bench.predict_erp,
the exact benchmark stitch — so the picture is scored at the metric's eval image size, not a
cosmetic resolution). Each figure: input ERP -> a few single-tile ERP footprints (patch-wise)
-> overlap count -> STITCHED-as-scored (mIoU annotated) -> GT.

Only S2D3D has a trained seghead on disk; DensePASS/Structured3D get a small CAPPED-train probe
head (labelled illustrative — NOT a benchmark number). Datasets with no panos on disk are skipped.

Run: CUDA_VISIBLE_DEVICES=0 conda run -n pano python scripts/viz_seg_integration.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("ENC_ADAPTER", "")          # frozen encoder unless overridden — precede bench import
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import probe_seg_dinov3 as P  # noqa: E402
import runlog  # noqa: E402
import seg_s2d3d_bench as B  # noqa: E402  (SegHead/render_pano/encode/coord_map/predict_erp/SH/SW/EH/EW)

DEVICE = B.DEVICE
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEGHEAD_S2D3D = os.path.join(ROOT, "runs", "0704_0524_seg_s2d3d_frozen_f1", "weights", "seghead.pt")

CAP_TR = int(os.environ.get("CAP_TR", 100))       # capped probe-train panos (viz, not a leaderboard)
CAP_VA = int(os.environ.get("CAP_VA", 40))
EPOCHS = int(os.environ.get("EPOCHS", 12))

# (dataset, hfov, plan-mode) — S2D3D/S3D indoor full-sphere 65°; DensePASS outdoor equator band 50°.
DATASETS = [("stanford2d3d", 65.0, "full_sphere"),
            ("densepass", 50.0, "ring"),
            ("structured3d", 65.0, "full_sphere")]


def build_plan(hfov: float, mode: str):
    """Coverage-complete tile schedule for the dataset's geometry."""
    if mode == "ring":                                                # DensePASS: content is an equator band
        n = P.a2p._ring_yaw_count(hfov, 0.25, 0.0, 90.0)
        return [P.a2p.TilePlan(y, 0.0) for y in P.a2p._ring_yaws(n, 0.0)]
    return P.a2p.plan_tiles("full_sphere", hfov, hfov, 0.25)          # indoor 3-ring + pole caps


def render(rgb_np: np.ndarray, lab_np: np.ndarray, plan, hfov: float):
    """(H,W,3) ERP + (H,W) label -> (T,3,TILE,TILE) RGB tiles + (T,HEAD_OUT,HEAD_OUT) label tiles.

    Uses the already-loaded rgb/lab (P.load_rgb_label handles each dataset's ERP quirks, e.g.
    DensePASS's band-into-2:1 pad) rather than B.render_pano's naive resize, so RGB and GT align."""
    tiles, labs = [], []
    for tp in plan:
        t = np.asarray(P.a2p.erp_to_pinhole_tile(rgb_np, tp.yaw_deg, tp.pitch_deg, hfov, B.TILE))
        tiles.append(torch.from_numpy(t).float().permute(2, 0, 1) / 255.0)
        gl = P.e2p_label(lab_np, tp.yaw_deg, tp.pitch_deg, hfov, B.TILE)
        gl = np.array(Image.fromarray(gl.astype(np.uint8)).resize((B.HEAD_OUT, B.HEAD_OUT), Image.NEAREST))
        labs.append(torch.from_numpy(gl.astype(np.int64)))
    return torch.stack(tiles), torch.stack(labs)


def train_head(head, ctr) -> None:
    """Compact CAPPED probe train (mirrors seg_s2d3d_bench's loop) — for datasets with no saved head."""
    opt = torch.optim.AdamW(head.parameters(), 1e-3, weight_decay=1e-4)
    lossf = torch.nn.CrossEntropyLoss(ignore_index=P.IGNORE)
    g = torch.Generator().manual_seed(0)
    for _ in range(EPOCHS):
        head.train()
        for i in torch.randperm(len(ctr), generator=g).tolist():
            feat, labs = ctr[i]
            opt.zero_grad()
            for s in range(0, feat.shape[0], B.CHUNK):
                yb = labs[s:s + B.CHUNK].long().to(DEVICE)
                if not (yb != P.IGNORE).any():
                    continue
                lg = head(feat[s:s + B.CHUNK].float().to(DEVICE))
                (lossf(lg, yb) * (yb.shape[0] / feat.shape[0])).backward()
            opt.step()


def sample_miou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Per-pano mIoU over classes PRESENT in this pano's GT (for the panel annotation)."""
    m = gt != P.IGNORE
    ious = []
    for c in range(1, P.N_CLASS):
        gc = (gt == c) & m
        if gc.sum() == 0:
            continue
        pc = (pred == c) & m
        u = (pc | gc).sum()
        ious.append(((pc & gc).sum() / u).item() if u > 0 else 0.0)
    return float(np.mean(ious)) if ious else 0.0


@torch.no_grad()
def scatter_tile_pred(head, feat_ti, cid) -> np.ndarray:
    """One tile's per-pixel argmax scattered onto the (SH,SW) stitch grid; -1 where uncovered."""
    lg = head(feat_ti[None].float().to(DEVICE))
    lg = torch.nn.functional.interpolate(lg, (B.TILE_OUT, B.TILE_OUT), mode="bilinear", align_corners=False)[0]
    am = lg.argmax(0).reshape(-1).cpu()
    grid = torch.full((B.SH * B.SW,), -1, dtype=torch.long)
    grid[cid] = am
    return grid.reshape(B.SH, B.SW).numpy()


def run_dataset(enc, run: str, ds: str, hfov: float, mode: str) -> None:
    P.configure(ds)
    P.TILE, P.WORK_HW = B.TILE, (B.EH, B.EW)                          # GT/eval at the metric resolution
    P.enc_patch = enc.patch
    B.HFOV = hfov                                                     # B.coord_map/predict use this
    plan = build_plan(hfov, mode)
    cids = [B.coord_map(tp.yaw_deg, tp.pitch_deg) for tp in plan]

    panos, _, train = P.grouped()
    tr = [f for g, f in panos if g in train][:CAP_TR]
    va = [f for g, f in panos if g not in train][:CAP_VA]
    if not va:
        print(f"[{ds}] skipped — no val panos on disk", flush=True)
        return
    print(f"[{ds}] N_CLASS={P.N_CLASS} tiles/pano={len(plan)} hfov={hfov} tr={len(tr)} va={len(va)} "
          f"stitch={B.SH}x{B.SW} eval={B.EH}x{B.EW}", flush=True)

    ctr = []
    for f in tr:
        tiles, labs = render(*P.load_rgb_label(f), plan, hfov)
        ctr.append((B.encode(enc, tiles).half(), labs.to(torch.int16)))
    cva = []
    for f in va:                                                     # val: cache full-ERP GT at eval res
        rgb, lab = P.load_rgb_label(f)
        tiles, _ = render(rgb, lab, plan, hfov)
        cva.append((B.encode(enc, tiles).half(), lab))

    head = B.SegHead(enc.dim, P.N_CLASS).to(DEVICE)
    # Load the trained frozen-run head ONLY for the frozen encoder; an SSL adapter changes the
    # features, so its head must be re-probed (the frozen head would score garbage on adapted feats).
    if ds == "stanford2d3d" and not B.ADAPTER and os.path.exists(SEGHEAD_S2D3D):
        head.load_state_dict(torch.load(SEGHEAD_S2D3D, map_location="cpu"))
        note, fig_note = "loaded frozen-run seghead", "trained seghead"
    else:
        train_head(head, ctr)
        note = f"illustrative probe: {len(tr)}-pano capped train, {EPOCHS}ep (NOT a benchmark)"
        fig_note = "illustrative probe head (not a benchmark)"
    head.eval()

    # dataset mIoU over the cached val panos (dataset-aggregated, all 13/19/40 classes)
    inter, union, cov_sum = torch.zeros(P.N_CLASS), torch.zeros(P.N_CLASS), 0.0
    preds = []
    for feat, gt in cva:
        pred, cov = B.predict_erp(head, feat, cids)
        preds.append((pred, cov))
        cov_sum += cov
        m = gt != P.IGNORE
        for c in range(1, P.N_CLASS):
            pc, gc = (pred == c) & m, (gt == c) & m
            inter[c] += (pc & gc).sum()
            union[c] += (pc | gc).sum()
    ious = [(inter[c] / union[c]).item() for c in range(1, P.N_CLASS) if union[c] > 0]
    ds_miou = float(np.mean(ious)) if ious else 0.0
    coverage = cov_sum / max(len(cva), 1)
    print(f"[{ds}] dataset mIoU={ds_miou*100:.1f} ({len(ious)}/{P.N_CLASS-1} cls present) "
          f"coverage={coverage*100:.1f}% | {note}", flush=True)

    cnt = torch.zeros(B.SH * B.SW)                                    # overlap: # distinct tiles per stitch cell
    for cid in cids:
        cnt[cid.unique()] += 1.0
    count = cnt.reshape(B.SH, B.SW).numpy()
    eq = [j for j, tp in enumerate(plan) if abs(tp.pitch_deg) < 1e-3]
    tpick = ([eq[0], eq[len(eq) // 3], eq[2 * len(eq) // 3]] if len(eq) >= 3 else list(range(min(3, len(plan)))))
    pal = runlog.seg_palette(P.N_CLASS)

    for i in runlog.spread_indices(len(cva), 3):                      # spread picks (not first-3 near-dups)
        feat, gt = cva[i]
        pred, cov = preds[i]
        rgb = P.load_rgb_label(va[i])[0].astype(np.float32) / 255.0
        panels = [("input ERP (equirectangular panorama)", rgb, "rgb")]
        for ti in tpick:
            sc = scatter_tile_pred(head, feat[ti], cids[ti])
            panels.append((f"single tile only  (yaw={plan[ti].yaw_deg:.0f}°, pitch={plan[ti].pitch_deg:.0f}°) "
                           f" -> its ERP footprint [per-tile / patch-wise]", runlog.colorize(sc, pal), "field"))
        panels.append((f"overlap  (# distinct tiles covering each cell, max={int(count.max())})", count, "count"))
        panels.append((f"STITCHED = INTEGRATED  (coverage-normalized logit mean -> argmax)  "
                       f"mIoU={sample_miou(pred, gt)*100:.1f}, coverage={cov*100:.1f}%",
                       runlog.colorize(pred, pal), "field"))
        panels.append((f"GT ({P.N_CLASS-1}-class)", runlog.colorize(gt, pal), "field"))
        runlog.save_integration_figure(
            run, ds, i, panels,
            suptitle=f"SEG eval INTEGRATION: per-tile -> overlap-averaged ERP @ {B.EH}x{B.EW}  "
                     f"({ds}, {'frozen DINOv3' if not B.ADAPTER else B.TAG} · {fig_note})")
        runlog.save_seg_sample(run, ds, i, rgb, gt, {B.TAG: pred}, pal, scale=1)
    print(f"[{ds}] saved {len(runlog.spread_indices(len(cva), 3))} integration figures", flush=True)


def emit(run: str, enc) -> None:
    """Emit seg integration figures for every wired seg dataset into run/viz/ (given a built encoder).
    Importable entry point for train-time viz (train_viz.py); one dataset failing skips only that one."""
    for ds, hfov, mode in DATASETS:
        try:
            run_dataset(enc, run, ds, hfov, mode)
        except Exception as e:                                        # one dataset failing must not kill the rest
            print(f"[{ds}] FAILED: {type(e).__name__}: {e}", flush=True)


def main() -> None:
    torch.manual_seed(0)
    enc = B.build_encoder()                                           # frozen DINOv3 (or ENC_ADAPTER override)
    run = runlog.create_run("seg_integration", {
        "purpose": "seg eval integration viz across all seg datasets", "encoder": B.ADAPTER or "frozen",
        "tag": B.TAG, "datasets": [d for d, _, _ in DATASETS], "cap_tr": CAP_TR, "cap_va": CAP_VA,
        "epochs": EPOCHS, "eval_hw": [B.EH, B.EW], "stitch_hw": [B.SH, B.SW], "tile_out": B.TILE_OUT,
        "s2d3d_seghead": SEGHEAD_S2D3D,
        "note": "S2D3D uses the trained frozen-run head; DensePASS/S3D use a capped illustrative probe"})
    emit(run, enc)
    print(f"saved -> {run} (config + per-dataset integration viz)", flush=True)


if __name__ == "__main__":
    main()
