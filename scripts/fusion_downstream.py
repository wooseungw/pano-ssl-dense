"""Stage 2 (headline): does F-2's fusion win survive a REAL decoder + MULTIPLE SEEDS?

Encoder fixed = TC3 (champion; per-tile accuracy is encoder-flat, so the axis here is
FUSION). For each seed, PAIRED: mean-fusion vs learned F-2 SetFusion, feeding the SAME
UPerNet decoder (shared capacity/data/init — only the fusion step differs). Tasks seg +
depth on Structured3D (where F-2's +0.078 was measured), scene-disjoint split.

This IS the F-2 de-risk the user deferred (advisor): §3.9 showed UPerNet can wash out what
fusion adds, and single-split produced two false positives this session — so multi-seed +
real decoder is the honest test. Verdict per task = mean±std of the paired (attn − mean) Δ.

Run: OPENCV_IO_ENABLE_OPENEXR=1 CUDA_VISIBLE_DEVICES=<n> conda run -n pano \
       python scripts/fusion_downstream.py [seg|depth]
"""
from __future__ import annotations

import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data  # noqa: E402
import probe_seg_dinov3 as P  # noqa: E402
import train_fusion_f2 as F2  # noqa: E402  (build_config, encode_tiles, render_cfg_tiles, MAX_COV)
import runlog  # noqa: E402
from encoder import PanoEncoder  # noqa: E402
from fusion import SetFusion, masked_mean, pack_sets  # noqa: E402
from multitask_eval import UPerNet  # noqa: E402  (real decoder head)

DEVICE = P.DEVICE
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAPTER = os.environ.get("ENC_ADAPTER", os.path.join(ROOT, "runs", "ckpt_ssl_tc3"))
TASK = sys.argv[1] if len(sys.argv) > 1 else "seg"
N_SEEDS = int(os.environ.get("N_SEEDS", 5))               # n=5 -> unanimous-sign p=0.031 (n=3 was 0.25)
S3D_TRAIN = int(os.environ.get("S3D_TRAIN", 400))
S3D_VAL = int(os.environ.get("S3D_VAL", 120))
EPOCHS = int(os.environ.get("EPOCHS", 12))
HFOV = 65.0
LOG125 = math.log(1.25)


def load_gt(f, task, hf, wf):
    """(hf,wf) target grid + valid mask, at the ERP field resolution."""
    if task == "seg":
        lab = P.label_to_grid(P.load_rgb_label(f)[1], hf, wf)
        return torch.from_numpy(lab), torch.from_numpy(lab != P.IGNORE)
    d = np.array(Image.open(data.s3d_gt_path(f, "depth")).resize((wf, hf), Image.NEAREST)).astype(np.float32)
    v = (d > 0) & (d < 65535)
    med = np.median(d[v]) if v.any() else 1.0
    dlog = np.log(np.clip(d / med, 1e-3, None))
    return torch.from_numpy(dlog).float(), torch.from_numpy(v)


def prep_cpu(f, cfg, hf, wf):
    """CPU-only: render 22 tiles (py360convert) + load GT. This is the serial bottleneck
    that was STARVING the GPU (util ~0%); run it in a thread pool to overlap with encode."""
    rgb = np.array(Image.open(f).convert("RGB").resize((P.WORK_HW[1], P.WORK_HW[0]), Image.BILINEAR))
    tiles = F2.render_cfg_tiles(rgb, cfg)
    gt, val = load_gt(f, TASK, hf, wf)
    return tiles, gt, val


def pack(feats, cfg):
    """(T*N, D) tile features -> padded packed sets (fset,gset,mask). Redone per step: the
    padded (ncell,max_cov,D) tensor is ~300MB, far too big to cache 520 of (OOM)."""
    return pack_sets(cfg["cid"], feats.float(), cfg["geo"], cfg["ncell"], F2.MAX_COV)


FEAT_CACHE = os.path.join(ROOT, "runs", "_feat_cache")
# Disk feature-cache is OPT-IN (FEAT_DISK_CACHE=1) and only sane for SMALL subsets: a full
# dataset is ~34MB/pano * N -> 21.8k panos ≈ 740GB/encoder (storage-infeasible). By default
# we cache features IN RAM per run only (encode once, reuse across seeds/epochs), freed on exit.
DISK_CACHE = os.environ.get("FEAT_DISK_CACHE", "0") == "1"


@torch.no_grad()
def build_cache(enc, files, cfg, hf, wf, split, workers=8):
    """COMPACT per-tile features (fp16) for THIS run's subset (RAM). GT reloaded per task."""
    covered = torch.zeros(cfg["ncell"], dtype=torch.bool)
    covered[cfg["cid"]] = True                                            # covered cells (F-2 basis)
    path = None
    if DISK_CACHE:
        os.makedirs(FEAT_CACHE, exist_ok=True)
        path = os.path.join(FEAT_CACHE, f"{P.DATASET}_{os.path.basename(ADAPTER)}_h{int(HFOV)}_{split}{len(files)}.pt")
    feats_list = None
    if path and os.path.exists(path):
        blob = torch.load(path, map_location="cpu")
        if blob.get("files") == files:
            feats_list = blob["feats"]
            print(f"loaded cached features {split} ({len(files)})", flush=True)
    if feats_list is None:
        feats_list = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for tiles, _, _ in ex.map(lambda ff: prep_cpu(ff, cfg, hf, wf), files):
                feats = F2.encode_tiles(enc, tiles)                      # GPU (main thread)
                feats_list.append(feats.reshape(-1, feats.shape[-1]).half().cpu())
        if path:
            torch.save({"files": files, "feats": feats_list}, path)
    gts = [load_gt(f, TASK, hf, wf) for f in files]                      # per-task GT, no GPU
    return [(feats_list[i], g.reshape(-1), (v.reshape(-1) & covered)) for i, (g, v) in enumerate(gts)]


def fuse_grid(fusion, fset, gset, mask, hf, wf, dim):
    """Packed sets -> dense ERP feature grid (1,D,hf,wf). mean or trainable SetFusion."""
    cov = mask.any(1)
    fused = fset.new_zeros(fset.shape[0], dim)
    fc = fusion(fset[cov], gset[cov], mask[cov]) if fusion is not None else masked_mean(fset[cov], mask[cov])
    fused[cov] = fc
    return fused.reshape(1, hf, wf, dim).permute(0, 3, 1, 2), cov


def run_one(enc, cfg, hf, wf, dim, cache_tr, cache_va, fusion_kind, seed):
    """Train UPerNet (+ SetFusion if attn) on cached panos; return (val_metric, dec, fusion)."""
    torch.manual_seed(seed)
    c = P.N_CLASS if TASK == "seg" else 1
    dec = UPerNet(dim, c).to(DEVICE)
    fusion = SetFusion(dim=dim).to(DEVICE) if fusion_kind == "attn" else None
    params = list(dec.parameters()) + (list(fusion.parameters()) if fusion else [])
    opt = torch.optim.AdamW(params, 1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        if fusion:
            fusion.train()
        dec.train()
        for i in torch.randperm(len(cache_tr), generator=g).tolist():
            feats, gt, val = cache_tr[i]
            fset, gset, mask = pack(feats, cfg)
            grid, _ = fuse_grid(fusion, fset.to(DEVICE), gset.to(DEVICE), mask.to(DEVICE), hf, wf, dim)
            out = F.interpolate(dec(grid), size=(hf, wf), mode="bilinear", align_corners=False)  # (1,C,hf,wf)
            gtl = gt.to(DEVICE); m = val.to(DEVICE)                 # val = covered & labelled
            if TASK == "seg":
                logits = out.reshape(c, -1).t()                     # (hf*wf, C)
                loss = F.cross_entropy(logits[m], gtl[m]) if m.any() else None
            else:
                pr = out.reshape(-1)[m]; tg = gtl[m]
                loss = F.l1_loss(pr, tg) if m.any() else None
            if loss is None:
                continue
            opt.zero_grad(); loss.backward(); opt.step()
    # eval
    if fusion:
        fusion.eval()
    dec.eval()
    if TASK == "seg":
        inter = torch.zeros(P.N_CLASS); union = torch.zeros(P.N_CLASS)
    else:
        err = dac = n = 0
    with torch.no_grad():
        for feats, gt, val in cache_va:
            fset, gset, mask = pack(feats, cfg)
            grid, _ = fuse_grid(fusion, fset.to(DEVICE), gset.to(DEVICE), mask.to(DEVICE), hf, wf, dim)
            out = F.interpolate(dec(grid), size=(hf, wf), mode="bilinear", align_corners=False)
            gtl = gt; m = val
            if TASK == "seg":
                pr = out.argmax(1)[0].cpu().reshape(-1); mm = m
                for cc in range(1, P.N_CLASS):
                    inter[cc] += ((pr == cc) & (gtl == cc) & mm).sum()
                    union[cc] += (((pr == cc) | (gtl == cc)) & mm).sum()
            else:
                pr = out.reshape(-1).cpu()[m]; tg = gtl[m]; e = (pr - tg).abs()
                err += e.sum().item(); dac += (e < LOG125).float().sum().item(); n += int(m.sum())
    if TASK == "seg":
        metric = float(np.mean([(inter[c] / union[c]).item() for c in range(1, P.N_CLASS) if union[c] > 0]))
    else:
        metric = (err / max(n, 1), dac / max(n, 1))
    return metric, dec, fusion


@torch.no_grad()
def predict_grid(dec, fusion, feats, cfg, hf, wf, dim):
    fset, gset, mask = pack(feats, cfg)
    grid, _ = fuse_grid(fusion, fset.to(DEVICE), gset.to(DEVICE), mask.to(DEVICE), hf, wf, dim)
    out = F.interpolate(dec(grid), size=(hf, wf), mode="bilinear", align_corners=False)[0].cpu()
    return out.argmax(0).numpy() if TASK == "seg" else out[0].numpy()      # (hf,wf)


def save_run(run, models, cfg, enc, va_files, cache_va, hf, wf, dim, k=3):
    """weights/ (both fusions' dec+fusion) + viz/ (input, GT, pred_mean, pred_attn)."""
    for kind, (dec, fusion) in models.items():
        torch.save({"decoder": dec.state_dict(),
                    "fusion": fusion.state_dict() if fusion else None,
                    "kind": kind, "task": TASK, "encoder": ADAPTER},
                   os.path.join(run, "weights", f"{kind}.pt"))
    pal = runlog.seg_palette(P.N_CLASS)
    for i in range(min(k, len(cache_va))):
        feats, gt, val = cache_va[i]
        rgb = np.array(Image.open(va_files[i]).convert("RGB").resize(
            (P.WORK_HW[1], P.WORK_HW[0]), Image.BILINEAR)).astype(np.float32) / 255.0
        preds = {kind: predict_grid(dec, fusion, feats, cfg, hf, wf, dim)
                 for kind, (dec, fusion) in models.items()}
        gt_grid = gt.reshape(hf, wf).numpy()
        if TASK == "seg":
            runlog.save_seg_sample(run, "s3d", i, rgb, gt_grid, preds, pal)
            col = lambda g: runlog.colorize(g, pal)
        else:
            vg = val.reshape(hf, wf).numpy()
            runlog.save_depth_sample(run, "s3d", i, rgb, gt_grid, preds, vg)
            v = vg.astype(bool); lo, hi = (gt_grid[v].min(), gt_grid[v].max()) if v.any() else (0, 1)
            col = lambda g: runlog._turbo((g - lo) / max(hi - lo, 1e-6))
        runlog.save_panel(run, "s3d", i, [("input ERP", rgb), ("GT", col(gt_grid)),
                                          ("pred: mean-fusion", col(preds["mean"])),
                                          ("pred: F-2 attn-fusion", col(preds["attn"]))])


def main():
    P.configure("structured3d"); P.TILE = 512
    F2.D.DATASET, F2.D.HFOV, F2.D.OVERLAP, F2.D.TILE = "structured3d", HFOV, 0.25, 512
    enc = PanoEncoder(model_id=P.MODEL, adapter_path=ADAPTER).to(DEVICE).eval()
    P.enc_patch = enc.patch
    cfg = F2.build_config(HFOV)
    hf, wf = P.WORK_HW[0] // enc.patch, P.WORK_HW[1] // enc.patch
    dim = enc.dim

    allf = data.list_structured3d()
    by_scene = {}
    for fp in allf:
        by_scene.setdefault(fp.split("scene_")[1][:5], []).append(fp)
    scenes = sorted(by_scene)
    nval = max(1, len(scenes) // 10)
    va = [fp for s in scenes[-nval:] for fp in by_scene[s]][:S3D_VAL]
    tr = [fp for s in scenes[:-nval] for fp in by_scene[s]][:S3D_TRAIN]
    print(f"fusion-downstream TASK={TASK} enc={os.path.basename(ADAPTER)} decoder=UPerNet "
          f"tr={len(tr)} va={len(va)} seeds={N_SEEDS} ep={EPOCHS}", flush=True)

    t0 = time.time()
    cache_tr = build_cache(enc, tr, cfg, hf, wf, "tr")
    cache_va = build_cache(enc, va, cfg, hf, wf, "va")
    print(f"encoded {len(tr)}+{len(va)} panos ({time.time()-t0:.0f}s)", flush=True)

    rows = []
    seed0_models = {}
    for seed in range(N_SEEDS):
        r = {}
        for kind in ("mean", "attn"):
            metric, dec, fusion = run_one(enc, cfg, hf, wf, dim, cache_tr, cache_va, kind, seed)
            r[kind] = metric
            if seed == 0:
                seed0_models[kind] = (dec, fusion)
            print(f"  seed{seed} {kind:4s} -> {metric}", flush=True)
        rows.append(r)

    print(f"\n=== {TASK}: mean vs F-2 attn fusion, UPerNet, {N_SEEDS} seeds (paired, covered cells) ===", flush=True)
    p_unan = 0.5 ** (N_SEEDS - 1)                              # unanimous-sign p under null
    if TASK == "seg":
        d = np.array([r["attn"] - r["mean"] for r in rows])
        mean_m = np.array([r["mean"] for r in rows]); attn_m = np.array([r["attn"] for r in rows])
        npos = int((d > 0).sum())
        verdict = "consistent" if (abs(d.mean()) > d.std() and npos in (0, N_SEEDS)) else "within-noise"
        print(f"mean mIoU {mean_m.mean():.3f}±{mean_m.std():.3f}  attn {attn_m.mean():.3f}±{attn_m.std():.3f}", flush=True)
        print(f"Δ(attn−mean) = {d.mean():+.4f} ± {d.std():.4f}  ({npos}/{N_SEEDS} positive, unanimous p={p_unan:.3f})  [{verdict}]", flush=True)
    else:
        de = np.array([r["mean"][0] - r["attn"][0] for r in rows])   # |Δlog| lower better -> mean−attn
        print(f"mean |Δlog| {np.mean([r['mean'][0] for r in rows]):.4f}  "
              f"attn {np.mean([r['attn'][0] for r in rows]):.4f}  "
              f"Δ(improve) = {de.mean():+.4f} ± {de.std():.4f}  (unanimous p={p_unan:.3f})", flush=True)
    print("caveat: attn adds SetFusion's ~1.48M params -> a win = 'learned fusion module helps' "
          "(incl. its capacity), not 'fusion beats mean at equal params'.", flush=True)
    print(f"(total {time.time()-t0:.0f}s)", flush=True)

    run = runlog.create_run(f"fusion_downstream_{TASK}", {
        "task": TASK, "encoder": ADAPTER, "decoder": "UPerNet", "fusions": ["mean", "attn"],
        "seeds": N_SEEDS, "s3d_train": len(tr), "s3d_val": len(va), "epochs": EPOCHS,
        "rows": [{k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items()} for r in rows]})
    save_run(run, seed0_models, cfg, enc, va, cache_va, hf, wf, dim)   # weights/ + viz/ (GT alongside)
    print(f"saved -> {run} (config + weights + viz)", flush=True)


if __name__ == "__main__":
    main()
