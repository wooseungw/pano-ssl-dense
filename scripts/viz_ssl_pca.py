"""DINOv2/v3-style PCA feature visualization for panorama SSL pretraining.

Standard DINO viz = patch features -> PCA(3) -> RGB. For our pano setting we scatter the
per-tile patch features onto the shared ERP cell grid (scatter_mean_field, the SAME mean
field the downstream 'mean' fusion feeds UPerNet) and PCA-color THAT — so the picture shows
the fused panorama-wide representation, including whether semantics stay consistent ACROSS
the tile seams. Two columns per sample: frozen DINOv3 (adapter off) vs SSL (adapter on),
differing only by the LoRA delta — you see exactly what pretraining changed.

Runs on any saved adapter; wired into the scaled-geo run via RUN_DIR (writes into its viz/).
Run: ENC_ADAPTER=runs/ckpt_geo_scaled CUDA_VISIBLE_DEVICES=1 python scripts/viz_ssl_pca.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data  # noqa: E402
import probe_seg_dinov3 as P  # noqa: E402
import runlog  # noqa: E402
import train_fusion_f2 as F2  # noqa: E402
from encoder import PanoEncoder, normalize_tiles  # noqa: E402
from fusion import scatter_mean_field  # noqa: E402

DEVICE = P.DEVICE
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAPTER = os.environ.get("ENC_ADAPTER", os.path.join(ROOT, "runs", "ckpt_ssl_lora"))
DATASET = os.environ.get("VIZ_DATASET", "structured3d")
HFOV = float(os.environ.get("HFOV", 65.0))
N_VIZ = int(os.environ.get("N_VIZ", 3))                # fixed designated val samples
CHUNK = 8


@torch.no_grad()
def dense_flat(enc, tiles, frozen):
    """(T,3,H,W) tiles -> (T*N, D) patch features. frozen=True disables the LoRA adapter."""
    outs = []
    for i in range(0, tiles.shape[0], CHUNK):
        x = normalize_tiles(tiles[i:i + CHUNK].to(DEVICE))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            if frozen:
                with enc.backbone.disable_adapter():
                    f = P.dense(enc, x)                # (b,D,gh,gw)
            else:
                f = P.dense(enc, x)
        outs.append(f.float().permute(0, 2, 3, 1).reshape(f.shape[0], -1, f.shape[1]).cpu())
    return torch.cat(outs).reshape(-1, enc.dim)


def fit_pca(fields, covered):
    """Fit PCA(3) + robust color scale over all covered cells (jointly across samples ->
    consistent colors). Returns one shared color frame (mu, comps, lo, hi)."""
    X = torch.cat([f[covered] for f in fields], 0)     # (sum covered, D)
    mu = X.mean(0, keepdim=True)
    _, _, Vt = torch.linalg.svd(X - mu, full_matrices=False)
    comps = Vt[:3]                                      # (3, D)
    projX = (X - mu) @ comps.t()
    lo = torch.quantile(projX, 0.02, dim=0)            # robust scale (2-98 pct)
    hi = torch.quantile(projX, 0.98, dim=0)
    return mu, comps, lo, hi


def apply_pca(fields, basis):
    """Project fields through a FIXED PCA frame -> list of (ncell,3) in [0,1]. Frozen and SSL
    share the same 768-d space (SSL = frozen + LoRA delta), so applying the FROZEN frame to
    SSL makes colors directly comparable: same color == same feature direction; a color that
    shifts marks where SSL actually moved the feature."""
    mu, comps, lo, hi = basis
    rng = (hi - lo).clamp_min(1e-6)
    return [(((f - mu) @ comps.t() - lo) / rng).clamp(0, 1) for f in fields]


def to_img(rgb_cells, covered, hf, wf, patch):
    """(ncell,3) [0,1] -> upsampled HxW RGB; uncovered cells -> gray."""
    img = rgb_cells.clone()
    img[~covered] = 0.2
    grid = img.reshape(hf, wf, 3).numpy()
    return np.kron(grid, np.ones((patch, patch, 1)))


def main():
    P.configure(DATASET); P.TILE = 512
    F2.D.DATASET, F2.D.HFOV, F2.D.OVERLAP, F2.D.TILE = DATASET, HFOV, 0.25, 512
    enc = PanoEncoder(model_id=P.MODEL, adapter_path=ADAPTER).to(DEVICE).eval()
    P.enc_patch = enc.patch
    cfg = F2.build_config(HFOV)
    hf, wf = P.WORK_HW[0] // enc.patch, P.WORK_HW[1] // enc.patch

    allf = data.list_structured3d()
    by_scene = {}
    for fp in allf:
        by_scene.setdefault(fp.split("scene_")[1][:5], []).append(fp)
    scenes = sorted(by_scene)
    nval = max(1, len(scenes) // 10)
    va = [fp for s in scenes[-nval:] for fp in by_scene[s]]     # held-out scenes (same as downstream)
    samples = va[:N_VIZ]
    tag = os.path.basename(ADAPTER.rstrip("/"))
    print(f"viz-pca adapter={tag} dataset={DATASET} samples={len(samples)} cells={hf}x{wf}", flush=True)

    rgbs, fields_fro, fields_ssl, covered = [], [], [], None
    for f in samples:
        rgb = np.array(Image.open(f).convert("RGB").resize((P.WORK_HW[1], P.WORK_HW[0]), Image.BILINEAR))
        tiles = F2.render_cfg_tiles(rgb, cfg)
        ff = dense_flat(enc, tiles, frozen=True)
        fs = dense_flat(enc, tiles, frozen=False)
        field_f, counts = scatter_mean_field(cfg["cid"], ff.float(), cfg["ncell"])
        field_s, _ = scatter_mean_field(cfg["cid"], fs.float(), cfg["ncell"])
        rgbs.append(rgb.astype(np.float32) / 255.0)
        fields_fro.append(field_f); fields_ssl.append(field_s)
        covered = counts > 0                                    # geometry-fixed, same every sample

    basis = fit_pca(fields_fro, covered)               # PCA frame from FROZEN features...
    rgb_fro = apply_pca(fields_fro, basis)
    rgb_ssl = apply_pca(fields_ssl, basis)             # ...applied to SSL -> comparable colors

    run = os.environ.get("RUN_DIR")
    if run:
        os.makedirs(os.path.join(run, "viz"), exist_ok=True)
    else:
        run = runlog.create_run(f"ssl_pca_{tag}", {
            "adapter": ADAPTER, "dataset": DATASET, "hfov": HFOV, "n_viz": len(samples),
            "cells": [hf, wf], "viz": "ERP mean-field PCA(3)->RGB, shared FROZEN basis (frozen vs SSL comparable)"})
    viz = os.path.join(run, "viz")
    for i, f in enumerate(samples):
        img_in = rgbs[i]
        img_fro = to_img(rgb_fro[i], covered, hf, wf, enc.patch)
        img_ssl = to_img(rgb_ssl[i], covered, hf, wf, enc.patch)
        for name, im in [("input", img_in), ("pca_frozen", img_fro), (f"pca_ssl_{tag}", img_ssl)]:
            Image.fromarray((np.clip(im, 0, 1) * 255).astype(np.uint8)).save(
                os.path.join(viz, f"{DATASET}_s{i}_{name}.png"))
        runlog.save_panel(run, DATASET, i, [
            ("input ERP", img_in),
            ("frozen DINOv3  (adapter off) — PCA(3) basis", img_fro),
            (f"SSL {tag}  (adapter on) — SAME frozen PCA basis", img_ssl)])
    print(f"saved DINO-style PCA viz -> {viz}", flush=True)


if __name__ == "__main__":
    main()
