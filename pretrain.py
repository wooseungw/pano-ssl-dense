"""E2P-overlap SSL pretraining skeleton for the panorama encoder.

Stage-4 of the plan (memory/ssl-loss-recommendation.md): frozen DINOv2 + LoRA, adapted with
warp-equivariance + distillation + VICReg, with a WARM-UP that ramps the warp weight from 0
(so the dense hard constraint cannot drive early constant-collapse before the adapter settles).

This is a runnable skeleton (`python pretrain.py`), not a finished training run: it wires
the geometry, losses, and schedule end-to-end and verifies the loss behaves. Plug in a real
ERP dataloader + multi-GPU + checkpointing for production.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

import anyres_e2p as a2p
import data
import geometry as G
from encoder import PanoEncoder, normalize_tiles
from losses import combined_loss

ERP_W, ERP_H = 2048, 1024


@dataclass
class Geometry:
    specs: List[Tuple[float, float]]                 # (yaw, pitch) per tile
    pairs: List[Tuple[int, int]]                     # adjacent overlapping tile ids
    warps: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]  # (grid, valid, weight) per pair


def build_geometry(encoder: PanoEncoder, img_size: int, hfov: float = 90.0,
                   overlap: float = 0.3, pitch_centers=(-45.0, 0.0, 45.0)) -> Geometry:
    yaws = a2p.make_yaw_centers_closed_loop(hfov, overlap, start_deg=-180.0)
    k = len(yaws)
    specs: List[Tuple[float, float]] = []
    idx = {}
    for r, pitch in enumerate(pitch_centers):
        for c, yaw in enumerate(yaws):
            idx[(r, c)] = len(specs)
            specs.append((yaw, pitch))

    pairs: List[Tuple[int, int]] = []
    for r in range(len(pitch_centers)):
        for c in range(k):
            pairs.append((idx[(r, c)], idx[(r, (c + 1) % k)]))            # horizontal (wraps seam)
            if r + 1 < len(pitch_centers):
                pairs.append((idx[(r, c)], idx[(r + 1, c)]))              # vertical

    cmaps = [G.render_coordmap(ERP_H, ERP_W, y, p, hfov, img_size) for (y, p) in specs]
    warps = []
    kept = []
    for (a, b) in pairs:
        wf = G.warp_field_from_coordmaps(cmaps[a], cmaps[b], encoder.patch, hfov,
                                         erp_w=ERP_W, dst_stride=3)
        if wf.valid.mean() < 0.05:
            continue                                                     # skip non-overlapping
        warps.append((torch.from_numpy(wf.grid), torch.from_numpy(wf.valid),
                      torch.from_numpy(wf.weight)))
        kept.append((a, b))
    return Geometry(specs=specs, pairs=kept, warps=warps)


def render_tiles(erp_np: np.ndarray, specs: List[Tuple[float, float]], hfov: float,
                 img_size: int) -> torch.Tensor:
    tiles = []
    for (yaw, pitch) in specs:
        t = a2p.erp_to_pinhole_tile(erp_np, yaw, pitch, hfov, img_size)
        tiles.append(torch.from_numpy(np.asarray(t)).float().permute(2, 0, 1) / 255.0)
    return torch.stack(tiles, 0)


def train(model_id: str = "facebook/dinov2-base", img_size: int = 518, steps: int = 12,
          warmup: int = 5, w_warp_max: float = 1.0, lr: float = 1e-4, n_erp: int = 4,
          device: str = "cuda") -> None:
    torch.manual_seed(0)
    enc = PanoEncoder(model_id=model_id, lora_rank=16).to(device).train()
    geom = build_geometry(enc, img_size)
    warps = [(g.to(device), v.to(device), w.to(device)) for (g, v, w) in geom.warps]
    print(f"tiles={len(geom.specs)} overlapping_pairs={len(geom.pairs)} "
          f"trainable_params={sum(p.numel() for p in enc.trainable_parameters())/1e6:.2f}M")

    files = data.list_erps("stanford2d3d", n_erp)
    erps = [np.array(Image.open(f).convert("RGB").resize((ERP_W, ERP_H))) for f in files]

    opt = torch.optim.AdamW(enc.trainable_parameters(), lr=lr)
    for step in range(steps):
        erp = erps[step % len(erps)]
        tiles = normalize_tiles(render_tiles(erp, geom.specs, 90.0, img_size).to(device))
        student = enc(tiles)                                             # (T,D,Gh,Gw)
        teacher = enc.teacher(tiles)

        w_warp = w_warp_max * min(1.0, step / max(1, warmup))            # ramp 0 -> max
        total = 0.0
        agg = {}
        for (a, b), warp in zip(geom.pairs, warps):
            loss, comps = combined_loss(student[a:a + 1], student[b:b + 1],
                                        teacher[a:a + 1], teacher[b:b + 1], warp,
                                        w_warp=w_warp)
            total = total + loss
            for kk, vv in comps.items():
                agg[kk] = agg.get(kk, 0.0) + vv.item()
        total = total / len(geom.pairs)

        opt.zero_grad(); total.backward(); opt.step()
        msg = " ".join(f"{k}={v/len(geom.pairs):.3f}" for k, v in agg.items())
        print(f"step {step:2d} w_warp={w_warp:.2f} | {msg}")


if __name__ == "__main__":
    train()
