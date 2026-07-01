"""E2P-overlap SSL: LoRA-adapt frozen DINOv3 on a mixed unlabeled pano pool.

One adapter, trained label-free on Structured3D + Stanford2D3D(train) + DensePASS(train).
Each domain is tiled at its eval-matched FOV (indoor 65deg 3-ring, outdoor 50deg equator).
Geometry (tile specs + overlap warp fields) is image-independent -> built ONCE per domain
and reused every step. Loss = warp-equivariance (warm-up ramped) + distill(token+relational)
+ VICReg(var+cov). Saves the LoRA adapter for eval_ssl.py.

Run: CUDA_VISIBLE_DEVICES=1 conda run -n pano python scripts/train_ssl.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import anyres_e2p as a2p  # noqa: E402
import data  # noqa: E402
import geometry as G  # noqa: E402
from encoder import PanoEncoder, normalize_tiles  # noqa: E402
from losses import combined_loss  # noqa: E402

MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"
ERP_H, ERP_W = 1024, 2048
TILE, OVERLAP = 512, 0.25
DEVICE = "cuda"
CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ckpt_ssl_lora")

EPOCHS = int(os.environ.get("EPOCHS", 3))
CAP_S3D = int(os.environ.get("CAP_S3D", 300))
CAP_S2D = int(os.environ.get("CAP_S2D", 300))
LR = 1e-4

DOMAINS = {                                    # (hfov, pitch_centers)
    "in": (65.0, (-45.0, 0.0, 45.0)),
    "out": (50.0, (0.0,)),
}


def build_geometry(enc, hfov, pitch_centers):
    yaws = a2p.make_yaw_centers_closed_loop(hfov, OVERLAP, start_deg=-180.0)
    k = len(yaws)
    specs, idx = [], {}
    for r, pitch in enumerate(pitch_centers):
        for c, yaw in enumerate(yaws):
            idx[(r, c)] = len(specs); specs.append((yaw, pitch))
    pairs = []
    for r in range(len(pitch_centers)):
        for c in range(k):
            pairs.append((idx[(r, c)], idx[(r, (c + 1) % k)]))            # horizontal wrap
            if r + 1 < len(pitch_centers):
                pairs.append((idx[(r, c)], idx[(r + 1, c)]))              # vertical
    cmaps = [G.render_coordmap(ERP_H, ERP_W, y, p, hfov, TILE) for (y, p) in specs]
    warps, kept = [], []
    for (a, b) in pairs:
        wf = G.warp_field_from_coordmaps(cmaps[a], cmaps[b], enc.patch, hfov, erp_w=ERP_W, dst_stride=3)
        if wf.valid.mean() < 0.05:
            continue
        warps.append((torch.from_numpy(wf.grid).to(DEVICE), torch.from_numpy(wf.valid).to(DEVICE),
                      torch.from_numpy(wf.weight).to(DEVICE)))
        kept.append((a, b))
    return {"specs": specs, "pairs": kept, "warps": warps, "hfov": hfov}


def load_erp(f, kind):
    if kind == "out":                                                    # densepass band -> padded ERP
        sh = round(400 * ERP_W / 2048)
        img = np.array(Image.open(f).convert("RGB").resize((ERP_W, sh), Image.BILINEAR))
        top = (ERP_H - sh) // 2
        erp = np.zeros((ERP_H, ERP_W, 3), np.uint8); erp[top:top + sh] = img
        return erp
    return np.array(Image.open(f).convert("RGB").resize((ERP_W, ERP_H), Image.BILINEAR))


def render_tiles(erp_np, specs, hfov):
    ts = [torch.from_numpy(np.asarray(a2p.erp_to_pinhole_tile(erp_np, y, p, hfov, TILE))).float().permute(2, 0, 1) / 255.0
          for (y, p) in specs]
    return torch.stack(ts, 0)


def erank(feat):
    x = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
    x = x - x.mean(0, keepdim=True)
    cov = (x.T @ x) / (x.shape[0] - 1)
    ev = torch.linalg.eigvalsh(cov).clamp_min(1e-9)
    p = ev / ev.sum()
    return float(torch.exp(-(p * p.log()).sum()))


def build_pool():
    pool = []
    s2d = [f for f in data.list_erps("stanford2d3d") if "5" not in f.split("extracted_data/")[1].split("/")[0]]
    pool += [(f, "in") for f in s2d[:CAP_S2D]]
    pool += [(f, "in") for f in data.list_structured3d(limit=CAP_S3D)]
    dp = data.list_densepass()
    # outdoor has the biggest headroom but few panos (70) with fewer pairs/pano (equator ring);
    # oversample so the indoor-heavy pool doesn't drown the outdoor signal.
    pool += [(f, "out") for f in dp[:int(len(dp) * 0.7)]] * int(os.environ.get("OUT_REPEAT", 3))
    return pool


def main():
    torch.manual_seed(0)
    enc = PanoEncoder(model_id=MODEL, lora_rank=16).to(DEVICE).train()
    geom = {k: build_geometry(enc, hf, pc) for k, (hf, pc) in DOMAINS.items()}
    for k, g in geom.items():
        print(f"geom[{k}] hfov={g['hfov']} tiles={len(g['specs'])} pairs={len(g['pairs'])}", flush=True)
    pool = build_pool()
    n_in = sum(1 for _, k in pool if k == "in"); n_out = len(pool) - n_in
    print(f"pool={len(pool)} (in={n_in} out={n_out}) trainable="
          f"{sum(p.numel() for p in enc.trainable_parameters())/1e6:.3f}M epochs={EPOCHS}", flush=True)

    opt = torch.optim.AdamW(enc.trainable_parameters(), lr=LR)
    total_steps = EPOCHS * len(pool)
    warmup = max(1, total_steps // 4)
    g0 = torch.Generator().manual_seed(0)
    step, t0 = 0, time.time()
    agg = {}
    for ep in range(EPOCHS):
        order = torch.randperm(len(pool), generator=g0).tolist()
        for i in order:
            f, kind = pool[i]
            try:
                erp = load_erp(f, kind)
            except Exception:
                continue
            g = geom[kind]
            tiles = normalize_tiles(render_tiles(erp, g["specs"], g["hfov"]).to(DEVICE))
            student = enc(tiles)
            teacher = enc.teacher(tiles)
            w_warp = min(1.0, step / warmup)
            total = 0.0
            npairs = len(g["pairs"])
            step_sum = {}
            for (a, b), warp in zip(g["pairs"], g["warps"]):
                # gamma=0.04 (~just below DINOv3's natural per-channel std ~0.07): VICReg acts
                # as a collapse FLOOR, not a dominating target, so warp-equivariance drives learning
                # (distill anchor to the full-rank teacher is the primary anti-collapse guard).
                loss, comps = combined_loss(student[a:a + 1], student[b:b + 1],
                                            teacher[a:a + 1], teacher[b:b + 1], warp,
                                            w_warp=w_warp, gamma=0.04)
                total = total + loss
                for kk, vv in comps.items():
                    step_sum[kk] = step_sum.get(kk, 0.0) + vv.item()
            total = total / npairs
            opt.zero_grad(); total.backward(); opt.step()
            for kk, vv in step_sum.items():                              # accumulate per-step MEAN
                agg[kk] = agg.get(kk, 0.0) + vv / npairs
            step += 1
            if step % 50 == 0:
                er = erank(student.detach()); ert = erank(teacher.detach())
                msg = " ".join(f"{k}={v/50:.3f}" for k, v in agg.items())
                print(f"ep{ep} step{step}/{total_steps} w_warp={w_warp:.2f} erank={er:.1f}/{ert:.1f} {msg} "
                      f"({(time.time()-t0)/step:.2f}s/it)", flush=True)
                agg = {}

    os.makedirs(CKPT, exist_ok=True)
    enc.backbone.save_pretrained(CKPT)
    print(f"saved adapter -> {CKPT}", flush=True)


if __name__ == "__main__":
    main()
