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
from collections import deque
from concurrent.futures import ThreadPoolExecutor

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
CKPT = os.environ.get("CKPT_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "ckpt_ssl_lora"))

EPOCHS = int(os.environ.get("EPOCHS", 3))
CAP_S3D = int(os.environ.get("CAP_S3D", 300))
CAP_S2D = int(os.environ.get("CAP_S2D", 300))
LR = 1e-4
LORA_RANK = int(os.environ.get("LORA_RANK", 16))                 # scale LoRA capacity here
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", 2 * LORA_RANK))    # alpha=2r keeps the update scale as r grows
LORA_TARGETS = os.environ.get("LORA_TARGETS", "qv")              # "qv" (baseline) or "all" (attn+MLP linears)
# "modules > rank" (LoRA Learns Less/Forgets Less, arXiv:2405.09673): widening beats deeper rank.
_LORA_TARGET_SETS = {"qv": None,
                     "all": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]}

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


def ram_avail_gb():                                # watch RAM: bounded prefetch must keep this flat
    with open("/proc/meminfo") as fh:
        for ln in fh:
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) / 1024 / 1024
    return -1.0


def build_pool():
    # POOL_PIN=<file>: freeze the pool to a snapshot (datasets are still downloading —
    # a pinned list keeps multi-config comparisons internally valid as data lands).
    pin = os.environ.get("POOL_PIN")
    if pin and os.path.exists(pin):
        with open(pin) as fh:
            pool = [tuple(ln.rstrip("\n").split("\t")) for ln in fh if ln.strip()]
        print(f"pool pinned from {pin} ({len(pool)} entries)", flush=True)
        return pool
    pool = []
    s2d = [f for f in data.list_erps("stanford2d3d") if "5" not in f.split("extracted_data/")[1].split("/")[0]]
    pool += [(f, "in") for f in s2d[:CAP_S2D]]
    pool += [(f, "in") for f in data.list_structured3d(limit=CAP_S3D)]
    dp = data.list_densepass()
    # outdoor has the biggest headroom but few panos (70) with fewer pairs/pano (equator ring);
    # oversample so the indoor-heavy pool doesn't drown the outdoor signal.
    pool += [(f, "out") for f in dp[:int(len(dp) * 0.7)]] * int(os.environ.get("OUT_REPEAT", 3))
    if pin:
        with open(pin, "w") as fh:
            fh.writelines(f"{p}\t{k}\n" for p, k in pool)
        print(f"pool snapshot written -> {pin}", flush=True)
    return pool


def main():
    torch.manual_seed(0)
    enc = PanoEncoder(model_id=MODEL, lora_rank=LORA_RANK, lora_alpha=LORA_ALPHA,
                      lora_targets=_LORA_TARGET_SETS.get(LORA_TARGETS)).to(DEVICE).train()
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

    def prep(i):                                   # threaded CPU load+render (else GPU starves)
        f, kind = pool[i]
        try:
            erp = load_erp(f, kind)
        except Exception:
            return None
        return kind, render_tiles(erp, geom[kind]["specs"], geom[kind]["hfov"])

    NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 6))
    PREFETCH = NUM_WORKERS * 2                        # BOUNDED lookahead. ex.map() eager-submits the
    ex = ThreadPoolExecutor(max_workers=NUM_WORKERS)  # entire pool and buffers completed results in
    for ep in range(EPOCHS):                          # order -> unbounded RAM at 21.8k (OOM-killed
        order = torch.randperm(len(pool), generator=g0).tolist()  # ~step250). Sliding window fixes it.
        it = iter(order)
        inflight = deque()
        for _ in range(PREFETCH):                     # prime the window
            try:
                inflight.append(ex.submit(prep, next(it)))
            except StopIteration:
                break
        while inflight:
            fut = inflight.popleft()
            try:                                      # refill as we consume -> window stays bounded
                inflight.append(ex.submit(prep, next(it)))
            except StopIteration:
                pass
            item = fut.result()
            if item is None:
                continue
            kind, tiles_cpu = item
            g = geom[kind]
            tiles = normalize_tiles(tiles_cpu.to(DEVICE))
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
                      f"({(time.time()-t0)/step:.2f}s/it ram={ram_avail_gb():.0f}GB)", flush=True)
                agg = {}

    os.makedirs(CKPT, exist_ok=True)
    enc.backbone.save_pretrained(CKPT)
    print(f"saved adapter -> {CKPT}", flush=True)
    slug = os.environ.get("RUN_SLUG")
    if slug:                                       # reproducible runs/<stamp>_<slug>/ artifact (config + weights)
        import runlog
        run = runlog.create_run(slug, {
            "model": MODEL, "cap_s3d": CAP_S3D, "cap_s2d": CAP_S2D,
            "out_repeat": int(os.environ.get("OUT_REPEAT", 3)),
            "epochs": EPOCHS, "lr": LR, "tile": TILE, "overlap": OVERLAP,
            "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA, "lora_targets": LORA_TARGETS,
            "erp": [ERP_H, ERP_W], "pool": len(pool), "final_step": step,
            "trainable_M": round(sum(p.numel() for p in enc.trainable_parameters()) / 1e6, 3),
            "domains": {k: {"hfov": hf, "pitch": list(pc)} for k, (hf, pc) in DOMAINS.items()},
            "ckpt": CKPT})
        enc.backbone.save_pretrained(os.path.join(run, "weights"))
        print(f"saved run -> {run}", flush=True)
        try:                                       # default train-time viz (opt out: TRAIN_VIZ=0)
            import train_viz
            train_viz.emit_train_viz(run, CKPT)
        except Exception as e:
            print(f"train-viz skipped: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
