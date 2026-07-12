"""SPHERE-COMPLETION SSL (P3 axis) — gated by scripts/diag_context_headroom.py (HEADROOM, 2026-07-11).

The licensed bet: a LoRA student whose pooled tile features make a FIXED-CLASS linear completer
predict a hidden tile's FROZEN feature from ZERO-OVERLAP sphere context better than frozen features
do (held-out frozen-ridge floor: ccos 0.496 indoor). This is a CAPABILITY objective (scene-closure /
amodal completion), never a per-tile accuracy claim (docs/CAN_SSL_RAISE_ACCURACY.md iron law 1).

Guards (pre-registered, one per failure-analysis root cause):
  L (locus)  — the completion head is EXACTLY the gate's linear class [ctx_mean, f_near, geom5]->D,
               so the head cannot absorb the task; the encoder is the only free capacity. A CONTROL
               head is co-trained on the FROZEN teacher's pooled features with the same loss/budget:
               delta(student_ccos - control_ccos) on held-out panos IS the encoder contribution.
  A (erosion)— dominant distill anchor (token+Gram) on a random tile subset every step
               (anchor-strength thesis, SEMANTIC_IDENTITY_SSL.md §12: L_ANCHOR=1.0 >> L_COMP=0.25),
               plus online token-drift instrumentation.
  B (M2)     — the target is a capability (completion), not an ensemble-average distilled into a
               single view; nothing here claims single-view accuracy.

  L = L_ANCHOR * (tok + gram)                       # dominant, erosion defense
    + L_COMP   * (1 - ccos(head(student ctx), frozen_target))
    + L_COMP   * (1 - ccos(head_ctrl(frozen ctx), frozen_target))   # control lane (head params only)

Success criterion: held-out student-lane ccos beats the control lane by a stable margin while token
drift stays small. Post-hoc: full erosion/purity suite before any deployment claim.

Run: OPENCV_IO_ENABLE_OPENEXR=1 CUDA_VISIBLE_DEVICES=1 conda run -n pano python scripts/train_ssl_completion.py
Knobs: EPOCHS(15) BATCH(2) LORA_RANK(16) L_ANCHOR(1.0) L_COMP(0.25) M_TGT(4) ANCH_TILES(6)
       EXCL(85) VAL_PANOS(16) VAL_EVERY(1) LR(1e-4) SMOKE_STEPS(0)
"""
from __future__ import annotations

import math
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_ssl as TS  # noqa: E402
import anyres_e2p as a2p  # noqa: E402
from encoder import PanoEncoder, normalize_tiles  # noqa: E402
from losses import distill_loss  # noqa: E402

DEVICE = "cuda"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.environ.get("CKPT_DIR", os.path.join(ROOT, "runs", "ckpt_ssl_completion"))
MODEL = TS.MODEL
EPOCHS = int(os.environ.get("EPOCHS", 15))
BATCH = int(os.environ.get("BATCH", 2))
LORA_RANK = int(os.environ.get("LORA_RANK", 16))
LR = float(os.environ.get("LR", 1e-4))
L_ANCHOR = float(os.environ.get("L_ANCHOR", 1.0))
L_COMP = float(os.environ.get("L_COMP", 0.25))
M_TGT = int(os.environ.get("M_TGT", 4))
ANCH_TILES = int(os.environ.get("ANCH_TILES", 6))
EXCL = float(os.environ.get("EXCL", 85.0))
VAL_PANOS = int(os.environ.get("VAL_PANOS", 16))
VAL_EVERY = int(os.environ.get("VAL_EVERY", 1))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 6))
SMOKE = int(os.environ.get("SMOKE_STEPS", 0))
SEED = int(os.environ.get("SEED", 0))
LOG_EVERY = 5 if SMOKE else 50
CHUNK = 8
MU_MOM = 0.99

os.environ.setdefault("POOL_PIN", os.path.join(ROOT, "configs", "pool_pin_20260702.tsv"))


def build_specs(hfov, pitches):
    yaws = a2p.make_yaw_centers_closed_loop(hfov, getattr(TS, "OVERLAP", 0.25), start_deg=-180.0)
    return [(y, p) for p in pitches for y in yaws]


def unit_dir(yaw_deg, pitch_deg):
    y, p = np.deg2rad(yaw_deg), np.deg2rad(pitch_deg)
    return np.array([np.cos(p) * np.cos(y), np.sin(p), np.cos(p) * np.sin(y)])


def context_plan(specs):
    """Per-target: (allowed indices beyond EXCL, nearest-allowed index, geom5 vector)."""
    dirs = np.stack([unit_dir(y, p) for (y, p) in specs])
    ang = np.rad2deg(np.arccos(np.clip(dirs @ dirs.T, -1.0, 1.0)))
    plan = []
    for t in range(len(specs)):
        allowed = ang[t] > EXCL
        allowed[t] = False
        idx = np.where(allowed)[0]
        if len(idx) < 2:
            plan.append(None)
            continue
        near = int(idx[np.argmin(ang[t][idx])])
        dyaw = np.deg2rad(specs[t][0] - specs[near][0])
        g = np.array([np.sin(dyaw), np.cos(dyaw), specs[t][1] / 45.0, specs[near][1] / 45.0,
                      ang[t][near] / 180.0], dtype=np.float32)
        plan.append((torch.from_numpy(idx), near, torch.from_numpy(g)))
    return plan


GEO = {}
for k, (hf, pc) in TS.DOMAINS.items():
    sp = build_specs(hf, pc)
    GEO[k] = (hf, sp, context_plan(sp))


class Completer(nn.Module):
    """EXACTLY the gate's linear class: [ctx_mean(D), f_near(D), geom(5)] -> D. Starved by design."""

    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(2 * d + 5, d)

    def forward(self, ctx_mean, f_near, g):
        return self.lin(torch.cat([ctx_mean, f_near, g], dim=-1))


def comp_loss(head, pooled, plan, targets_t, teacher_pooled, mu):
    """pooled: (T,D) lane features (student or detached teacher). Returns (loss, ccos, n)."""
    losses, coss = [], []
    for t in targets_t:
        pl = plan[t]
        if pl is None:
            continue
        idx, near, g = pl
        ctx = pooled[idx.to(pooled.device)].mean(0)
        pred = head(ctx, pooled[near], g.to(pooled.device))
        targ = teacher_pooled[t]
        c = F.cosine_similarity((pred - mu)[None], (targ - mu)[None]).squeeze(0)
        losses.append(1.0 - c)
        coss.append(c.detach())
    if not losses:
        z = pooled.new_zeros(())
        return z, z, 0
    return torch.stack(losses).mean(), torch.stack(coss).mean(), len(losses)


def main():
    torch.manual_seed(SEED)
    enc = PanoEncoder(model_id=MODEL, lora_rank=LORA_RANK).to(DEVICE)
    enc.train()
    D = enc.dim
    head = Completer(D).to(DEVICE)
    head_ctrl = Completer(D).to(DEVICE)
    params = enc.trainable_parameters() + list(head.parameters()) + list(head_ctrl.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.05)

    pool = TS.build_pool()
    val_pool, tr_pool = pool[-VAL_PANOS:], pool[:-VAL_PANOS]
    steps_per_ep = max(1, len(tr_pool) // BATCH)
    total_steps = SMOKE if SMOKE else EPOCHS * steps_per_ep
    n_tr = sum(p.numel() for p in enc.trainable_parameters()) / 1e6
    print(f"SPHERE-COMPLETION: pool tr={len(tr_pool)} val={len(val_pool)} lora={n_tr:.2f}M "
          f"L_ANCHOR={L_ANCHOR} L_COMP={L_COMP} M_TGT={M_TGT} EXCL={EXCL} epochs={EPOCHS} "
          f"batch={BATCH} steps={total_steps}", flush=True)

    mu = torch.zeros(D, device=DEVICE)
    mu_init = False
    g0 = torch.Generator().manual_seed(SEED)

    def prep(i):
        f, kind = tr_pool[i]
        try:
            erp = TS.load_erp(f, kind)
        except Exception:
            return None
        hf, specs, _ = GEO[kind]
        return kind, TS.render_tiles(erp, specs, hf)

    def forward_pano(kind, tiles_cpu, grad=True):
        """Return (student_dense selected-subset, student_pooled(T,D), teacher_dense subset,
        teacher_pooled(T,D), anchor tile indices)."""
        tiles = tiles_cpu.to(DEVICE)
        Tn = tiles.shape[0]
        anch = torch.randperm(Tn, generator=g0)[:ANCH_TILES]
        x = normalize_tiles(tiles)
        s_pool, s_anch = [], []
        ctxmgr = torch.enable_grad if grad else torch.no_grad
        with ctxmgr():
            for i in range(0, Tn, CHUNK):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    fs = enc(x[i:i + CHUNK]).float()
                s_pool.append(fs.mean(dim=(-2, -1)))
                for j, ti in enumerate(range(i, min(i + CHUNK, Tn))):
                    if (anch == ti).any():
                        s_anch.append((ti, fs[j]))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            t_dense = torch.cat([enc.teacher(x[i:i + CHUNK]).float()
                                 for i in range(0, Tn, CHUNK)], 0)
        t_pool = t_dense.mean(dim=(-2, -1))
        s_dense_anch = torch.stack([f for (_, f) in sorted(s_anch, key=lambda z: z[0])])
        t_dense_anch = t_dense[sorted(ti for (ti, _) in s_anch)]
        return torch.cat(s_pool, 0), s_dense_anch, t_dense_anch, t_pool

    @torch.no_grad()
    def validate():
        enc.eval()
        cs, cc, n = 0.0, 0.0, 0
        for f, kind in val_pool:
            try:
                erp = TS.load_erp(f, kind)
            except Exception:
                continue
            hf, specs, plan = GEO[kind]
            tiles = TS.render_tiles(erp, specs, hf)
            s_pool, _, _, t_pool = forward_pano(kind, tiles, grad=False)
            tgts = [t for t in range(len(specs)) if plan[t] is not None]
            _, c_s, k1 = comp_loss(head, s_pool, plan, tgts, t_pool, mu)
            _, c_c, k2 = comp_loss(head_ctrl, t_pool, plan, tgts, t_pool, mu)
            if k1:
                cs += float(c_s) * k1; cc += float(c_c) * k2; n += k1
        enc.train()
        return (cs / max(n, 1), cc / max(n, 1), n)

    step, t0 = 0, time.time()
    agg = {}
    ex = ThreadPoolExecutor(max_workers=NUM_WORKERS)
    best_delta = -1e9
    for ep in range(EPOCHS):
        order = torch.randperm(len(tr_pool), generator=g0).tolist()
        it = iter(order)
        inflight = deque()
        for _ in range(NUM_WORKERS * 2):
            try:
                inflight.append(ex.submit(prep, next(it)))
            except StopIteration:
                break
        nb = 0
        opt.zero_grad()
        while inflight:
            fut = inflight.popleft()
            try:
                inflight.append(ex.submit(prep, next(it)))
            except StopIteration:
                pass
            item = fut.result()
            if item is None:
                continue
            kind, tiles_cpu = item
            _, specs, plan = GEO[kind]
            s_pool, s_da, t_da, t_pool = forward_pano(kind, tiles_cpu, grad=True)
            if not mu_init:
                mu.copy_(t_pool.mean(0)); mu_init = True
            mu.mul_(MU_MOM).add_(t_pool.mean(0), alpha=1.0 - MU_MOM)

            cand = [t for t in range(len(specs)) if plan[t] is not None]
            tgts = [cand[i] for i in torch.randperm(len(cand), generator=g0)[:M_TGT].tolist()]
            l_comp, c_s, _ = comp_loss(head, s_pool, plan, tgts, t_pool, mu)
            l_ctrl, c_c, _ = comp_loss(head_ctrl, t_pool, plan, tgts, t_pool, mu)
            tok, rel = distill_loss(s_da, t_da)
            drift = float(1.0 - F.cosine_similarity(
                s_da.permute(0, 2, 3, 1).reshape(-1, D),
                t_da.permute(0, 2, 3, 1).reshape(-1, D)).mean().detach())
            loss = (L_ANCHOR * (tok + rel) + L_COMP * l_comp + L_COMP * l_ctrl) / BATCH
            loss.backward()
            nb += 1
            for kk, vv in (("comp", float(l_comp)), ("ctrl", float(l_ctrl)), ("ccos_s", float(c_s)),
                           ("ccos_c", float(c_c)), ("tok", float(tok)), ("gram", float(rel)),
                           ("drift", drift)):
                agg[kk] = agg.get(kk, 0.0) + vv
            if nb % BATCH == 0:
                for gg in opt.param_groups:
                    gg["lr"] = LR * 0.5 * (1 + math.cos(math.pi * step / max(1, total_steps)))
                opt.step(); opt.zero_grad()
                step += 1
                if step % LOG_EVERY == 0:
                    m = LOG_EVERY * BATCH
                    print(f"ep{ep} step{step}/{total_steps} " +
                          " ".join(f"{k}={v / m:.4f}" for k, v in agg.items()) +
                          f" ({(time.time() - t0) / max(step, 1):.2f}s/it)", flush=True)
                    agg = {}
                if SMOKE and step >= SMOKE:
                    break
        if SMOKE and step >= SMOKE:
            c_s, c_c, n = validate()
            print(f"[smoke-val] student={c_s:.4f} control={c_c:.4f} delta={c_s - c_c:+.4f} n={n}", flush=True)
            break
        if (ep + 1) % VAL_EVERY == 0:
            c_s, c_c, n = validate()
            delta = c_s - c_c
            tag = ""
            if delta > best_delta:
                best_delta = delta
                d = os.path.join(CKPT, "best")
                os.makedirs(d, exist_ok=True)
                enc.backbone.save_pretrained(d)
                torch.save({"head": head.state_dict(), "head_ctrl": head_ctrl.state_dict(),
                            "mu": mu.cpu(), "epoch": ep, "val_student": c_s, "val_control": c_c},
                           os.path.join(d, "heads.pt"))
                tag = " [best->saved]"
            print(f"[val ep{ep + 1}] student_ccos={c_s:.4f} control_ccos={c_c:.4f} "
                  f"delta={delta:+.4f} (encoder contribution) n={n}{tag}", flush=True)

    if not SMOKE:
        d = os.path.join(CKPT, "last")
        os.makedirs(d, exist_ok=True)
        enc.backbone.save_pretrained(d)
        torch.save({"head": head.state_dict(), "head_ctrl": head_ctrl.state_dict(), "mu": mu.cpu()},
                   os.path.join(d, "heads.pt"))
        print(f"saved final -> {d}", flush=True)


if __name__ == "__main__":
    main()
