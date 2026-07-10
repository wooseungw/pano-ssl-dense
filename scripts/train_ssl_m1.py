"""M1 semantic-identity SSL: overlap semantic-code agreement (SI-SSL, docs §2-3).

Extends the E2P-overlap recipe (train_ssl.py) with a SwAV-style code head: two views
of the same ray must agree on a shared PROTOTYPE CODE (semantic identity), not just on
raw features (geometric identity). Staged single run on a fresh LoRA:

  Stage A (0..25%%):  warp ramps 0->1, full distill — current recipe, features settle.
  Stage B (25%%..):   w_code ramps 0->1 (25..50%%); token-distill anneals 1->0.1
                      (25..75%%) so the code loss may RESTRUCTURE features while the
                      relational Gram anchor keeps inter-region structure (anti-forget).

Differences vs train_ssl.py (deliberate):
  * warp fields built in BOTH directions per adjacent pair -> symmetric swapped
    prediction L_code = (A->B) + (B->A) over the doubled pair list.
  * distill + VICReg computed ONCE per step over the full tile batch (equivalent to
    the old per-pair average up to tile-degree weighting, and cheaper).
  * sinkhorn targets = stop-grad student codes (design option i), balanced over ALL
    tiles' tokens of the pano jointly.

Anti-collapse: sinkhorn balancing + VICReg floor (gamma=0.04) + prototype
normalization per step + monitors (code perplexity, erank).

Saves LoRA adapter -> runs/ckpt_ssl_m1/ and the code head -> runs/ckpt_ssl_m1/code_head.pt.
Run: CUDA_VISIBLE_DEVICES=0 conda run -n pano python scripts/train_ssl_m1.py
Smoke: SMOKE_STEPS=40 CUDA_VISIBLE_DEVICES=0 conda run -n pano python scripts/train_ssl_m1.py
"""
from __future__ import annotations

import os
import sys
import time
from typing import Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import geometry as G  # noqa: E402
import train_ssl as T  # noqa: E402
from encoder import CodeHead, PanoEncoder, normalize_tiles  # noqa: E402
from losses import code_swap_loss, distill_loss, sinkhorn, vicreg_var_cov, warp_equivariance_loss  # noqa: E402

DEVICE = "cuda"
CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "ckpt_ssl_m1")

EPOCHS = int(os.environ.get("EPOCHS", 3))
SMOKE_STEPS = int(os.environ.get("SMOKE_STEPS", 0))     # >0: cap total steps (smoke run)
LR = 1e-4
K_PROTO = int(os.environ.get("K_PROTO", 512))
PROJ_DIM = 256
TAU_S = 0.1                                             # student softmax temperature
SK_EPS = 0.05                                           # sinkhorn epsilon
GAMMA = 0.04                                            # VICReg floor (see train_ssl.py)
TOK_FLOOR = 0.1                                         # annealed token-distill lower bound
LOG_EVERY = 10 if SMOKE_STEPS else 50                   # smoke runs are short — log densely


def build_geometry_bidir(enc: PanoEncoder, hfov: float, pitch_centers,
                         footprint_safe: bool = False,
                         sub_patch: int | None = None) -> dict:
    """Same tiles/adjacency as train_ssl.build_geometry, but warp fields in BOTH
    directions per pair (symmetric swapped prediction needs A->B and B->A)."""
    yaws = T.a2p.make_yaw_centers_closed_loop(hfov, T.OVERLAP, start_deg=-180.0)
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
    cmaps = None
    if not footprint_safe:
        cmaps = [G.render_coordmap(T.ERP_H, T.ERP_W, y, p, hfov, T.TILE) for (y, p) in specs]
    warps, sub_warps, kept = [], [], []
    for (a, b) in pairs:
        for (src, dst) in ((a, b), (b, a)):
            if footprint_safe:
                ya, pa = specs[src]
                yb, pb = specs[dst]
                wf = G.warp_field_from_homography(
                    hfov, T.TILE, ya, pa, yb, pb, enc.patch, footprint=True)
            else:
                wf = G.warp_field_from_coordmaps(cmaps[src], cmaps[dst], enc.patch, hfov,
                                                 erp_w=T.ERP_W, dst_stride=3)
            if wf.valid.mean() < 0.05:
                continue
            warps.append((torch.from_numpy(wf.grid).to(DEVICE),
                          torch.from_numpy(wf.valid).to(DEVICE),
                          torch.from_numpy(wf.weight).to(DEVICE)))
            if sub_patch is not None:
                ya, pa = specs[src]
                yb, pb = specs[dst]
                swf = G.warp_field_from_homography(
                    hfov, T.TILE, ya, pa, yb, pb, sub_patch, footprint=True)
                sub_warps.append((torch.from_numpy(swf.grid).to(DEVICE),
                                  torch.from_numpy(swf.valid).to(DEVICE),
                                  torch.from_numpy(swf.weight).to(DEVICE)))
            kept.append((src, dst))
    return {"specs": specs, "pairs": kept, "warps": warps,
            "sub_warps": sub_warps, "hfov": hfov,
            "footprint_safe": footprint_safe}


def schedule(step: int, total: int) -> Tuple[float, float, float]:
    """(w_warp, w_code, w_tok) — Stage A then Stage B (docs §3.4)."""
    f = step / max(1, total)
    w_warp = min(1.0, f / 0.25)
    w_code = 0.0 if f < 0.25 else min(1.0, (f - 0.25) / 0.25)
    w_tok = 1.0 if f < 0.25 else max(TOK_FLOOR, 1.0 - (1.0 - TOK_FLOOR) * (f - 0.25) / 0.5)
    return w_warp, w_code, w_tok


@torch.no_grad()
def code_targets(s_code: torch.Tensor) -> torch.Tensor:
    """(T,K,Gh,Gw) student scores -> balanced sinkhorn codes, jointly over all tiles."""
    t, k, gh, gw = s_code.shape
    q = sinkhorn(s_code.detach().permute(0, 2, 3, 1).reshape(-1, k), eps=SK_EPS)
    return q.reshape(t, gh, gw, k).permute(0, 3, 1, 2)


@torch.no_grad()
def code_stats(s_code: torch.Tensor) -> Tuple[float, float]:
    """(usage perplexity, mean confidence) of the STUDENT softmax — the informative
    collapse monitor. (Sinkhorn targets are balanced BY CONSTRUCTION, so their
    perplexity is always ~K and detects nothing.) perplexity -> 1 = code collapse;
    confidence -> 1 = assignments sharpening (healthy as training progresses)."""
    p = torch.softmax(s_code / TAU_S, dim=1)                 # (T,K,Gh,Gw)
    usage = p.mean(dim=(0, 2, 3)).clamp_min(1e-9)
    usage = usage / usage.sum()
    perp = float(torch.exp(-(usage * usage.log()).sum()))
    conf = float(p.max(dim=1).values.mean())
    return perp, conf


def main() -> None:
    torch.manual_seed(0)
    enc = PanoEncoder(model_id=T.MODEL, lora_rank=16).to(DEVICE).train()
    code_head = CodeHead(enc.dim, proj_dim=PROJ_DIM, n_proto=K_PROTO).to(DEVICE).train()
    geom = {k: build_geometry_bidir(enc, hf, pc) for k, (hf, pc) in T.DOMAINS.items()}
    for k, g in geom.items():
        print(f"geom[{k}] hfov={g['hfov']} tiles={len(g['specs'])} dir-pairs={len(g['pairs'])}", flush=True)
    pool = T.build_pool()
    n_in = sum(1 for _, kk in pool if kk == "in")
    n_lora = sum(p.numel() for p in enc.trainable_parameters())
    n_head = sum(p.numel() for p in code_head.parameters())
    total_steps = SMOKE_STEPS if SMOKE_STEPS else EPOCHS * len(pool)
    print(f"pool={len(pool)} (in={n_in} out={len(pool) - n_in}) lora={n_lora/1e6:.3f}M "
          f"code_head={n_head/1e6:.3f}M (K={K_PROTO}) steps={total_steps}", flush=True)

    opt = torch.optim.AdamW(list(enc.trainable_parameters()) + list(code_head.parameters()), lr=LR)
    g0 = torch.Generator().manual_seed(0)
    step, t0, agg = 0, time.time(), {}
    done = False
    for ep in range(EPOCHS):
        if done:
            break
        order = torch.randperm(len(pool), generator=g0).tolist()
        for i in order:
            f, kind = pool[i]
            try:
                erp = T.load_erp(f, kind)
            except Exception:
                continue
            g = geom[kind]
            tiles = normalize_tiles(T.render_tiles(erp, g["specs"], g["hfov"]).to(DEVICE))
            student = enc(tiles)                                         # (T,D,Gh,Gw)
            teacher = enc.teacher(tiles)
            code_head.normalize_prototypes()
            s_code = code_head(student)                                  # (T,K,Gh,Gw)
            q = code_targets(s_code)

            w_warp, w_code, w_tok = schedule(step, total_steps)
            l_warp = student.new_zeros(())
            l_code = student.new_zeros(())
            for (a, b), warp in zip(g["pairs"], g["warps"]):
                l_warp = l_warp + warp_equivariance_loss(student[a:a + 1], student[b:b + 1], *warp)
                if w_code > 0:
                    l_code = l_code + code_swap_loss(s_code[a:a + 1], q[b:b + 1], *warp, tau_s=TAU_S)
            npair = len(g["pairs"])
            l_warp, l_code = l_warp / npair, l_code / npair
            tok, rel = distill_loss(student, teacher)
            var, cov = vicreg_var_cov(student, gamma=GAMMA)
            total = w_warp * l_warp + w_code * l_code + w_tok * tok + rel + 25.0 * var + cov

            opt.zero_grad(); total.backward(); opt.step()
            comps = {"warp": l_warp, "code": l_code, "tok": tok, "rel": rel,
                     "var": var, "cov": cov, "total": total}
            for kk, vv in comps.items():
                agg[kk] = agg.get(kk, 0.0) + float(vv.detach())
            step += 1
            if step % LOG_EVERY == 0:
                er, ert = T.erank(student.detach()), T.erank(teacher.detach())
                perp, conf = code_stats(s_code.detach())
                msg = " ".join(f"{kk}={vv/LOG_EVERY:.3f}" for kk, vv in agg.items())
                print(f"ep{ep} step{step}/{total_steps} w=[warp {w_warp:.2f} code {w_code:.2f} "
                      f"tok {w_tok:.2f}] erank={er:.1f}/{ert:.1f} perp={perp:.0f}/{K_PROTO} "
                      f"conf={conf:.2f} {msg} ({(time.time()-t0)/step:.2f}s/it)", flush=True)
                agg = {}
            if step >= total_steps:
                done = True
                break

    os.makedirs(CKPT, exist_ok=True)
    enc.backbone.save_pretrained(CKPT)
    torch.save({"state_dict": code_head.state_dict(), "dim": enc.dim,
                "proj_dim": PROJ_DIM, "n_proto": K_PROTO}, os.path.join(CKPT, "code_head.pt"))
    print(f"saved adapter + code head -> {CKPT}", flush=True)


if __name__ == "__main__":
    main()
