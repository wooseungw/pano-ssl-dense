"""F-3 Pano-JEPA: masked-view prediction with an EMA teacher (docs §11).

The canonical formulation (docs §0) promoted to the objective itself: hold out k tiles,
fuse the VISIBLE tiles into an ERP context field (64x128 — the validated sweet spot),
and predict the masked tiles' representations at their exact warp locations. One loss
subsumes geometric identity (prediction sits on the warp correspondence), semantic
identity (prediction in representation space), INTEGRATION (the field is inside the
loss), and whole-panorama coherence (context = everything else).

Anchor: EMA teacher initialized from the TC3 adapter — a slowly evolving reference,
NOT a permanent freeze (the M1-refuted no-anchor and the ceiling-bound hard-freeze are
the two ends this sits between). Anti-collapse: EMA asymmetry + predictor (JEPA/BYOL
recipe) + VICReg floor + drift/erank monitors. Erosion is adjudicated post-hoc by the
laundering-proof eval suite (ADAPTER=runs/ckpt_ssl_f3 / runs/ckpt_ssl_f3_ema).

  L = w_pred · mean_masked-cells w_obl·(1 − cos(predictor(field(Hp), geo), sg[EMA(x_masked)]))
      + 25·L_var + L_cov                                      (VICReg floor, student)

Pool: pinned snapshot (configs/pool_pin_20260702.tsv) for comparability with TC3.
Run:   CUDA_VISIBLE_DEVICES=<n> conda run -n pano python scripts/train_ssl_f3.py
Smoke: SMOKE_STEPS=10 BATCH=2 CUDA_VISIBLE_DEVICES=<n> ... python scripts/train_ssl_f3.py
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import probe_seg_dinov3 as P  # noqa: E402
import diag_seam as D  # noqa: E402
import geometry as G  # noqa: E402
import train_ssl as T  # noqa: E402
import runlog  # noqa: E402
from encoder import PanoEncoder, ema_update, normalize_tiles  # noqa: E402
from fusion import scatter_mean_field  # noqa: E402
from losses import vicreg_var_cov  # noqa: E402

DEVICE = "cuda"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "runs", "ckpt_ssl_f3")
INIT = os.environ.get("F3_INIT", os.path.join(ROOT, "runs", "ckpt_ssl_tc3"))

EPOCHS = int(os.environ.get("EPOCHS", 2))
SMOKE_STEPS = int(os.environ.get("SMOKE_STEPS", 0))
BATCH = int(os.environ.get("BATCH", 4))                 # panos per optimizer step (grad accum)
WORKERS = int(os.environ.get("NUM_WORKERS", BATCH))     # parallel load+render threads
AMP = os.environ.get("AMP", "1") == "1"                 # bf16 encoder forwards (recorded in config)
LR = 5e-5
EMA_M = float(os.environ.get("EMA_M", 0.996))
W_PRED = 1.0
GAMMA = 0.04                                            # VICReg floor (recipe constant)
MASK_K = {"in": (2, 6), "out": (1, 3)}                  # masked tiles per pano, by domain
LOG_EVERY = 5 if SMOKE_STEPS else 50

os.environ.setdefault("POOL_PIN", os.path.join(ROOT, "configs", "pool_pin_20260702.tsv"))


class Predictor(nn.Module):
    """View-conditioned JEPA predictor: fused context + masked-view geometry -> feature."""

    def __init__(self, dim: int = 768, geo_dim: int = 4, hidden: int = 1024):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim + geo_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, dim))

    def forward(self, ctx: torch.Tensor, geo: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([ctx, geo], dim=-1))


def build_domain(enc: PanoEncoder, hfov: float, pitch_centers) -> Dict:
    """Tile specs + per-tile (cid into the 64x128 ERP field, geo feats, obliquity w)."""
    yaws = T.a2p.make_yaw_centers_closed_loop(hfov, T.OVERLAP, start_deg=-180.0)
    specs = [(y, p) for p in pitch_centers for y in yaws]
    D.TILE, D.HFOV = T.TILE, hfov                        # coord_grid reads module globals
    gh = gw = T.TILE // enc.patch
    ii, jj = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
    r = np.sqrt((ii - (gh - 1) / 2) ** 2 + (jj - (gw - 1) / 2) ** 2).reshape(-1)
    r = (r / r.max()).astype(np.float32)
    col = ((jj + 0.5) * T.TILE / gw).reshape(-1).astype(np.float32)
    row = ((ii + 0.5) * T.TILE / gh).reshape(-1).astype(np.float32)
    wobl = G._offaxis_cos(col, row, T.TILE, hfov).astype(np.float32)
    cids, geos = [], []
    for (yaw, pitch) in specs:
        cid, _ = D.coord_grid((T.ERP_H, T.ERP_W), T.a2p.TilePlan(yaw, pitch), gh, gw)
        cids.append(torch.from_numpy(cid.reshape(-1).astype(np.int64)).to(DEVICE))
        g = np.stack([wobl, r, np.full_like(r, pitch / 45.0), np.ones_like(r)], 1)
        geos.append(torch.from_numpy(g).to(DEVICE))
    ncell = (T.ERP_H // enc.patch) * (T.ERP_W // enc.patch)          # 64x128 = 8192
    return {"hfov": hfov, "specs": specs, "cid": cids, "geo": geos,
            "wobl": torch.from_numpy(wobl).to(DEVICE), "ncell": ncell}


def flat_feats(f: torch.Tensor) -> torch.Tensor:
    """(B, D, gh, gw) -> (B, gh*gw, D)."""
    return f.permute(0, 2, 3, 1).reshape(f.shape[0], -1, f.shape[1])


def pano_loss(enc, tea, predictor, tiles, dom, gen) -> Tuple[torch.Tensor, dict]:
    """Masked-view prediction loss for one pano. tiles: (T,3,512,512) cpu in [0,1]."""
    n_tiles = tiles.shape[0]
    klo, khi = MASK_K["in" if n_tiles > 12 else "out"]
    k = int(torch.randint(klo, khi + 1, (1,), generator=gen))
    perm = torch.randperm(n_tiles, generator=gen).tolist()
    masked, visible = perm[:k], perm[k:]

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
        student = enc(normalize_tiles(tiles[visible].to(DEVICE)))    # grad, (V,D,gh,gw)
        with torch.no_grad():
            target = tea(normalize_tiles(tiles[masked].to(DEVICE)))  # EMA,  (K,D,gh,gw)
    student, target = student.float(), target.float()                # field/predictor/loss fp32

    sf = flat_feats(student).reshape(-1, student.shape[1])           # (V*N, D)
    cid_vis = torch.cat([dom["cid"][t] for t in visible])
    field, counts = scatter_mean_field(cid_vis, sf, dom["ncell"])    # differentiable

    tf = F.normalize(flat_feats(target), dim=-1)                     # (K,N,D)
    l_pred = student.new_zeros(())
    n_terms = 0
    for j, t in enumerate(masked):
        cid_t = dom["cid"][t]
        sel = counts[cid_t] > 0                                      # cells the context covers
        if int(sel.sum()) < 8:
            continue
        pred = F.normalize(predictor(field[cid_t[sel]], dom["geo"][t][sel]), dim=-1)
        cos = (pred * tf[j][sel]).sum(-1)
        w = dom["wobl"][sel]
        l_pred = l_pred + ((1.0 - cos) * w).sum() / w.sum().clamp_min(1.0)
        n_terms += 1
    l_pred = l_pred / max(n_terms, 1)
    var, cov = vicreg_var_cov(student, gamma=GAMMA)
    total = W_PRED * l_pred + 25.0 * var + cov
    return total, {"pred": float(l_pred.detach()), "var": float(var.detach()),
                   "cov": float(cov.detach()), "student": student.detach()}


def main() -> None:
    torch.manual_seed(0)
    enc = PanoEncoder(model_id=T.MODEL, adapter_path=INIT, adapter_trainable=True).to(DEVICE).train()
    tea = PanoEncoder(model_id=T.MODEL, adapter_path=INIT).to(DEVICE).eval()
    n_tr = sum(p.numel() for p in enc.trainable_parameters())
    assert n_tr > 0, "student adapter loaded frozen"
    predictor = Predictor(enc.dim).to(DEVICE).train()
    P.enc_patch = enc.patch

    doms = {kk: build_domain(enc, hf, pc) for kk, (hf, pc) in T.DOMAINS.items()}
    for kk, dm in doms.items():
        print(f"dom[{kk}] hfov={dm['hfov']} tiles={len(dm['specs'])} field={dm['ncell']}", flush=True)
    pool = T.build_pool()
    opt = torch.optim.AdamW(list(enc.trainable_parameters()) + list(predictor.parameters()), lr=LR)
    steps_per_ep = (len(pool) + BATCH - 1) // BATCH
    total_steps = SMOKE_STEPS if SMOKE_STEPS else EPOCHS * steps_per_ep
    print(f"init<-{INIT} lora={n_tr/1e6:.3f}M pred={sum(p.numel() for p in predictor.parameters())/1e6:.2f}M "
          f"pool={len(pool)} batch={BATCH} ema_m={EMA_M} steps={total_steps}", flush=True)

    def prep(i: int):
        """CPU-side load+render for one pano (threaded — the measured bottleneck).
        No RNG here: mask draws stay in the sequential loss loop (reproducible)."""
        fpath, kind = pool[i]
        try:
            erp = T.load_erp(fpath, kind)
        except Exception:
            return None
        dom = doms[kind]
        return T.render_tiles(erp, dom["specs"], dom["hfov"]), dom

    exec_pool = ThreadPoolExecutor(max_workers=WORKERS)
    g0 = torch.Generator().manual_seed(0)
    step, t0, agg, done = 0, time.time(), {}, False
    last_student, last_tiles = None, None
    pbar = tqdm(total=total_steps, desc="f3-jepa", mininterval=10,
                file=sys.stdout, dynamic_ncols=True)
    for ep in range(EPOCHS):
        if done:
            break
        order = torch.randperm(len(pool), generator=g0).tolist()
        for bs in range(0, len(order), BATCH):
            opt.zero_grad()
            n_ok = 0
            for item in exec_pool.map(prep, order[bs:bs + BATCH]):
                if item is None:
                    continue
                tiles, dom = item
                total, comps = pano_loss(enc, tea, predictor, tiles, dom, g0)
                (total / BATCH).backward()
                for kk in ("pred", "var", "cov"):
                    agg[kk] = agg.get(kk, 0.0) + comps[kk] / BATCH
                last_student, last_tiles = comps["student"], tiles
                n_ok += 1
            if n_ok == 0:
                continue
            opt.step()
            ema_update(enc, tea, EMA_M)
            step += 1
            pbar.update(1)
            pbar.set_postfix(pred=f"{comps['pred']:.3f}", refresh=False)
            if step % LOG_EVERY == 0:
                er = T.erank(last_student)
                with torch.no_grad():                                # drift vs ORIGINAL frozen DINOv3
                    frozen = enc.teacher(normalize_tiles(last_tiles[:4].to(DEVICE)))
                    stud4 = enc(normalize_tiles(last_tiles[:4].to(DEVICE)))
                    drift = float(1.0 - F.cosine_similarity(
                        flat_feats(stud4).reshape(-1, enc.dim),
                        flat_feats(frozen).reshape(-1, enc.dim), dim=-1).mean())
                msg = " ".join(f"{kk}={vv/LOG_EVERY:.3f}" for kk, vv in agg.items())
                print(f"ep{ep} step{step}/{total_steps} erank={er:.1f} drift={drift:.3f} {msg} "
                      f"({(time.time()-t0)/step:.2f}s/it, {BATCH} panos/it)", flush=True)
                agg = {}
            if step >= total_steps:
                done = True
                break

    pbar.close()
    os.makedirs(CKPT, exist_ok=True)
    enc.backbone.save_pretrained(CKPT)                               # student adapter
    tea.backbone.save_pretrained(CKPT + "_ema")                      # EMA (JEPA-style eval ckpt)
    torch.save({"state_dict": predictor.state_dict(), "dim": enc.dim},
               os.path.join(CKPT, "predictor.pt"))
    run = runlog.create_run("f3_jepa", {
        "init": INIT, "epochs": EPOCHS, "batch": BATCH, "lr": LR, "ema_m": EMA_M,
        "w_pred": W_PRED, "mask_k": MASK_K, "pool": len(pool),
        "pool_pin": os.environ.get("POOL_PIN"), "steps": total_steps,
        "smoke_steps": SMOKE_STEPS, "amp_bf16": AMP, "num_workers": WORKERS})
    enc.backbone.save_pretrained(os.path.join(run, "weights", "adapter_student"))
    tea.backbone.save_pretrained(os.path.join(run, "weights", "adapter_ema"))
    torch.save({"state_dict": predictor.state_dict(), "dim": enc.dim},
               os.path.join(run, "weights", "predictor.pt"))
    # F-3 is representation-level — downstream seg/depth viz lands here at eval time
    print(f"saved student -> {CKPT}, EMA -> {CKPT}_ema, run -> {run}", flush=True)
    try:                                           # default train-time viz (opt out: TRAIN_VIZ=0)
        import train_viz
        train_viz.emit_train_viz(run, CKPT)
    except Exception as e:
        print(f"train-viz skipped: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
