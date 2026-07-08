"""F-2: learned set-fusion at scale (docs/SEMANTIC_IDENTITY_SSL.md §9.6).

Question: with the encoder FROZEN (per-tile features preserved), can a neural set-fusion
beat the uniform mean — the F-1 winner — now that labeled data is ~90x larger
(Structured3D 21.8k panos with semantic GT) and tiling-config augmentation is free?
Prior negatives (deformable tie @250 panos, scalar-weight axis dead) tested a NARROW
function class at tiny scale; SetFusion is residual + zero-init, so it *starts as* the
mean baseline and can only learn on top.

Protocol (paired): run twice with the same seed/data — FUSION=mean trains only the
linear head on mean-fused features; FUSION=attn trains SetFusion + head jointly.
Group-disjoint scene split. Augmentation: random horizontal ERP roll (lossless yaw
shift, per pano) + hfov config bank {63, 65, 67} (per batch).

Batched (BATCH panos/step, default 8): tiles of the whole batch are encoded in CHUNK-
sized batched forwards (the frozen encoder was the bottleneck at 1 tile/forward).

Run:  FUSION=attn CUDA_VISIBLE_DEVICES=<n> conda run -n pano python scripts/train_fusion_f2.py
      FUSION=mean CUDA_VISIBLE_DEVICES=<n> conda run -n pano python scripts/train_fusion_f2.py
Smoke: SMOKE_STEPS=3 S3D_TRAIN=32 S3D_VAL=6 FUSION=attn ... python scripts/train_fusion_f2.py
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import probe_seg_dinov3 as P  # noqa: E402
import diag_seam as D  # noqa: E402
import geometry as G  # noqa: E402
import runlog  # noqa: E402
from encoder import PanoEncoder, normalize_tiles  # noqa: E402
from fusion import SetFusion, masked_mean, pack_sets  # noqa: E402

DEVICE = P.DEVICE
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FUSION = os.environ.get("FUSION", "attn")               # attn | mean (paired baseline)
ADAPTER = os.environ.get("ENC_ADAPTER", os.path.join(ROOT, "runs", "ckpt_ssl_tc3"))
S3D_TRAIN = int(os.environ.get("S3D_TRAIN", 3000))
S3D_VAL = int(os.environ.get("S3D_VAL", 300))
EPOCHS = int(os.environ.get("EPOCHS", 1))
SMOKE_STEPS = int(os.environ.get("SMOKE_STEPS", 0))
BATCH = int(os.environ.get("BATCH", 8))                 # panos per optimizer step
WORKERS = int(os.environ.get("NUM_WORKERS", BATCH))     # parallel load+render threads
AMP = os.environ.get("AMP", "0") == "1"                 # bf16 encoder forward (default off:
CHUNK = 24                                              # keep paired runs numerically identical)
LR = 5e-4
MAX_COV = 48                                            # measured max coverage is 44 — no truncation
HFOV_BANK = (63.0, 65.0, 67.0)                          # config-bank augmentation
OVERLAP, PMAX = 0.25, 45.0
CKPT = os.path.join(ROOT, "runs", f"ckpt_fusion_f2_{FUSION}")
LOG_EVERY = 2 if SMOKE_STEPS else 25


def build_config(hfov: float):
    """Image-independent per-config geometry: tile plan + per-tile (cid, geo feats)."""
    plan = P.a2p.plan_tiles("band", hfov, hfov, OVERLAP, pmax_deg=PMAX)
    h, w = P.WORK_HW
    gh = gw = D.TILE // P.enc_patch
    ii, jj = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
    r = np.sqrt((ii - (gh - 1) / 2) ** 2 + (jj - (gw - 1) / 2) ** 2).reshape(-1)
    r = (r / r.max()).astype(np.float32)
    col = ((jj + 0.5) * D.TILE / gw).reshape(-1).astype(np.float32)
    row = ((ii + 0.5) * D.TILE / gh).reshape(-1).astype(np.float32)
    wobl = G._offaxis_cos(col, row, D.TILE, hfov).astype(np.float32)
    D.HFOV = hfov                                        # coord_grid reads module globals
    cids, geos = [], []
    for tp in plan:
        cid, _ = D.coord_grid((h, w), tp, gh, gw)
        cids.append(torch.from_numpy(cid.reshape(-1).astype(np.int64)))
        g = np.stack([wobl, r, np.full_like(r, tp.pitch_deg / PMAX), np.ones_like(r)], 1)
        geos.append(torch.from_numpy(g))
    return {"hfov": hfov, "plan": plan, "cid": torch.cat(cids),
            "geo": torch.cat(geos), "ncell": (h // P.enc_patch) * (w // P.enc_patch)}


def render_cfg_tiles(rgb: np.ndarray, cfg) -> torch.Tensor:
    """All tiles of one pano under one config -> (T, 3, 512, 512) in [0,1]."""
    ts = [np.asarray(P.a2p.erp_to_pinhole_tile(rgb, tp.yaw_deg, tp.pitch_deg,
                                               cfg["hfov"], D.TILE)).copy()
          for tp in cfg["plan"]]
    return torch.from_numpy(np.stack(ts)).float().permute(0, 3, 1, 2) / 255.0


@torch.no_grad()
def encode_tiles(enc, tiles: torch.Tensor) -> torch.Tensor:
    """Chunked batched frozen-encoder forward: (T,3,H,W) -> (T, N, D) cpu."""
    outs = []
    for i in range(0, tiles.shape[0], CHUNK):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
            f = P.dense(enc, normalize_tiles(tiles[i:i + CHUNK].to(DEVICE)))  # (b,D,gh,gw)
        f = f.float()                                     # fusion/head stay fp32
        outs.append(f.permute(0, 2, 3, 1).reshape(f.shape[0], -1, f.shape[1]).cpu())
    return torch.cat(outs)


def pano_pack(feat_tnd: torch.Tensor, lab: np.ndarray, cfg):
    """(T,N,D) tile features + label map -> covered packed sets + per-cell GT."""
    f, g, m = pack_sets(cfg["cid"], feat_tnd.reshape(-1, feat_tnd.shape[-1]),
                        cfg["geo"], cfg["ncell"], MAX_COV)
    hf, wf = P.WORK_HW[0] // P.enc_patch, P.WORK_HW[1] // P.enc_patch
    gt = torch.from_numpy(P.label_to_grid(lab, hf, wf).reshape(-1))
    cov = m.any(1)
    return f[cov], g[cov], m[cov], gt[cov]


def fuse(fusion, f, g, m):
    return fusion(f, g, m) if fusion is not None else masked_mean(f, m)


@torch.no_grad()
def evaluate(enc, fusion, head, val_files, cfg):
    """mIoU on all covered cells + split by cross-view feature variance (hi 30%)."""
    if fusion is not None:
        fusion.eval()                                    # no_grad does NOT disable module mode
    preds, gts, vars_ = [], [], []
    for fpath in val_files:
        try:
            rgb, lab = P.load_rgb_label(fpath)
        except Exception:
            continue
        feats = encode_tiles(enc, render_cfg_tiles(rgb, cfg))
        f, g, m, gt = pano_pack(feats, lab, cfg)
        f, g, m = f.to(DEVICE), g.to(DEVICE), m.to(DEVICE)
        logits = head(fuse(fusion, f, g, m))
        mean = masked_mean(f, m)
        dev = ((f - mean.unsqueeze(1)) ** 2).sum(-1).sqrt()                  # (B,K)
        v = (dev * m.float()).sum(1) / m.float().sum(1).clamp_min(1.0)
        preds.append(logits.argmax(1).cpu()); gts.append(gt); vars_.append(v.cpu())
    if fusion is not None:
        fusion.train()
    pred, gt, v = torch.cat(preds), torch.cat(gts), torch.cat(vars_)
    hi = v >= v.quantile(0.7)
    return (P.miou_acc(pred, gt)[0], P.miou_acc(pred[hi], gt[hi])[0],
            P.miou_acc(pred[~hi], gt[~hi])[0])


@torch.no_grad()
def save_val_viz(enc, fusion, head, val_files, cfg, run: str, k: int = 3) -> None:
    """Designated (sorted-first) val samples -> run/viz/ PNGs, GT alongside pred."""
    pal = runlog.seg_palette(P.N_CLASS)
    hf, wf = P.WORK_HW[0] // P.enc_patch, P.WORK_HW[1] // P.enc_patch
    if fusion is not None:
        fusion.eval()
    for i, fpath in enumerate(sorted(val_files)[:k]):
        rgb, lab = P.load_rgb_label(fpath)
        feats = encode_tiles(enc, render_cfg_tiles(rgb, cfg))
        f, g, m = pack_sets(cfg["cid"], feats.reshape(-1, feats.shape[-1]),
                            cfg["geo"], cfg["ncell"], MAX_COV)
        cov = m.any(1)
        logits = head(fuse(fusion, f[cov].to(DEVICE), g[cov].to(DEVICE), m[cov].to(DEVICE)))
        grid = np.full(hf * wf, -1, np.int64)
        grid[cov.numpy()] = logits.argmax(1).cpu().numpy()
        runlog.save_seg_sample(run, "s3d", i, rgb.astype(np.float32) / 255.0,
                               P.label_to_grid(lab, hf, wf), {FUSION: grid.reshape(hf, wf)}, pal)
    if fusion is not None:
        fusion.train()


def main() -> None:
    torch.manual_seed(0)
    P.configure("structured3d"); P.TILE = 512
    D.DATASET, D.OVERLAP, D.TILE = "structured3d", OVERLAP, 512
    enc = PanoEncoder(model_id=P.MODEL, adapter_path=ADAPTER).to(DEVICE).eval()
    P.enc_patch = enc.patch

    allf = P.data.list_structured3d()
    by_scene: dict = {}
    for fpath in allf:
        by_scene.setdefault(fpath.split("scene_")[1][:5], []).append(fpath)
    scenes = sorted(by_scene)
    n_val_scenes = max(1, len(scenes) // 10)
    val_files = [fp for s in scenes[-n_val_scenes:] for fp in by_scene[s]][:S3D_VAL]
    train_files = [fp for s in scenes[:-n_val_scenes] for fp in by_scene[s]][:S3D_TRAIN]

    configs = [build_config(hf) for hf in HFOV_BANK]
    # head BEFORE fusion: SetFusion init must not consume RNG ahead of the head,
    # or the mean/attn runs would start from different head weights (breaks pairing)
    head = torch.nn.Linear(enc.dim, P.N_CLASS).to(DEVICE)
    fusion = SetFusion(dim=enc.dim).to(DEVICE) if FUSION == "attn" else None
    params = list(head.parameters()) + (list(fusion.parameters()) if fusion else [])
    opt = torch.optim.AdamW(params, lr=LR)
    lossf = torch.nn.CrossEntropyLoss(ignore_index=P.IGNORE)
    steps_per_ep = (len(train_files) + BATCH - 1) // BATCH
    total_steps = SMOKE_STEPS if SMOKE_STEPS else EPOCHS * steps_per_ep
    print(f"F-2 fusion={FUSION} enc={ADAPTER} train={len(train_files)} val={len(val_files)} "
          f"batch={BATCH} tiles/pano~{len(configs[1]['plan'])} "
          f"params={sum(p.numel() for p in params)/1e6:.2f}M steps={total_steps}", flush=True)

    g0 = torch.Generator().manual_seed(0)
    step, t0, run_loss, done = 0, time.time(), 0.0, False
    pbar = tqdm(total=total_steps, desc=f"f2-{FUSION}", mininterval=10,
                file=sys.stdout, dynamic_ncols=True)
    for ep in range(EPOCHS):
        if done:
            break
        order = torch.randperm(len(train_files), generator=g0).tolist()
        pool_exec = ThreadPoolExecutor(max_workers=WORKERS)  # render is the CPU bottleneck
        for bs in range(0, len(order), BATCH):
            cfg = configs[int(torch.randint(0, len(configs), (1,), generator=g0))]
            jobs, labs = [], []
            for i in order[bs:bs + BATCH]:
                try:
                    rgb, lab = P.load_rgb_label(train_files[i])
                except Exception:
                    continue
                shift = int(torch.randint(0, rgb.shape[1], (1,), generator=g0))
                jobs.append(pool_exec.submit(render_cfg_tiles, np.roll(rgb, shift, axis=1), cfg))
                labs.append(np.roll(lab, shift, axis=1))
            tiles_list = [j.result() for j in jobs]
            if not tiles_list:
                continue
            n_tile = len(cfg["plan"])
            feats = encode_tiles(enc, torch.cat(tiles_list))                 # (P*T, N, D)
            packs = [pano_pack(feats[pi * n_tile:(pi + 1) * n_tile], labs[pi], cfg)
                     for pi in range(len(tiles_list))]
            f = torch.cat([p[0] for p in packs]).to(DEVICE)
            g = torch.cat([p[1] for p in packs]).to(DEVICE)
            m = torch.cat([p[2] for p in packs]).to(DEVICE)
            gt = torch.cat([p[3] for p in packs]).to(DEVICE)
            logits = head(fuse(fusion, f, g, m))
            loss = lossf(logits, gt)
            opt.zero_grad(); loss.backward(); opt.step()
            run_loss += float(loss.detach()); step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{float(loss.detach()):.3f}", refresh=False)
            if step % LOG_EVERY == 0:
                print(f"ep{ep} step{step}/{total_steps} loss={run_loss/LOG_EVERY:.3f} "
                      f"({(time.time()-t0)/step:.2f}s/it, {BATCH} panos/it)", flush=True)
                run_loss = 0.0
            if step >= total_steps:
                done = True
                break
        mi, mi_hi, mi_lo = evaluate(enc, fusion, head, val_files, configs[1])
        print(f"[val ep{ep}] mIoU={mi:.3f}  hi-var(30%)={mi_hi:.3f}  lo-var={mi_lo:.3f}", flush=True)

    pbar.close()
    os.makedirs(CKPT, exist_ok=True)
    state = {"fusion": fusion.state_dict() if fusion else None,
             "head": head.state_dict(), "fusion_kind": FUSION, "adapter": ADAPTER}
    torch.save(state, os.path.join(CKPT, "fusion_f2.pt"))
    run = runlog.create_run(f"f2_fusion_{FUSION}", {
        "fusion": FUSION, "adapter": ADAPTER, "s3d_train": len(train_files),
        "s3d_val": len(val_files), "epochs": EPOCHS, "batch": BATCH, "lr": LR,
        "max_cov": MAX_COV, "hfov_bank": HFOV_BANK, "smoke_steps": SMOKE_STEPS,
        "steps": total_steps, "params_M": sum(p.numel() for p in params) / 1e6})
    torch.save(state, os.path.join(run, "weights", "fusion_f2.pt"))
    save_val_viz(enc, fusion, head, val_files, configs[1], run)
    print(f"saved -> {CKPT}/fusion_f2.pt and {run}", flush=True)


if __name__ == "__main__":
    main()
