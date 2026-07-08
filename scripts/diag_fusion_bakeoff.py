"""F-1 fusion bake-off (frozen encoder, NO training beyond one shared linear head).

Which combination of per-tile predictions and the integrated (blended) feature
harvests the measured ensemble headroom (blend_fair - single_fair ~ +0.09)?
Prior results exhausted the GEOMETRIC weighting axis (naive ~ obliquity ~ deformable,
RESULTS.md 3.8) because E2P correspondence is exact. This bake-off probes the two
untested axes on the same ERP-stitched basis as eval_ssl:

  level  : feature-mean (1 decode)  vs  prediction/logit-mean (N decodes)  vs
           agreement-gated SELECTION (blend only where tiles agree; D1 showed 24-28%%
           of overlap cells hold contradictory predictions -> averaging launders them)
  weight : uniform / obliquity (geometric, known)  vs  confidence (content, untested)

Variants (per ERP cell, over its covering tiles):
  single  least-oblique tile's prediction                      [cheap baseline]
  featU   head( mean feature )                                 [current 'blend']
  featW   head( obliquity-weighted mean feature )
  logitU  argmax( mean logits )                                [prediction-level]
  logitC  argmax( confidence-weighted mean logits )
  gateC   agree -> consensus ; disagree -> most-CONFIDENT tile's pred   [selection]
  gateO   agree -> consensus ; disagree -> least-OBLIQUE tile's pred    [selection]

Metrics: mIoU on all covered cells / seam cells (cov>=2) / DISAGREEMENT cells
(cov>=2 & contradictory preds) - fusion choices only differ where tiles disagree.

Run: CUDA_VISIBLE_DEVICES=0 conda run -n pano python scripts/diag_fusion_bakeoff.py [densepass|stanford2d3d]
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import probe_seg_dinov3 as P  # noqa: E402
import diag_seam as D  # noqa: E402
import geometry as G  # noqa: E402
from encoder import PanoEncoder, normalize_tiles  # noqa: E402

DEVICE = P.DEVICE
DOMAINS = {"densepass": 50.0, "stanford2d3d": 65.0}   # eval-matched FOV per domain


@torch.no_grad()
def tile_pass(enc, head, rgb, tp):
    """One tile -> (feat (N,D) cpu, logits (N,C) cpu, conf (N,), pred (N,), (gh,gw))."""
    tile = np.asarray(P.a2p.erp_to_pinhole_tile(rgb, tp.yaw_deg, tp.pitch_deg, D.HFOV, D.TILE)).copy()
    x = torch.from_numpy(tile).float().permute(2, 0, 1)[None] / 255.0
    feat = P.dense(enc, normalize_tiles(x.to(DEVICE)))[0]              # (D,gh,gw)
    d, gh, gw = feat.shape
    fm = feat.permute(1, 2, 0).reshape(-1, d).cpu()
    logits = head(fm.to(DEVICE).float()).cpu()                          # (N,C)
    prob = logits.softmax(1)
    conf, pred = prob.max(1)
    return fm, logits, conf.numpy(), pred.numpy(), (gh, gw)


def fuse_pano(enc, head, rgb, lab, plan):
    """Scatter all tiles of one pano onto the ERP grid and compute every variant."""
    h, w = rgb.shape[:2]
    hf, wf = h // P.enc_patch, w // P.enc_patch
    ncell, Dd, C = hf * wf, enc.dim, P.N_CLASS
    fsum = torch.zeros(ncell, Dd); fwsum = torch.zeros(ncell, Dd); wsum = np.zeros(ncell)
    lsum = torch.zeros(ncell, C); lcsum = torch.zeros(ncell, C)
    cov = np.zeros(ncell, int)
    first_pred = np.full(ncell, -1, int)
    disagree = np.zeros(ncell, bool)
    best_r = np.full(ncell, 1e9); best_pred = np.zeros(ncell, int)
    best_conf = np.zeros(ncell); conf_pred = np.zeros(ncell, int)
    gt = P.label_to_grid(lab, hf, wf).reshape(-1)

    gh = gw = D.TILE // P.enc_patch
    ii, jj = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
    r = np.sqrt((ii - (gh - 1) / 2) ** 2 + (jj - (gw - 1) / 2) ** 2).reshape(-1)
    col = ((jj + 0.5) * D.TILE / gw).reshape(-1).astype(np.float32)
    row = ((ii + 0.5) * D.TILE / gh).reshape(-1).astype(np.float32)
    wobl = G._offaxis_cos(col, row, D.TILE, D.HFOV)                    # (N,) in (0,1]

    for tp in plan:
        fm, logits, conf, pred, (gh, gw) = tile_pass(enc, head, rgb, tp)
        cid, _ = D.coord_grid((h, w), tp, gh, gw)
        cid = cid.reshape(-1)
        tcid = torch.from_numpy(cid)
        fsum.index_add_(0, tcid, fm)
        fwsum.index_add_(0, tcid, fm * torch.from_numpy(wobl)[:, None].float())
        lsum.index_add_(0, tcid, logits)
        lcsum.index_add_(0, tcid, logits * torch.from_numpy(conf)[:, None])
        np.add.at(wsum, cid, wobl)
        np.add.at(cov, cid, 1)
        for k in range(cid.shape[0]):                                  # order-dependent bits
            c = cid[k]
            if first_pred[c] < 0:
                first_pred[c] = pred[k]
            elif pred[k] != first_pred[c]:
                disagree[c] = True
            if r[k] < best_r[c]:
                best_r[c] = r[k]; best_pred[c] = pred[k]
            if conf[k] > best_conf[c]:
                best_conf[c] = conf[k]; conf_pred[c] = pred[k]

    m = cov >= 1
    covm = torch.from_numpy(cov[m]).float()[:, None]
    with torch.no_grad():
        featU = head((fsum[m] / covm).to(DEVICE).float()).argmax(1).cpu().numpy()
        featW = head((fwsum[m] / torch.from_numpy(wsum[m]).float()[:, None]).to(DEVICE).float()).argmax(1).cpu().numpy()
    preds = {
        "single": best_pred[m], "featU": featU, "featW": featW,
        "logitU": lsum[m].argmax(1).numpy(), "logitC": lcsum[m].argmax(1).numpy(),
        "gateC": np.where(disagree[m], conf_pred[m], first_pred[m]),
        "gateO": np.where(disagree[m], best_pred[m], first_pred[m]),
    }
    return preds, gt[m], cov[m], disagree[m]


VARIANTS = ("single", "featU", "featW", "logitU", "logitC", "gateC", "gateO")


def main():
    domain = sys.argv[1] if len(sys.argv) > 1 else "densepass"
    P.configure(domain); P.TILE = 512
    D.DATASET, D.HFOV, D.OVERLAP, D.TILE = domain, DOMAINS[domain], 0.25, 512
    adapter = os.environ.get("ADAPTER")                 # e.g. runs/ckpt_ssl_m1: fusion delta vs frozen
    enc = (PanoEncoder(model_id=P.MODEL, adapter_path=adapter) if adapter
           else PanoEncoder(model_id=P.MODEL, lora_rank=0)).to(DEVICE).eval()
    print(f"encoder={'adapter:' + adapter if adapter else 'frozen'}", flush=True)
    P.enc_patch = enc.patch
    plan = D.tile_plan()
    panos, groups, train = P.grouped()
    cache = {"tr": [], "va": []}
    for g, f in panos:
        cache["tr" if g in train else "va"].append(P.load_rgb_label(f))
    print(f"domain={domain} hfov={D.HFOV} N_CLASS={P.N_CLASS} tiles/pano={len(plan)} "
          f"tr={len(cache['tr'])} va={len(cache['va'])}", flush=True)
    if not cache["va"]:
        print("no val panos on disk (partial download?) -> abort", flush=True)
        return

    t0 = time.time()
    head = D.head_on_tiles(enc, cache, plan)

    agg = {k: [] for k in VARIANTS}
    gts, covs, dis = [], [], []
    for rgb, lab in cache["va"]:
        preds, gt, cov, dsg = fuse_pano(enc, head, rgb, lab, plan)
        for k in VARIANTS:
            agg[k].append(preds[k])
        gts.append(gt); covs.append(cov); dis.append(dsg)
    gts = np.concatenate(gts); covs = np.concatenate(covs); dis = np.concatenate(dis)
    seam = covs >= 2
    dcell = seam & dis
    lab_ok = gts != P.IGNORE
    print(f"\nval cells={len(gts)}  seam={seam.mean():.3f}  "
          f"disagree|seam={dis[seam & lab_ok].mean():.3f}", flush=True)

    def mi(pred, sel):
        return P.miou_acc(torch.from_numpy(pred[sel]), torch.from_numpy(gts[sel]))[0]

    print(f"\n{'variant':8s} {'all':>7} {'seam':>7} {'disagree':>9}", flush=True)
    for k in VARIANTS:
        p = np.concatenate(agg[k])
        print(f"{k:8s} {mi(p, np.ones_like(seam)):7.3f} {mi(p, seam):7.3f} {mi(p, dcell):9.3f}", flush=True)
    print(f"\n(total {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
