"""Multi-seed paired de-risk of the TC3 single-split claims (RESULTS §3.9 discipline).

Claims under test (single-split, DensePASS@50):
  1. Δsingle_fair (tc3−frozen)  +0.012   probe accuracy
  2. Δblend_fair  (tc3−frozen)  +0.018   "adapter raises the ensemble ceiling" hint
  3. Δpurity      (tc3−geo)     +0.008   no-erosion half of the pre-registered criterion
  4. ΔARI         (tc3−geo)     +0.100   semantic-code-agreement half

Protocol: 4 seeds; per seed re-draw (i) train-pano subset (50/70), (ii) linear-head
init, (iii) 40k patch subsample, (iv) k-means init — all three encoders SHARE the
seed's draws (paired deltas). Val = the fixed 30 held-out panos. Features are
extracted ONCE per encoder; the seed loop only refits heads/k-means.

Verdict per claim: mean±std of the paired delta + sign-consistency (n/4 seeds).
Run: CUDA_VISIBLE_DEVICES=0 conda run -n pano python scripts/derisk_tc3.py
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
import train_ssl as T  # noqa: E402
import diag_semantic_headroom as H  # noqa: E402
from encoder import PanoEncoder  # noqa: E402

DEVICE = P.DEVICE
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENCODERS = {
    "frozen": None,
    "geo": os.path.join(ROOT, "runs", "ckpt_ssl_lora"),
    "tc3": os.path.join(ROOT, "runs", "ckpt_ssl_tc3"),
}
SEEDS = (0, 1, 2, 3)
N_SUBSET = 50                    # train panos per seed (of 70)
N_PATCH = 40000                  # head-training patch subsample
N_KM_FIT = 40000                 # k-means fit sample
K = 64
TILE_KEEP = 3000                 # stored train tile-cells per pano (k-means pool)


@torch.no_grad()
def extract_pano(enc, rgb, lab, plan, keep_tiles: bool):
    """One pass per pano: ERP-scatter (single/blend/GT) + optional per-tile features."""
    h, w = rgb.shape[:2]
    hf, wf = h // P.enc_patch, w // P.enc_patch
    ncell, dd = hf * wf, enc.dim
    fsum = torch.zeros(ncell, dd)
    single = torch.zeros(ncell, dd)
    cov = np.zeros(ncell, int)
    best_r = np.full(ncell, 1e9)
    gt = P.label_to_grid(lab, hf, wf).reshape(-1)
    gh = gw = D.TILE // P.enc_patch
    ii, jj = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
    r = np.sqrt((ii - (gh - 1) / 2) ** 2 + (jj - (gw - 1) / 2) ** 2).reshape(-1)
    tfs, tls = [], []
    for tp in plan:
        fmap, _, (gh, gw) = D.tile_feat_pred(enc, rgb, tp)          # (gh,gw,D) cpu
        gl = P.label_to_grid(P.e2p_label(lab, tp.yaw_deg, tp.pitch_deg, D.HFOV, D.TILE), gh, gw)
        fm = fmap.reshape(-1, dd)
        tfs.append(fm.half()); tls.append(torch.from_numpy(gl.reshape(-1)))
        cid, _ = D.coord_grid((h, w), tp, gh, gw)
        cid = cid.reshape(-1)
        fsum.index_add_(0, torch.from_numpy(cid), fm)
        np.add.at(cov, cid, 1)
        for k in range(cid.shape[0]):
            c = cid[k]
            if r[k] < best_r[c]:
                best_r[c] = r[k]; single[c] = fm[k]
    m = cov >= 1
    blend = fsum[m] / torch.from_numpy(cov[m]).float()[:, None]
    out = {"S": single[m].half(), "B": blend.half(), "G": torch.from_numpy(gt[m])}
    if keep_tiles:
        out["tf"] = torch.stack(tfs)                                 # (Ttiles, N, D) fp16
        out["tl"] = torch.stack(tls)
    return out


def fit_km(x: torch.Tensor, seed: int):
    from sklearn.cluster import MiniBatchKMeans
    xn = torch.nn.functional.normalize(x.float(), dim=1).numpy()
    return MiniBatchKMeans(n_clusters=K, random_state=seed, batch_size=4096,
                           n_init=5, max_iter=200).fit(xn)


def km_assign(km, x: torch.Tensor) -> np.ndarray:
    return km.predict(torch.nn.functional.normalize(x.float(), dim=1).numpy())


def seed_metrics(data, geom, seed: int) -> dict:
    """Paired metrics for one encoder under one seed's shared draws."""
    from sklearn.metrics import adjusted_rand_score
    g = torch.Generator().manual_seed(seed)
    tr_idx = torch.randperm(70, generator=g)[:N_SUBSET].tolist()     # shared across encoders

    s_tr = torch.cat([data["tr"][i]["S"] for i in tr_idx]).float()
    b_tr = torch.cat([data["tr"][i]["B"] for i in tr_idx]).float()
    g_tr = torch.cat([data["tr"][i]["G"] for i in tr_idx])
    s_tr, g_s = P.subsample(s_tr, g_tr, N_PATCH, seed)
    b_tr, g_b = P.subsample(b_tr, g_tr, N_PATCH, seed)
    single_fair = H.miou(H.predict(H.train_head(s_tr, g_s, seed), data["S_va"]), data["G_va"])
    blend_fair = H.miou(H.predict(H.train_head(b_tr, g_b, seed), data["B_va"]), data["G_va"])

    pool = torch.cat([data["tr"][i]["tf"].reshape(-1, data["tr"][i]["tf"].shape[-1])
                      for i in tr_idx])
    pool, _ = P.subsample(pool.float(), torch.zeros(pool.shape[0], dtype=torch.long), N_KM_FIT, seed)
    km = fit_km(pool, seed)

    # purity on val tiles (GT-labelled cells)
    asg_all, lab_all = [], []
    for pano in data["va"]:
        asg_all.append(km_assign(km, pano["tf"].reshape(-1, pano["tf"].shape[-1])))
        lab_all.append(pano["tl"].reshape(-1).numpy())
    asg, lab = np.concatenate(asg_all), np.concatenate(lab_all)
    mm = lab != P.IGNORE
    asg, lab = asg[mm], lab[mm]
    purity_num, purity_den = 0, 0
    for c in range(K):
        sel = asg == c
        if sel.sum():
            purity_num += np.bincount(lab[sel]).max(); purity_den += sel.sum()
    purity = purity_num / max(purity_den, 1)

    # cross-view code ARI on val overlap pairs
    a_lab, b_lab = [], []
    for pano in data["va"]:
        asgs = [km_assign(km, pano["tf"][t]) for t in range(pano["tf"].shape[0])]
        gh = gw = D.TILE // P.enc_patch
        for (a, b), (grid, valid, weight) in zip(geom["pairs"], geom["warps"]):
            v = valid.cpu().bool().numpy()
            if v.sum() < 4:
                continue
            tb = H.true_b_cell(grid.cpu(), gh, gw).numpy()
            a_lab.append(asgs[a][v]); b_lab.append(asgs[b][tb][v])
    ari = adjusted_rand_score(np.concatenate(a_lab), np.concatenate(b_lab))
    return {"single_fair": single_fair, "blend_fair": blend_fair, "purity": purity, "ari": ari}


def main() -> None:
    P.configure("densepass"); P.TILE = 512
    D.DATASET, D.HFOV, D.OVERLAP, D.TILE = "densepass", 50.0, 0.25, 512
    plan = D.tile_plan()
    panos, groups, train = P.grouped()
    cache_tr = [P.load_rgb_label(f) for gname, f in panos if gname in train]
    cache_va = [P.load_rgb_label(f) for gname, f in panos if gname not in train]
    print(f"derisk: tr={len(cache_tr)} va={len(cache_va)} seeds={SEEDS} subset={N_SUBSET} K={K}", flush=True)

    t0 = time.time()
    data, geom = {}, None
    for tag, path in ENCODERS.items():
        enc = (PanoEncoder(model_id=P.MODEL, adapter_path=path) if path
               else PanoEncoder(model_id=P.MODEL, lora_rank=0)).to(DEVICE).eval()
        P.enc_patch = enc.patch
        if geom is None:
            geom = T.build_geometry(enc, 50.0, (0.0,))
        d = {"tr": [], "va": []}
        for pi, (rgb, lab) in enumerate(cache_tr):
            e = extract_pano(enc, rgb, lab, plan, keep_tiles=True)
            # keep the SAME per-pano random cell subset for every encoder (paired k-means pool)
            gk = torch.Generator().manual_seed(99 + pi)
            idx = torch.randperm(e["tf"].shape[1], generator=gk)[:TILE_KEEP // e["tf"].shape[0]]
            e["tf"] = e["tf"][:, idx].reshape(-1, e["tf"].shape[-1])
            d["tr"].append(e)
        for rgb, lab in cache_va:
            d["va"].append(extract_pano(enc, rgb, lab, plan, keep_tiles=True))
        d["S_va"] = torch.cat([p["S"] for p in d["va"]]).float()
        d["B_va"] = torch.cat([p["B"] for p in d["va"]]).float()
        d["G_va"] = torch.cat([p["G"] for p in d["va"]])
        data[tag] = d
        del enc
        torch.cuda.empty_cache()
        print(f"extracted {tag} ({time.time()-t0:.0f}s)", flush=True)

    rows = {tag: [] for tag in ENCODERS}
    for s in SEEDS:
        for tag in ENCODERS:
            m = seed_metrics(data[tag], geom, s)
            rows[tag].append(m)
            print(f"seed{s} {tag:7s} single_fair={m['single_fair']:.3f} blend_fair={m['blend_fair']:.3f} "
                  f"purity={m['purity']:.3f} ARI={m['ari']:.3f}", flush=True)

    def delta(metric, a, b):
        return np.array([rows[a][i][metric] - rows[b][i][metric] for i in range(len(SEEDS))])

    print(f"\n=== paired deltas over {len(SEEDS)} seeds (mean±std, sign-consistency) ===", flush=True)
    for name, metric, a, b in [
        ("Δsingle_fair tc3−frozen", "single_fair", "tc3", "frozen"),
        ("Δblend_fair  tc3−frozen", "blend_fair", "tc3", "frozen"),
        ("Δsingle_fair geo−frozen", "single_fair", "geo", "frozen"),
        ("Δblend_fair  geo−frozen", "blend_fair", "geo", "frozen"),
        ("Δpurity      tc3−geo   ", "purity", "tc3", "geo"),
        ("ΔARI         tc3−geo   ", "ari", "tc3", "geo"),
        ("Δpurity      tc3−frozen", "purity", "tc3", "frozen"),
        ("ΔARI         geo−frozen", "ari", "geo", "frozen"),
    ]:
        dl = delta(metric, a, b)
        n_pos = int((dl > 0).sum())
        robust = "ROBUST" if (abs(dl.mean()) > dl.std() and n_pos in (0, len(SEEDS))) else "noise?"
        print(f"{name}: {dl.mean():+.4f} ± {dl.std():.4f}  ({n_pos}/{len(SEEDS)} positive)  [{robust}]", flush=True)
    print(f"\n(total {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
