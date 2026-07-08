"""Pre-training diagnostics for Semantic-Identity SSL (docs/SEMANTIC_IDENTITY_SSL.md).

NO ENCODER TRAINING. Frozen DINOv3 only. Answers three questions BEFORE committing
to any SSL run (project law: diagnose before training):

  D-A  single-view recoverability of the ENSEMBLE answer  (M2 gate, load-bearing)
       eval_ssl showed the multi-view blend beats single-best by ~+0.12 mIoU. Is that
       gain a deterministic distortion the encoder could learn to correct (canonicalization,
       single-view-recoverable -> M2 promising), or irreducible multi-view variance-reduction
       (a single view structurally cannot reconstruct the average -> M2 flat)?
         - reproduce headroom = blend - single  (should be ~+0.12 outdoor)
         - y_ens = ensemble's own argmax (the M2 teacher target)
         - train V: single_feat -> y_ens on TRAIN cells; eval V(single) vs GT on VAL
         - recoverable fraction = (recover - single_fair) / (blend_fair - single_fair)

  D-B  cross-view CODE agreement over overlaps at the BACKBONE feature level  (M1 headroom)
       Cluster frozen backbone features into K prototypes; on every overlap correspondence
       (reuse warp geometry), how often do the two views land in the SAME prototype?
       Low agreement -> frozen codes disagree across views -> M1 has room. High -> little.
       (This is measured on backbone F, exactly where the user wants the invariance.)

  D-C  prototype purity vs GT  (semantic ceiling of the code approach)
       Do frozen-feature prototypes align with GT categories? cluster->majority-class mIoU
       and mean purity bound how much semantic accuracy a code objective could encode.

Run: CUDA_VISIBLE_DEVICES=0 conda run -n pano python scripts/diag_semantic_headroom.py [densepass|stanford2d3d] [K]
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import probe_seg_dinov3 as P  # noqa: E402
import diag_seam as D  # noqa: E402
import train_ssl as T  # noqa: E402
from encoder import PanoEncoder, normalize_tiles  # noqa: E402

DEVICE = P.DEVICE
# domain -> (dataset, hfov, pitch_centers) matched to the SSL/eval protocol
DOMAINS = {
    "densepass": (50.0, (0.0,)),                 # outdoor equator ring, biggest +0.12 headroom
    "stanford2d3d": (65.0, (-45.0, 0.0, 45.0)),  # indoor 3-ring
}
N_KM = 40000          # feature sample for KMeans fit
STEPS = 800


# --------------------------------------------------------------------------- #
# shared head helpers                                                          #
# --------------------------------------------------------------------------- #
def train_head(X, y, seed=0, steps=STEPS):
    torch.manual_seed(seed)
    clf = torch.nn.Linear(X.shape[1], P.N_CLASS).to(DEVICE)
    opt = torch.optim.Adam(clf.parameters(), 1e-3, weight_decay=1e-4)
    lf = torch.nn.CrossEntropyLoss(ignore_index=P.IGNORE)
    X, y = X.to(DEVICE).float(), y.to(DEVICE)
    for _ in range(steps):
        opt.zero_grad(); lf(clf(X), y).backward(); opt.step()
    return clf


@torch.no_grad()
def predict(clf, X):
    return clf(X.to(DEVICE).float()).argmax(1).cpu()


def train_regressor(X, Y, seed=0, steps=STEPS):
    """Linear D->D regressor (MSE). Used for the unconfounded M2 gate: single->blend."""
    torch.manual_seed(seed)
    lin = torch.nn.Linear(X.shape[1], Y.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(lin.parameters(), 1e-3, weight_decay=1e-4)
    X, Y = X.to(DEVICE).float(), Y.to(DEVICE).float()
    for _ in range(steps):
        opt.zero_grad(); F.mse_loss(lin(X), Y).backward(); opt.step()
    return lin


@torch.no_grad()
def apply_lin(lin, X):
    return lin(X.to(DEVICE).float()).cpu()


def miou(pred, gt):
    return P.miou_acc(pred, gt)[0]


# --------------------------------------------------------------------------- #
# per-cell ERP scatter: least-oblique SINGLE feature + BLEND feature + GT      #
# (like diag_seam.scatter_pano but keeps the single-tile feature vector)       #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def scatter_feats(enc, rgb, lab, plan):
    h, w = rgb.shape[:2]
    hf, wf = h // P.enc_patch, w // P.enc_patch
    ncell, Dd = hf * wf, enc.dim
    fsum = torch.zeros(ncell, Dd)
    single = torch.zeros(ncell, Dd)
    cov = np.zeros(ncell, int)
    best_r = np.full(ncell, 1e9)
    gt = P.label_to_grid(lab, hf, wf).reshape(-1)
    gh = gw = D.TILE // P.enc_patch
    ii, jj = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
    r = np.sqrt((ii - (gh - 1) / 2) ** 2 + (jj - (gw - 1) / 2) ** 2).reshape(-1)
    for tp in plan:
        fmap, _, (gh, gw) = D.tile_feat_pred(enc, rgb, tp)      # (gh,gw,D) cpu
        cid, _ = D.coord_grid((h, w), tp, gh, gw)
        cid = cid.reshape(-1)
        fm = fmap.reshape(-1, Dd)
        for k in range(cid.shape[0]):
            c = cid[k]
            fsum[c] += fm[k]; cov[c] += 1
            if r[k] < best_r[c]:
                best_r[c] = r[k]; single[c] = fm[k]
    m = cov >= 1
    blend = fsum[m] / torch.from_numpy(cov[m]).float()[:, None]
    return single[m], blend, torch.from_numpy(gt[m])


def collect_cells(enc, panos_cache, plan):
    S, B, G = [], [], []
    for rgb, lab in panos_cache:
        s, b, g = scatter_feats(enc, rgb, lab, plan)
        S.append(s); B.append(b); G.append(g)
    return torch.cat(S), torch.cat(B), torch.cat(G)


# --------------------------------------------------------------------------- #
# D-A: single-view recoverability of the ensemble answer                      #
# --------------------------------------------------------------------------- #
def diag_A(enc, cache, plan):
    W = D.head_on_tiles(enc, cache, plan)                       # tile-trained ref head (eval_ssl basis)
    S_tr, B_tr, G_tr = collect_cells(enc, cache["tr"], plan)
    S_va, B_va, G_va = collect_cells(enc, cache["va"], plan)

    # W-based single/blend (reproduces eval_ssl's headroom on the ERP grid — validates machinery)
    single_W = miou(predict(W, S_va), G_va)
    blend_W = miou(predict(W, B_va), G_va)

    # fair same-basis heads (cell-trained); h_blend is reused as W_blend for the gate
    h_single = train_head(S_tr, G_tr, 0)
    h_blend = train_head(B_tr, G_tr, 1)
    single_fair = miou(predict(h_single, S_va), G_va)
    blend_fair = miou(predict(h_blend, B_va), G_va)

    # UNCONFOUNDED M2 gate: single -> blend FEATURE regression, scored THROUGH the blend head.
    # Tests M2's real mechanism (feature transfer). Unlike a proxy-label probe it is NOT capped
    # at single_fair: if R reconstructs the blend's discriminative content, W_blend(R(single))
    # approaches blend_fair (recoverable -> M2 worth trying); if the +headroom is irreducible
    # multi-view variance-reduction, R(single) cannot reach it and it stays near single (flat).
    R = train_regressor(S_tr, B_tr, 3)
    Rs = apply_lin(R, S_va)
    recon_cos = F.cosine_similarity(Rs, B_va, dim=1).mean().item()
    recover_feat = miou(predict(h_blend, Rs), G_va)
    identity_floor = miou(predict(h_blend, S_va), G_va)        # raw single through the blend head
    denom = blend_fair - single_fair
    frac = (recover_feat - single_fair) / denom if abs(denom) > 1e-6 else float("nan")

    # context only (this hard-label probe is confounded: capped at single_fair by construction,
    # so its 'recovery' is a lower bound — kept for the informative 'agree' number)
    V = train_head(S_tr, predict(W, B_tr), 2)
    pv = predict(V, S_va); m = G_va != P.IGNORE
    agree = (pv[m] == predict(W, B_va)[m]).float().mean().item()

    return {
        "single_W": single_W, "blend_W": blend_W, "headroom_W": blend_W - single_W,
        "single_fair": single_fair, "blend_fair": blend_fair,
        "recon_cos": recon_cos, "recover_feat": recover_feat, "identity_floor": identity_floor,
        "frac": frac, "agree": agree, "n_va_cells": len(G_va),
    }


# --------------------------------------------------------------------------- #
# clustering (frozen backbone features) for D-B / D-C                          #
# --------------------------------------------------------------------------- #
def collect_tiles(enc, panos_cache, plan, n, seed=0):
    Xs, Ys = [], []
    for rgb, lab in panos_cache:
        for tp in plan:
            fmap, _, (gh, gw) = D.tile_feat_pred(enc, rgb, tp)
            gl = P.label_to_grid(P.e2p_label(lab, tp.yaw_deg, tp.pitch_deg, D.HFOV, D.TILE), gh, gw)
            Xs.append(fmap.reshape(-1, fmap.shape[-1])); Ys.append(torch.from_numpy(gl.reshape(-1)))
    return P.subsample(torch.cat(Xs), torch.cat(Ys), n, seed)


def fit_kmeans(X, k, seed=0):
    from sklearn.cluster import MiniBatchKMeans
    Xn = F.normalize(X.float(), dim=1).numpy()                 # cosine-ish (spherical) k-means
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096, n_init=5, max_iter=200)
    km.fit(Xn)
    return km


def km_assign(km, X):
    return km.predict(F.normalize(X.float(), dim=1).numpy())


def diag_C(km, X_va, Y_va, k):
    """cluster->majority-class mIoU + mean purity (GT-labelled cells only)."""
    asg = km_assign(km, X_va)
    y = Y_va.numpy()
    m = y != P.IGNORE
    asg, y = asg[m], y[m]
    cl2class = np.zeros(k, int)
    purity_num, purity_den = 0, 0
    for c in range(k):
        sel = asg == c
        if sel.sum() == 0:
            continue
        vals, cnts = np.unique(y[sel], return_counts=True)
        cl2class[c] = vals[cnts.argmax()]
        purity_num += cnts.max(); purity_den += sel.sum()
    pred = cl2class[asg]
    mi = miou(torch.from_numpy(pred), torch.from_numpy(y))
    purity = purity_num / max(purity_den, 1)
    return {"purity": purity, "cluster_miou": mi}


# --------------------------------------------------------------------------- #
# D-B: cross-view CODE agreement over overlaps (backbone level)               #
# --------------------------------------------------------------------------- #
def true_b_cell(grid, gh, gw):
    gx, gy = grid[:, 0], grid[:, 1]
    bx = torch.clamp(((gx + 1) / 2 * gw - 0.5).round().long(), 0, gw - 1)
    by = torch.clamp(((gy + 1) / 2 * gh - 0.5).round().long(), 0, gh - 1)
    return by * gw + bx


@torch.no_grad()
def diag_B(enc, km, panos_cache, geom):
    """For each overlap pair, assign both views to prototypes and measure agreement
    on the true geometric correspondents. Also feature cosine as a reference."""
    from sklearn.metrics import adjusted_rand_score
    same, tot = 0, 0
    cos_sum = 0.0
    a_lab, b_lab = [], []
    for rgb, lab in panos_cache:
        feats = []
        for (yaw, pitch) in geom["specs"]:
            tile = np.asarray(P.a2p.erp_to_pinhole_tile(rgb, yaw, pitch, geom["hfov"], D.TILE))
            x = torch.from_numpy(tile).float().permute(2, 0, 1)[None] / 255.0
            f = P.dense(enc, normalize_tiles(x.to(DEVICE)))[0]          # (D,gh,gw)
            d, gh, gw = f.shape
            feats.append(f.permute(1, 2, 0).reshape(-1, d).cpu())        # (gh*gw, D)
        asgs = [km_assign(km, fe) for fe in feats]
        for (a, b), (grid, valid, weight) in zip(geom["pairs"], geom["warps"]):
            v = valid.cpu().bool().numpy()
            if v.sum() < 4:
                continue
            tb = true_b_cell(grid.cpu(), gh, gw).numpy()
            ca, cb = asgs[a], asgs[b][tb]
            fa = F.normalize(feats[a], dim=1); fb = F.normalize(feats[b][tb], dim=1)
            cos = (fa * fb).sum(1).numpy()
            same += int((ca[v] == cb[v]).sum()); tot += int(v.sum())
            cos_sum += float(cos[v].sum())
            a_lab.append(ca[v]); b_lab.append(cb[v])
    a_lab, b_lab = np.concatenate(a_lab), np.concatenate(b_lab)
    ari = adjusted_rand_score(a_lab, b_lab)
    return {"code_agree": same / max(tot, 1), "ari": ari, "cos": cos_sum / max(tot, 1), "n_pairs_cells": tot}


# --------------------------------------------------------------------------- #
def main():
    domain = sys.argv[1] if len(sys.argv) > 1 else "densepass"
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    hfov, pitch_centers = DOMAINS[domain]

    P.configure(domain); P.TILE = 512
    D.DATASET, D.HFOV, D.OVERLAP, D.TILE = domain, hfov, 0.25, 512
    adapter = os.environ.get("ADAPTER")                 # post-training D-B on backbone F
    enc = (PanoEncoder(model_id=P.MODEL, adapter_path=adapter) if adapter
           else PanoEncoder(model_id=P.MODEL, lora_rank=0)).to(DEVICE).eval()
    print(f"encoder={'adapter:' + adapter if adapter else 'frozen'}", flush=True)
    P.enc_patch = enc.patch
    plan = D.tile_plan()
    panos, groups, train = P.grouped()
    cache = {"tr": [], "va": []}
    for g, f in panos:
        cache["tr" if g in train else "va"].append(P.load_rgb_label(f))
    print(f"domain={domain} hfov={hfov} K={K} N_CLASS={P.N_CLASS} tiles/pano={len(plan)} "
          f"tr_panos={len(cache['tr'])} va_panos={len(cache['va'])}", flush=True)

    t0 = time.time()
    a = diag_A(enc, cache, plan)
    print(f"\n=== D-A  M2 gate: is the ensemble's +headroom recoverable by a single view? ===", flush=True)
    print(f"  headroom (eval_ssl basis): single={a['single_W']:.3f} blend={a['blend_W']:.3f} "
          f"-> +{a['headroom_W']:.3f}  [reproduces RESULTS §3.1]", flush=True)
    print(f"  fair basis (cell-trained): single_fair={a['single_fair']:.3f} blend_fair={a['blend_fair']:.3f}", flush=True)
    print(f"  [feature-regression gate]  R:single->blend recon_cos={a['recon_cos']:.3f}", flush=True)
    print(f"     W_blend(R(single))={a['recover_feat']:.3f}  "
          f"(floor W_blend(single)={a['identity_floor']:.3f}, ceil blend_fair={a['blend_fair']:.3f})", flush=True)
    print(f"     RECOVERABLE FRACTION = {a['frac']:.2f}   "
          f"(->1 recoverable/M2 worth trying; ->0 irreducible multi-view/M2 flat)", flush=True)
    print(f"  [context] single linearly predicts ensemble hard-label on {a['agree']:.3f} of cells", flush=True)
    if len(sys.argv) > 3 and sys.argv[3] == "fast":
        print(f"\n(fast mode: D-A only, {time.time()-t0:.0f}s)", flush=True)
        return

    # clustering-based diagnostics
    X_tr, Y_tr = collect_tiles(enc, cache["tr"], plan, N_KM, 0)
    X_va, Y_va = collect_tiles(enc, cache["va"], plan, N_KM, 1)
    km = fit_kmeans(X_tr, K, 0)
    c = diag_C(km, X_va, Y_va, K)
    geom = T.build_geometry(enc, hfov, pitch_centers)
    b = diag_B(enc, km, cache["va"], geom)

    print(f"\n=== D-B  cross-view CODE agreement over overlaps (backbone F, M1 headroom) ===", flush=True)
    print(f"  same-prototype rate = {b['code_agree']:.3f}   ARI = {b['ari']:.3f}   "
          f"(feature cosine ref = {b['cos']:.3f})   cells={b['n_pairs_cells']}", flush=True)
    print(f"  -> lower agreement = more M1 headroom (frozen codes disagree across views)", flush=True)
    print(f"\n=== D-C  prototype purity vs GT (semantic ceiling, K={K}) ===", flush=True)
    print(f"  mean cluster purity = {c['purity']:.3f}   cluster->majority mIoU = {c['cluster_miou']:.3f}", flush=True)
    print(f"\n(total {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
