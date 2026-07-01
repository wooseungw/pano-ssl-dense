"""Multi-FOV combine gate: does combining COMPLEMENTARY views (different FOV = different context)
add accuracy, where same-FOV overlap fusion only tied?

Same locations encoded at FOV_a (narrow=detail) and FOV_b (wide=context), each obliquity-merged to
a 64x128 field. On the SAME cell set (covered by both), linear-probe seg for: single-A, single-B,
concat(A,B). concat > best-single => complementary multi-scale combine is a real lever (like FPN),
distinct from the redundant same-FOV fusion that tied.

Run: CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n pano python scripts/multifov_gate.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data  # noqa: E402
import anyres_e2p as a2p  # noqa: E402
import geometry as G  # noqa: E402
import probe_seg_dinov3 as P  # noqa: E402
from viz_merged_field import HF, WF, TILE, PATCH, ERP_W, ERP_H  # noqa: E402
from encoder import PanoEncoder, normalize_tiles  # noqa: E402

DEVICE = "cuda"
SEED = 0
MODEL = os.environ.get("MODEL", "facebook/dinov3-vitb16-pretrain-lvd1689m")
FA = float(os.environ.get("FA", 65.0))
FB = float(os.environ.get("FB", 110.0))
TR = int(os.environ.get("TR", 100))
VA = int(os.environ.get("VA", 30))


def obliq_fov(gh, fov):
    cy = (np.arange(gh) + 0.5) * TILE / gh
    XX, YY = np.meshgrid(cy, cy)
    return G._offaxis_cos(XX, YY, TILE, fov).reshape(-1)


def cell_ids_fov(plan, gh, fov):
    ids = []
    for tp in plan:
        cm = G.render_coordmap(ERP_H, ERP_W, tp.yaw_deg, tp.pitch_deg, fov, gh)
        uf = np.clip((cm[..., 0] / ERP_W * WF).astype(int), 0, WF - 1)
        vf = np.clip((cm[..., 1] / ERP_H * HF).astype(int), 0, HF - 1)
        ids.append((vf * WF + uf).reshape(-1))
    return ids


@torch.no_grad()
def build_field_fov(enc, erp, fov):
    plan = a2p.plan_tiles("full_sphere", fov, fov, 0.25)
    gh = TILE // enc.patch
    ids = cell_ids_fov(plan, gh, fov); w = obliq_fov(gh, fov); D = enc.dim
    fs = np.zeros((HF * WF, D), np.float32); ws = np.zeros(HF * WF, np.float32)
    for tp, c in zip(plan, ids):
        t = np.asarray(a2p.erp_to_pinhole_tile(erp, tp.yaw_deg, tp.pitch_deg, fov, TILE))
        x = normalize_tiles((torch.from_numpy(t).float().permute(2, 0, 1)[None] / 255.0).to(DEVICE))
        fmap = enc(x)[0].permute(1, 2, 0).reshape(-1, D).float().cpu().numpy()
        np.add.at(fs, c, w[:, None] * fmap); np.add.at(ws, c, w)
    cov = ws > 0; fs[cov] /= ws[cov][:, None]
    return fs.astype(np.float16), cov


def cache(enc, files):
    out = []
    for f in files:
        erp = np.array(Image.open(f).convert("RGB").resize((ERP_W, ERP_H), Image.BILINEAR))
        fa, ca = build_field_fov(enc, erp, FA); fb, cb = build_field_fov(enc, erp, FB)
        _, lab = P.load_rgb_label(f)
        seg = P.label_to_grid(lab, HF, WF).reshape(-1).astype(np.int64)
        out.append((fa, ca, fb, cb, seg))
    return out


def gather(c, mode):
    Xs, ys = [], []
    for fa, ca, fb, cb, seg in c:
        both = ca & cb
        X = {"a": fa[both], "b": fb[both]}.get(mode) if mode != "concat" else np.concatenate([fa[both], fb[both]], 1)
        Xs.append(X.astype(np.float32)); ys.append(seg[both])
    return np.concatenate(Xs), np.concatenate(ys)


def probe(Xtr, ytr, Xva, yva):
    keep = ytr != P.IGNORE; Xtr, ytr = Xtr[keep], ytr[keep]
    idx = np.random.RandomState(SEED).permutation(len(Xtr))[:300000]
    Xt = torch.from_numpy(Xtr[idx]).to(DEVICE); yt = torch.from_numpy(ytr[idx]).to(DEVICE)
    torch.manual_seed(SEED); clf = nn.Linear(Xt.shape[1], P.N_CLASS).to(DEVICE)
    opt = torch.optim.Adam(clf.parameters(), 1e-3, weight_decay=1e-4)
    for _ in range(800):
        opt.zero_grad(); F.cross_entropy(clf(Xt), yt, ignore_index=P.IGNORE).backward(); opt.step()
    with torch.no_grad():
        pr = clf(torch.from_numpy(Xva).to(DEVICE)).argmax(1).cpu()
    return P.miou_acc(pr, torch.from_numpy(yva))[0]


def main():
    np.random.seed(SEED)
    P.configure("stanford2d3d"); P.TILE = TILE
    enc = PanoEncoder(model_id=MODEL, lora_rank=0).to(DEVICE).eval(); P.enc_patch = enc.patch
    files = data.list_erps("stanford2d3d")
    def area(f): return f.split("extracted_data/")[1].split("/")[0]
    tr = [f for f in files if "5" not in area(f)][:TR]; va = [f for f in files if "5" in area(f)][:VA]
    print(f"multi-FOV combine gate | enc={MODEL.split('/')[-1]} FOV {FA:.0f}(detail)+{FB:.0f}(context) "
          f"field={HF}×{WF} tr={len(tr)} va={len(va)}", flush=True)
    ctr = cache(enc, tr); cva = cache(enc, va)
    both0 = float(np.mean([(ca & cb).mean() for ca_cb in [ctr] for fa, ca, fb, cb, _ in ca_cb]))
    print(f"  mean both-covered cells = {both0:.3f}", flush=True)

    res = {}
    for mode in ("a", "b", "concat"):
        Xtr, ytr = gather(ctr, mode); Xva, yva = gather(cva, mode)
        res[mode] = probe(Xtr, ytr, Xva, yva)
        print(f"  {mode:6} (dim {Xtr.shape[1]}) -> mIoU {res[mode]:.3f}", flush=True)
    best_single = max(res["a"], res["b"])
    d = res["concat"] - best_single
    gate = "✅ complementary combine helps" if d > 0.01 else "❌ no gain (single FOV suffices)"
    print(f"\n=== single-{FA:.0f} {res['a']:.3f} | single-{FB:.0f} {res['b']:.3f} | "
          f"concat {res['concat']:.3f}  Δ(vs best single)={d:+.3f}  {gate} ===", flush=True)


if __name__ == "__main__":
    main()
