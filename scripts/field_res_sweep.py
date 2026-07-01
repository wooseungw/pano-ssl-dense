"""Resolution sweep for the naive ERP feature-field fusion (frozen DINOv3): build the
fused field at increasing ERP resolutions, decode once -> seg mIoU. Shows the resolution
lever and where it saturates (field finer than the tile patch density starts leaving holes).

Run: CUDA_VISIBLE_DEVICES=0 conda run -n pano python scripts/field_res_sweep.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data  # noqa: E402
import probe_seg_dinov3 as P  # noqa: E402
import adaptive_field_deform as A  # noqa: E402
from encoder import PanoEncoder  # noqa: E402

DEVICE = P.DEVICE
RES = [(32, 64), (64, 128), (128, 256)]


def probe(Xtr, ytr, Xva, yva):
    Xtr, ytr = P.subsample(Xtr, ytr, 300000, 0)
    torch.manual_seed(0); clf = nn.Linear(Xtr.shape[1], P.N_CLASS).to(DEVICE)
    opt = torch.optim.Adam(clf.parameters(), 1e-3, weight_decay=1e-4)
    lf = nn.CrossEntropyLoss(ignore_index=P.IGNORE); Xt, yt = Xtr.to(DEVICE).float(), ytr.to(DEVICE)
    for _ in range(800):
        opt.zero_grad(); lf(clf(Xt), yt).backward(); opt.step()
    with torch.no_grad():
        pr = clf(Xva.to(DEVICE).float()).argmax(1).cpu()
    return P.miou_acc(pr, yva)[0]


def main():
    P.configure("stanford2d3d"); P.TILE = 512
    plan = P.a2p.plan_tiles("band", A.HFOV, A.HFOV, 0.25, pmax_deg=45.0); P.plan = plan
    files = data.list_erps("stanford2d3d")
    def area(f): return f.split("extracted_data/")[1].split("/")[0]
    tr = [f for f in files if "5" not in area(f)][:150]
    va = [f for f in files if "5" in area(f)][:40]
    enc = PanoEncoder(model_id=P.MODEL, lora_rank=0).to(DEVICE).eval(); P.enc_patch = enc.patch
    print(f"field resolution sweep (frozen DINOv3, naive scatter): tr={len(tr)} va={len(va)} tiles={len(plan)}", flush=True)

    cache = {"tr": [], "va": []}
    for sp, fl in [("tr", tr), ("va", va)]:
        for f in fl:
            rgb, lab = P.load_rgb_label(f)
            cache[sp].append((A.encode(enc, rgb, plan), lab))

    print(f"\n{'field':>10} {'cells':>7} {'coverage':>9} {'frozen mIoU':>12}", flush=True)
    for hf, wf in RES:
        A.HF, A.WF, A.NC = hf, wf, hf * wf
        scatter, _, _ = A.precompute_geom(plan)
        cov_any = torch.zeros(A.NC, dtype=torch.bool, device=DEVICE)
        Xtr, ytr, Xva, yva = [], [], [], []
        for sp, store in (("tr", (Xtr, ytr)), ("va", (Xva, yva))):
            for feats, lab in cache[sp]:
                nf, cov = A.naive_field(A.dev(feats), scatter)
                cov_any |= cov
                y = torch.from_numpy(P.label_to_grid(lab, hf, wf).reshape(-1))
                store[0].append(nf[cov].cpu()); store[1].append(y[cov.cpu()])
        miou = probe(torch.cat(Xtr), torch.cat(ytr), torch.cat(Xva), torch.cat(yva))
        print(f"{hf}x{wf:<6} {A.NC:7d} {cov_any.float().mean().item():9.2f} {miou:12.3f}", flush=True)


if __name__ == "__main__":
    main()
