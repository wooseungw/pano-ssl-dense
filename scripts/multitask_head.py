"""Unified multi-task head (MVP): one PIFu-implicit field -> shared trunk -> seg/normal/depth heads,
trained JOINTLY with uncertainty weighting (Kendall et al.). Tests the core unification claim:
does one shared representation serve all tasks SIMULTANEOUSLY without interference vs per-task heads?

Foundation (settled): frozen encoder -> PIFu-implicit merge (continuous sub-pixel) -> light head.
This adds the SHARED-TRUNK MULTI-TASK head on top. Compares joint vs single-task (same PIFu field).

Run: OPENCV_IO_ENABLE_OPENEXR=1 CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n pano python scripts/multitask_head.py
"""
from __future__ import annotations

import math
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
import probe_seg_dinov3 as P  # noqa: E402
import probe_normal as PN  # noqa: E402
import multitask_eval as M  # noqa: E402
import tiling_decode_compare as TDC  # noqa: E402
import pifu_alltasks as PF  # noqa: E402  (precompute, build_fields, loss_of, MLP, constants)
from encoder import PanoEncoder  # noqa: E402

DEVICE = "cuda"
SEED = 0
MODEL = os.environ.get("MODEL", "facebook/dinov3-vitb16-pretrain-lvd1689m")
ERP_W, ERP_H = 2048, 1024
HQ, WQ = PF.HQ, PF.WQ
TR = int(os.environ.get("TR", 80))
VA = int(os.environ.get("VA", 30))
EPOCHS = int(os.environ.get("EPOCHS", 25))
LOG125 = math.log(1.25)
TASKS = ("seg", "normal", "depth")


class MultiTaskNet(nn.Module):
    """Shared 1x1-conv trunk + per-task heads + learnable uncertainty weights (auto task balance)."""

    def __init__(self, d, n_cls):
        super().__init__()
        self.trunk = nn.Sequential(nn.Conv2d(d, 256, 1), nn.GroupNorm(16, 256), nn.GELU(),
                                   nn.Conv2d(256, 256, 1), nn.GroupNorm(16, 256), nn.GELU())
        self.heads = nn.ModuleDict({"seg": nn.Conv2d(256, n_cls, 1),
                                    "normal": nn.Conv2d(256, 3, 1), "depth": nn.Conv2d(256, 1, 1)})
        self.logvar = nn.Parameter(torch.zeros(3))                  # Kendall uncertainty weighting

    def forward(self, x):
        h = self.trunk(x)
        return {k: self.heads[k](h) for k in self.heads}


def eval_metric(task, out_fn, fields_va, gts_va, cov):
    """out_fn(x)->(1,C,H,W). Returns task metric on covered cells."""
    if task == "seg":
        inter = torch.zeros(P.N_CLASS); union = torch.zeros(P.N_CLASS)
    se, sd, sn = 0.0, 0.0, 0
    with torch.no_grad():
        for x_np, gt in zip(fields_va, gts_va):
            x = torch.from_numpy(x_np).permute(2, 0, 1)[None].float().to(DEVICE)
            o = out_fn(x)
            if task == "seg":
                pr = o.argmax(1)[0].cpu(); y = gt["seg"]; mm = (y != P.IGNORE) & cov
                for c in range(1, P.N_CLASS):
                    inter[c] += ((pr == c) & (y == c) & mm).sum(); union[c] += (((pr == c) | (y == c)) & mm).sum()
            elif task == "normal":
                p = F.normalize(o, dim=1)[0].permute(1, 2, 0).cpu(); m = gt["nval"] & cov
                cs = (p * gt["nrm"]).sum(-1).clamp(-1, 1)
                se += torch.rad2deg(torch.arccos(cs[m])).sum().item(); sn += int(m.sum())
            else:
                p = o[0, 0].cpu(); m = gt["dval"] & cov; e = (p[m] - gt["dlog"][m]).abs()
                se += e.sum().item(); sd += (e < LOG125).float().sum().item(); sn += int(m.sum())
    if task == "seg":
        return float(np.mean([(inter[c] / union[c]).item() for c in range(1, P.N_CLASS) if union[c] > 0]))
    return se / max(sn, 1) if task == "normal" else (se / max(sn, 1), sd / max(sn, 1))


def train_single(task, ftr, gtr, cov, D):
    c = P.N_CLASS if task == "seg" else M.OUT_CH[task]
    torch.manual_seed(SEED); dec = PF.MLP(D, c).to(DEVICE)
    opt = torch.optim.AdamW(dec.parameters(), 1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(SEED); covd = cov.to(DEVICE)
    for _ in range(EPOCHS):
        for i in torch.randperm(len(ftr), generator=g).tolist():
            x = torch.from_numpy(ftr[i]).permute(2, 0, 1)[None].float().to(DEVICE)
            ls = PF.loss_of(task, dec(x), gtr[i], covd)
            if ls is not None:
                opt.zero_grad(); ls.backward(); opt.step()
    dec.eval()
    return dec


def train_multi(ftr, gtr, cov, D):
    torch.manual_seed(SEED); net = MultiTaskNet(D, P.N_CLASS).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(SEED); covd = cov.to(DEVICE)
    for _ in range(EPOCHS):
        net.train()
        for i in torch.randperm(len(ftr), generator=g).tolist():
            x = torch.from_numpy(ftr[i]).permute(2, 0, 1)[None].float().to(DEVICE)
            out = net(x); total = 0.0
            for j, task in enumerate(TASKS):
                ls = PF.loss_of(task, out[task], gtr[i], covd)
                if ls is not None:
                    total = total + torch.exp(-net.logvar[j]) * ls + net.logvar[j]   # Kendall
            opt.zero_grad(); total.backward(); opt.step()
    net.eval()
    return net


def main():
    P.configure("stanford2d3d"); P.TILE = PF.TILE
    enc = PanoEncoder(model_id=MODEL, lora_rank=0).to(DEVICE).eval(); P.enc_patch = enc.patch
    plan = a2p.plan_tiles("full_sphere", PF.FOV, PF.FOV, 0.25)
    per_tile, anycov, pw = PF.precompute(plan, PF.TILE // enc.patch)
    cov = torch.from_numpy(anycov.reshape(HQ, WQ)); D = enc.dim
    files = data.list_erps("stanford2d3d")
    def area(f): return f.split("extracted_data/")[1].split("/")[0]
    tr = [f for f in files if "5" not in area(f)][:TR]; va = [f for f in files if "5" in area(f)][:VA]
    print(f"unified multi-task head | enc={MODEL.split('/')[-1]} PIFu field {HQ}×{WQ} tr={len(tr)} va={len(va)} ep={EPOCHS}\n", flush=True)

    fields, gts = {"tr": [], "va": []}, {"tr": [], "va": []}
    for sp, fl in [("tr", tr), ("va", va)]:
        for f in fl:
            erp = np.array(Image.open(f).convert("RGB").resize((ERP_W, ERP_H), Image.BILINEAR))
            pf, _ = PF.build_fields(enc, erp, plan, per_tile, pw)
            _, lab = P.load_rgb_label(f); nrm, nval = PN.load_rgb_normal(f)[1:]; dn, dval = M.load_depth(f)
            fields[sp].append(pf); gts[sp].append(TDC.gt_dict(lab, nrm, nval, dn, dval, HQ, WQ, warp=None))

    single = {}
    for task in TASKS:
        dec = train_single(task, fields["tr"], gts["tr"], cov, D)
        single[task] = eval_metric(task, lambda x, d=dec: d(x), fields["va"], gts["va"], cov)
        print(f"  [single] {task:6} -> {single[task]}", flush=True)

    net = train_multi(fields["tr"], gts["tr"], cov, D)
    multi = {t: eval_metric(t, lambda x, t=t: net(x)[t], fields["va"], gts["va"], cov) for t in TASKS}
    for t in TASKS:
        print(f"  [multi ] {t:6} -> {multi[t]}", flush=True)
    print(f"  learned task weights exp(-logvar) = {torch.exp(-net.logvar).detach().cpu().numpy().round(3)}", flush=True)

    print("\n=== single-task vs unified multi-task (same PIFu field) ===")
    for t in TASKS:
        unit = {"seg": "mIoU↑", "normal": "ang°↓", "depth": "|Δlog|↓/δ↑"}[t]
        s, m = single[t], multi[t]
        if t == "depth":
            print(f"  {t:6} ({unit}): single {s[0]:.3f}/{s[1]:.2f}  multi {m[0]:.3f}/{m[1]:.2f}  Δ={m[0]-s[0]:+.3f}/{m[1]-s[1]:+.2f}")
        else:
            d = (m - s) if t == "seg" else (s - m)
            print(f"  {t:6} ({unit}): single {s:.3f}  multi {m:.3f}  Δ(multi better)={d:+.3f}")


if __name__ == "__main__":
    main()
