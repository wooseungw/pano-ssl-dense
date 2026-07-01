"""PIFu-style implicit field vs grid-scatter merge, across ALL dense tasks (seg/normal/depth).

For each query cell on the ERP grid, find covering tiles and BILINEAR sub-pixel-sample their
feature at the exact projected (u,v) (PIFu pixel-alignment), obliquity-merge -> per-cell feature ->
shared MLP decoder. Baseline = the current patch-grid scatter merge (quantized) -> same MLP.
Geometry (per-tile uv-buffer + coverage) is image-independent -> precomputed once.

  PIFu > grid  => continuous pixel-aligned sampling recovers detail the grid quantizes away.
Run: CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n pano python scripts/pifu_alltasks.py
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
import geometry as G  # noqa: E402
import probe_seg_dinov3 as P  # noqa: E402
import probe_normal as PN  # noqa: E402
import multitask_eval as M  # noqa: E402
import tiling_decode_compare as TDC  # noqa: E402
from encoder import PanoEncoder, normalize_tiles  # noqa: E402

DEVICE = "cuda"
SEED = 0
MODEL = os.environ.get("MODEL", "facebook/dinov3-vitb16-pretrain-lvd1689m")
ERP_W, ERP_H = 2048, 1024
FOV = 65.0
TILE = int(os.environ.get("TILE", 512))
OUT = int(os.environ.get("OUT", 256))
HQ = int(os.environ.get("HQ", 64))
WQ = int(os.environ.get("WQ", 128))
NC = HQ * WQ
TR = int(os.environ.get("TR", 80))
VA = int(os.environ.get("VA", 30))
EPOCHS = int(os.environ.get("EPOCHS", 20))
LOG125 = math.log(1.25)
TASKS = ("seg", "normal", "depth")


def precompute(plan, gh):
    """Per tile: continuous uv per ERP cell (PIFu sample loc), coverage, obliquity weight, and the
    patch-center cell ids for the grid-scatter baseline. Image-independent."""
    rr, cc = np.mgrid[0:OUT, 0:OUT]
    u_all = (cc / (OUT - 1) * 2 - 1).reshape(-1); v_all = (rr / (OUT - 1) * 2 - 1).reshape(-1)
    w_all = G._offaxis_cos(cc.reshape(-1).astype(np.float64), rr.reshape(-1).astype(np.float64), OUT, FOV)
    cy = ((np.arange(gh) + 0.5) * OUT / gh).astype(int)
    pw = G._offaxis_cos(*np.meshgrid(cy.astype(np.float64), cy.astype(np.float64)), OUT, FOV).reshape(-1)
    per_tile, anycov = [], np.zeros(NC, bool)
    for tp in plan:
        cm = G.render_coordmap(ERP_H, ERP_W, tp.yaw_deg, tp.pitch_deg, FOV, OUT)
        uf = np.clip((cm[..., 0] / ERP_W * WQ).astype(int), 0, WQ - 1)
        vf = np.clip((cm[..., 1] / ERP_H * HQ).astype(int), 0, HQ - 1)
        cell = (vf * WQ + uf).reshape(-1)
        usum = np.zeros(NC); vsum = np.zeros(NC); wsum = np.zeros(NC); cnt = np.zeros(NC)
        np.add.at(usum, cell, u_all); np.add.at(vsum, cell, v_all)
        np.add.at(wsum, cell, w_all); np.add.at(cnt, cell, 1.0)
        cov = cnt > 0; cnt_ = np.maximum(cnt, 1)
        uv = np.stack([usum / cnt_, vsum / cnt_], 1).astype(np.float32)
        oblw = (wsum / cnt_).astype(np.float32)
        cellg = cell.reshape(OUT, OUT)[np.ix_(cy, cy)].reshape(-1)            # patch-center -> cell
        per_tile.append((uv, cov, oblw, cellg)); anycov |= cov
    return per_tile, anycov, pw.astype(np.float32)


@torch.no_grad()
def build_fields(enc, erp, plan, per_tile, pw):
    D = enc.dim
    pf = np.zeros((NC, D), np.float32); pwsum = np.zeros(NC, np.float32)
    gf = np.zeros((NC, D), np.float32); gwsum = np.zeros(NC, np.float32)
    for tp, (uv, cov, oblw, cellg) in zip(plan, per_tile):
        t = np.asarray(a2p.erp_to_pinhole_tile(erp, tp.yaw_deg, tp.pitch_deg, FOV, TILE))
        x = normalize_tiles((torch.from_numpy(t).float().permute(2, 0, 1)[None] / 255.0).to(DEVICE))
        feat = enc(x)                                                         # (1,D,gh,gw)
        idx = np.where(cov)[0]
        g = torch.from_numpy(uv[idx]).view(1, 1, -1, 2).to(DEVICE)
        s = F.grid_sample(feat, g, mode="bilinear", align_corners=False)[0, :, 0, :].T.cpu().numpy()  # (n,D)
        w = oblw[idx]
        np.add.at(pf, idx, w[:, None] * s); np.add.at(pwsum, idx, w)
        fmap = feat[0].permute(1, 2, 0).reshape(-1, D).cpu().numpy()          # (gh*gh, D)
        np.add.at(gf, cellg, pw[:, None] * fmap); np.add.at(gwsum, cellg, pw)
    pc = pwsum > 0; gc = gwsum > 0
    pf[pc] /= pwsum[pc][:, None]; gf[gc] /= gwsum[gc][:, None]
    return pf.reshape(HQ, WQ, D).astype(np.float16), gf.reshape(HQ, WQ, D).astype(np.float16)


class MLP(nn.Module):                                                         # per-cell implicit decoder (1x1)
    def __init__(self, d, c):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(d, 256, 1), nn.GroupNorm(16, 256), nn.GELU(),
                                 nn.Conv2d(256, 256, 1), nn.GroupNorm(16, 256), nn.GELU(),
                                 nn.Conv2d(256, c, 1))

    def forward(self, x):
        return self.net(x)


def loss_of(task, out, gt, cov):
    if task == "seg":
        y = gt["seg"].to(DEVICE); m = (y != P.IGNORE) & cov
        return F.cross_entropy(out.permute(0, 2, 3, 1)[0][m], y[m]) if m.any() else None
    if task == "normal":
        y = gt["nrm"].permute(2, 0, 1)[None].to(DEVICE); m = (gt["nval"].to(DEVICE)) & cov
        c = (F.normalize(out, dim=1) * y).sum(1)[0]
        return (1 - c)[m].mean() if m.any() else None
    y = gt["dlog"].to(DEVICE); m = gt["dval"].to(DEVICE) & cov
    return F.l1_loss(out[0, 0][m], y[m]) if m.any() else None


def train_eval(task, fields_tr, gts_tr, covs_tr, fields_va, gts_va, covs_va, D):
    c = P.N_CLASS if task == "seg" else M.OUT_CH[task]
    torch.manual_seed(SEED); dec = MLP(D, c).to(DEVICE)
    opt = torch.optim.AdamW(dec.parameters(), 1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(SEED)
    for _ in range(EPOCHS):
        for i in torch.randperm(len(fields_tr), generator=g).tolist():
            x = torch.from_numpy(fields_tr[i]).permute(2, 0, 1)[None].float().to(DEVICE)
            ls = loss_of(task, dec(x), gts_tr[i], covs_tr[i].to(DEVICE))
            if ls is not None:
                opt.zero_grad(); ls.backward(); opt.step()
    dec.eval()
    if task == "seg":
        inter = torch.zeros(P.N_CLASS); union = torch.zeros(P.N_CLASS)
    se, sd, sn = 0.0, 0.0, 0
    with torch.no_grad():
        for x_np, gt, cov in zip(fields_va, gts_va, covs_va):
            x = torch.from_numpy(x_np).permute(2, 0, 1)[None].float().to(DEVICE)
            o = dec(x); cm = cov.to(DEVICE)
            if task == "seg":
                pr = o.argmax(1)[0].cpu(); y = gt["seg"]; mm = (y != P.IGNORE) & cov
                for cc2 in range(1, P.N_CLASS):
                    inter[cc2] += ((pr == cc2) & (y == cc2) & mm).sum(); union[cc2] += (((pr == cc2) | (y == cc2)) & mm).sum()
            elif task == "normal":
                p = F.normalize(o, dim=1)[0].permute(1, 2, 0).cpu(); m = gt["nval"] & cov
                cs = (p * gt["nrm"]).sum(-1).clamp(-1, 1)
                se += torch.rad2deg(torch.arccos(cs[m])).sum().item(); sn += int(m.sum())
            else:
                p = o[0, 0].cpu(); m = gt["dval"] & cov; e = (p[m] - gt["dlog"][m]).abs()
                se += e.sum().item(); sd += (e < LOG125).float().sum().item(); sn += int(m.sum())
    if task == "seg":
        return float(np.mean([(inter[k] / union[k]).item() for k in range(1, P.N_CLASS) if union[k] > 0]))
    return se / max(sn, 1) if task == "normal" else (se / max(sn, 1), sd / max(sn, 1))


def main():
    P.configure("stanford2d3d"); P.TILE = TILE
    enc = PanoEncoder(model_id=MODEL, lora_rank=0).to(DEVICE).eval(); P.enc_patch = enc.patch
    plan = a2p.plan_tiles("full_sphere", FOV, FOV, 0.25)
    per_tile, anycov, pw = precompute(plan, TILE // enc.patch)
    covm = torch.from_numpy(anycov.reshape(HQ, WQ))
    files = data.list_erps("stanford2d3d")
    def area(f): return f.split("extracted_data/")[1].split("/")[0]
    tr = [f for f in files if "5" not in area(f)][:TR]; va = [f for f in files if "5" in area(f)][:VA]
    print(f"PIFu vs grid × all tasks | enc={MODEL.split('/')[-1]} grid={HQ}×{WQ} cov={anycov.mean():.3f} "
          f"tr={len(tr)} va={len(va)} ep={EPOCHS}\n", flush=True)

    cache = {"tr": {"pifu": [], "grid": [], "gt": []}, "va": {"pifu": [], "grid": [], "gt": []}}
    for sp, fl in [("tr", tr), ("va", va)]:
        for f in fl:
            erp = np.array(Image.open(f).convert("RGB").resize((ERP_W, ERP_H), Image.BILINEAR))
            pf, gf = build_fields(enc, erp, plan, per_tile, pw)
            _, lab = P.load_rgb_label(f); nrm, nval = PN.load_rgb_normal(f)[1:]; dn, dval = M.load_depth(f)
            gt = TDC.gt_dict(lab, nrm, nval, dn, dval, HQ, WQ, warp=None)
            cache[sp]["pifu"].append(pf); cache[sp]["grid"].append(gf); cache[sp]["gt"].append(gt)
    covs = {sp: [covm] * len(cache[sp]["gt"]) for sp in ("tr", "va")}

    res = {}
    for cond in ("grid", "pifu"):
        for task in TASKS:
            res[(cond, task)] = train_eval(task, cache["tr"][cond], cache["tr"]["gt"], covs["tr"],
                                           cache["va"][cond], cache["va"]["gt"], covs["va"], enc.dim)
            print(f"  [{cond:4}] {task:6} -> {res[(cond, task)]}", flush=True)

    print("\n=== PIFu-implicit vs grid-scatter (same MLP decoder) ===")
    for task in TASKS:
        unit = {"seg": "mIoU↑", "normal": "ang°↓", "depth": "|Δlog|↓ / δ↑"}[task]
        gv, pv = res[("grid", task)], res[("pifu", task)]
        if task == "depth":
            print(f"  {task:6} ({unit}): grid {gv[0]:.3f}/{gv[1]:.2f}  pifu {pv[0]:.3f}/{pv[1]:.2f}  "
                  f"Δ={pv[0]-gv[0]:+.3f}/{pv[1]-gv[1]:+.2f}")
        else:
            better = (pv - gv) if task == "seg" else (gv - pv)
            print(f"  {task:6} ({unit}): grid {gv:.3f}  pifu {pv:.3f}  Δ(pifu better by)={better:+.3f}")


if __name__ == "__main__":
    main()
