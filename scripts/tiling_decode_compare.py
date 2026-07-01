"""Fair decomposition comparison with a SHARED trainable decoder + downstream task losses
(not a linear probe), so even basic methods (cubemap) are measured fairly. Each ERP
decomposition -> frozen DINOv3 -> the SAME decoder (SETR-PUP) -> per-view dense prediction
trained with the task loss -> resample to the sphere -> sphere metric.

Methods: erp_direct / cube6 / cube_rot / e2p_full65(ours) / tangent_ico20 / tangent_ico80.
Tasks:   seg (sphere mIoU↑) | normal (ang°↓) | depth (|Δlog|↓ & δ↑).  Stanford2D3D, area5 val.

Run: CUDA_VISIBLE_DEVICES=1 conda run -n pano python scripts/tiling_decode_compare.py [seg|normal|depth|all]
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data  # noqa: E402
import probe_seg_dinov3 as P  # noqa: E402
import probe_normal as PN  # noqa: E402
import multitask_eval as M  # noqa: E402
import tiling_compare as TC  # noqa: E402
from encoder import PanoEncoder, normalize_tiles  # noqa: E402

DEVICE = P.DEVICE
TILE, IH, IW, SR, HS, WS = TC.TILE, TC.IH, TC.IW, TC.SR, TC.HS, TC.WS
TR, VA, EPOCHS = int(os.environ.get("TR", 40)), int(os.environ.get("VA", 20)), int(os.environ.get("EPOCHS", 12))
WHICH = sys.argv[1] if len(sys.argv) > 1 else "all"
TASKS = ("seg", "normal", "depth") if WHICH == "all" else (WHICH,)
LOG125 = math.log(1.25)


def _r1(a, h, w):
    return np.array(Image.fromarray(a.astype(np.float32)).resize((w, h), Image.NEAREST))


def _r3(a, h, w):
    return np.stack([_r1(a[:, :, i], h, w) for i in range(3)], -1)


def gt_dict(lab, nrm, nval, dn, dval, h, w, warp=None):
    """sphere/tile GT at (h,w). warp=(yaw,pitch,hfov) for a tile; None = direct ERP resize."""
    if warp is None:
        sg = P.label_to_grid(lab, h, w)
        nm = _r3(nrm, h, w); dd = _r1(dn, h, w)
        nv = _r1(nval, h, w) > 0.5; dv = (_r1(dval, h, w) > 0.5) & (dd > 1e-3)
    else:
        y, p, hf = warp; wf = lambda a, ch=1: PN.warp_to_grid(a, y, p, hf, h, w, ch)
        sg = wf(lab[:, :, None])[:, :, 0]
        nm = wf(nrm, 3); dd = wf(dn[:, :, None])[:, :, 0]
        nv = wf(nval[:, :, None])[:, :, 0] > 0.5; dv = (wf(dval[:, :, None])[:, :, 0] > 0.5) & (dd > 1e-3)
    nm = nm / np.clip(np.linalg.norm(nm, axis=2, keepdims=True), 1e-6, None)
    return dict(seg=torch.from_numpy(sg.astype(np.int64)), nrm=torch.from_numpy(nm).float(),
                nval=torch.from_numpy(nv), dlog=torch.from_numpy(np.log(np.clip(dd, 1e-3, None))).float(),
                dval=torch.from_numpy(dv))


@torch.no_grad()
def cache_method(enc, files, plan):
    out = []
    for f in files:
        rgb = np.array(Image.open(f).convert("RGB").resize((IW, IH), Image.BILINEAR))
        lab = P.load_rgb_label(f)[1]; nrm, nval = PN.load_rgb_normal(f)[1:]; dn, dval = M.load_depth(f)
        sgt = gt_dict(lab, nrm, nval, dn, dval, HS, WS, warp=None)
        if plan is None:                                    # erp_direct: one view = the ERP; train on sphere GT
            feat = P.dense(enc, normalize_tiles(torch.from_numpy(rgb).float().permute(2, 0, 1)[None].to(DEVICE) / 255.0))[0]
            out.append(([feat.half().cpu()], [sgt], sgt)); continue
        feats, tgts = [], []
        for (yaw, pitch, hfov) in plan:
            t = np.asarray(P.a2p.erp_to_pinhole_tile(rgb, yaw, pitch, hfov, TILE))
            feats.append(P.dense(enc, normalize_tiles(torch.from_numpy(t).float().permute(2, 0, 1)[None].to(DEVICE) / 255.0))[0].half().cpu())
            tgts.append(gt_dict(lab, nrm, nval, dn, dval, SR, SR, warp=(yaw, pitch, hfov)))
        out.append((feats, tgts, sgt))
    return out


def train_dec(task, dim, ctr):
    c = P.N_CLASS if task == "seg" else M.OUT_CH[task]
    torch.manual_seed(0); dec = M.PUP(dim, c).to(DEVICE)
    opt = torch.optim.AdamW(dec.parameters(), 1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(0)
    for _ in range(EPOCHS):
        for i in torch.randperm(len(ctr), generator=g).tolist():
            feats, tgts, _ = ctr[i]; opt.zero_grad()
            for s in range(0, len(feats), 8):
                fb = torch.stack([f.float() for f in feats[s:s + 8]]).to(DEVICE)
                ls = M.loss_of(task, dec(fb), tgts[s:s + 8])
                if ls is not None:
                    (ls * fb.shape[0] / len(feats)).backward()
            opt.step()
    return dec


@torch.no_grad()
def eval_sphere(task, dec, cva, plan):
    cells = None if plan is None else [TC.tile_cells(y, p, h) for (y, p, h) in plan]
    inter = torch.zeros(P.N_CLASS); union = torch.zeros(P.N_CLASS); se, sd, sn = 0.0, 0.0, 0
    ch = P.N_CLASS if task == "seg" else M.OUT_CH[task]
    for feats, _, sgt in cva:
        if plan is None:
            o = dec(feats[0].float()[None].to(DEVICE))[0]
            pseg = o.argmax(0).reshape(-1).cpu() if task == "seg" else None
            pvec = o.permute(1, 2, 0).reshape(-1, ch).cpu() if task != "seg" else None
            cov = torch.ones(HS * WS, dtype=torch.bool)
        else:
            num = np.zeros((HS * WS, 1 if task == "seg" else ch), np.float32); rbuf = np.full(HS * WS, np.inf)
            cid_a, r_a, v_a = [], [], []
            for f, (cid, r) in zip(feats, cells):
                o = dec(f.float()[None].to(DEVICE))[0]
                if task == "seg":
                    v = F.interpolate(o.argmax(0)[None, None].float(), size=(SR, SR), mode="nearest")[0, 0].cpu().numpy().reshape(-1, 1)
                else:
                    v = F.interpolate(o[None], size=(SR, SR), mode="bilinear", align_corners=False)[0].permute(1, 2, 0).reshape(-1, ch).cpu().numpy()
                cid_a.append(cid); r_a.append(r); v_a.append(v)
            cid_a = np.concatenate(cid_a); r_a = np.concatenate(r_a); v_a = np.concatenate(v_a)
            order = np.argsort(-r_a); num[cid_a[order]] = v_a[order]; rbuf[cid_a[order]] = r_a[order]
            cov = torch.from_numpy(rbuf < np.inf)
            pseg = torch.from_numpy(num[:, 0].astype(np.int64)) if task == "seg" else None
            pvec = torch.from_numpy(num) if task != "seg" else None
        if task == "seg":
            g = sgt["seg"].reshape(-1); m = (g != P.IGNORE) & cov
            for c in range(1, P.N_CLASS):
                inter[c] += ((pseg == c) & (g == c) & m).sum(); union[c] += (((pseg == c) | (g == c)) & m).sum()
        elif task == "normal":
            gn = sgt["nrm"].reshape(-1, 3); mv = sgt["nval"].reshape(-1) & cov
            cos = (F.normalize(pvec, dim=1) * gn).sum(1).clamp(-1, 1)
            se += torch.rad2deg(torch.arccos(cos[mv])).sum().item(); sn += int(mv.sum())
        else:
            gd = sgt["dlog"].reshape(-1); mv = sgt["dval"].reshape(-1) & cov
            e = (pvec[:, 0] - gd).abs()[mv]; se += e.sum().item(); sd += (e < LOG125).float().sum().item(); sn += int(mv.sum())
    if task == "seg":
        return float(np.mean([(inter[c] / union[c]).item() for c in range(1, P.N_CLASS) if union[c] > 0]))
    return se / max(sn, 1) if task == "normal" else (se / max(sn, 1), sd / max(sn, 1))


def main():
    P.configure("stanford2d3d"); P.TILE = TILE; P.WORK_HW = (IH, IW)
    enc = PanoEncoder(model_id=P.MODEL, lora_rank=0).to(DEVICE).eval(); P.enc_patch = enc.patch
    files = data.list_erps("stanford2d3d")
    def area(f): return f.split("extracted_data/")[1].split("/")[0]
    tr = [f for f in files if "5" not in area(f)][:TR]; va = [f for f in files if "5" in area(f)][:VA]
    Mth = TC.methods()
    print(f"decomposition × SHARED decoder (PUP) × downstream loss | tasks={TASKS} tr={len(tr)} va={len(va)} ep={EPOCHS}", flush=True)
    res = {}
    for name, plan in Mth.items():
        ctr = cache_method(enc, tr, plan); cva = cache_method(enc, va, plan)
        for task in TASKS:
            res[(name, task)] = eval_sphere(task, train_dec(task, enc.dim, ctr), cva, plan)
            print(f"  {name:14s} {task:6s} -> {res[(name, task)]}", flush=True)
        del ctr, cva; torch.cuda.empty_cache()
    for task in TASKS:
        unit = {"seg": "sphere mIoU↑", "normal": "ang°↓", "depth": "|Δlog|↓ / δ↑"}[task]
        rows = sorted(((n, res[(n, task)]) for n in Mth),
                      key=(lambda r: -r[1]) if task == "seg" else (lambda r: r[1][0] if task == "depth" else r[1]))
        print(f"\n=== [{task}] {unit} (same PUP decoder, downstream-trained) ===")
        for n, v in rows:
            tag = " ⭐OURS" if n.startswith("e2p") else (" 📄Tangent" if n.startswith("tangent") else "")
            vs = f"{v[0]:.3f}/{v[1]:.2f}" if task == "depth" else f"{v:.3f}"
            print(f"   {vs}  {n}{tag}")


if __name__ == "__main__":
    main()
