"""Geometry-guided DEFORMABLE cross-attention fusion of E2P tile features into a dense
ERP feature field, vs naive scatter-average. Encoder fixed (frozen / SSL-LoRA).

Fair-retry version (more data + regularization to remove the overfit confound):
 - 250 Stanford2D3D train panos (was 60), CPU fp16 feature cache
 - FIXED sinusoidal pos (was 0.5M learnable), offset = tanh*0.2 (kept near geometric ref),
   dropout 0.1, weight decay 1e-2

Each ERP-cell query samples each covering tile at (geometric reference + LEARNED offset)
with LEARNED softmax attention over tiles x K=4 samples (sub-pixel, content-adaptive).
Field 64x128. Decode fused field -> seg mIoU on Stanford2D3D. frozen vs LoRA, naive vs deform.

Run: CUDA_VISIBLE_DEVICES=0 conda run -n pano python scripts/adaptive_field_deform.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data  # noqa: E402
import geometry as G  # noqa: E402
import probe_seg_dinov3 as P  # noqa: E402
import train_ssl as T  # noqa: E402
from encoder import PanoEncoder, normalize_tiles  # noqa: E402

DEVICE = P.DEVICE
TILE, HFOV, SEED = 512, 65.0, 0
ERP_H, ERP_W = 1024, 2048
HF, WF = 64, 128
NC = HF * WF
EPOCHS = int(os.environ.get("EPOCHS", 20))
TR_PANOS = int(os.environ.get("TR_PANOS", 250))


def pos2d(d):
    d4 = d // 4
    omega = 1.0 / (10000 ** (torch.arange(d4).float() / d4))
    ii, jj = torch.meshgrid(torch.arange(HF).float(), torch.arange(WF).float(), indexing="ij")
    ii, jj = ii.reshape(-1, 1), jj.reshape(-1, 1)
    pe = torch.cat([torch.sin(ii * omega), torch.cos(ii * omega),
                    torch.sin(jj * omega), torch.cos(jj * omega)], 1)
    return F.pad(pe, (0, d - pe.shape[1])) if pe.shape[1] < d else pe[:, :d]


def precompute_geom(plan):
    scatter, refs, covs = [], [], []
    for tp in plan:
        cm = G.render_coordmap(ERP_H, ERP_W, tp.yaw_deg, tp.pitch_deg, HFOV, TILE)
        cy = ((np.arange(32) + 0.5) * TILE / 32).astype(int)
        pe = cm[np.ix_(cy, cy)]
        uf = np.clip((pe[..., 0] / ERP_W * WF).astype(int), 0, WF - 1)
        vf = np.clip((pe[..., 1] / ERP_H * HF).astype(int), 0, HF - 1)
        scatter.append(torch.from_numpy((vf * WF + uf).reshape(-1)).long().to(DEVICE))
        tr, tc = np.mgrid[0:TILE, 0:TILE]
        cu = np.clip((cm[..., 0] / ERP_W * WF).astype(int), 0, WF - 1)
        cv = np.clip((cm[..., 1] / ERP_H * HF).astype(int), 0, HF - 1)
        cell = (cv * WF + cu).reshape(-1)
        rs = np.zeros((NC, 2), np.float64); cnt = np.zeros(NC)
        np.add.at(rs[:, 0], cell, (tc / TILE * 2 - 1).reshape(-1))
        np.add.at(rs[:, 1], cell, (tr / TILE * 2 - 1).reshape(-1))
        np.add.at(cnt, cell, 1)
        refs.append(torch.from_numpy(rs / np.maximum(cnt[:, None], 1)).float().to(DEVICE))
        covs.append(torch.from_numpy(cnt > 0).to(DEVICE))
    return scatter, refs, covs


def naive_field(feats, scatter):
    fs = torch.zeros(NC, feats[0].shape[0], device=DEVICE); ws = torch.zeros(NC, 1, device=DEVICE)
    for f, sc in zip(feats, scatter):
        fmap = f.reshape(f.shape[0], -1).t()
        fs.index_add_(0, sc, fmap); ws.index_add_(0, sc, torch.ones(fmap.shape[0], 1, device=DEVICE))
    return fs / ws.clamp_min(1.0), (ws[:, 0] > 0)


class DeformFusion(nn.Module):
    def __init__(self, d, k=4, off_scale=0.2, drop=0.1):
        super().__init__()
        self.k, self.off_scale = k, off_scale
        self.register_buffer("pos", pos2d(d))
        self.off = nn.Linear(d, k * 2); self.lg = nn.Linear(d, k)
        self.vproj = nn.Linear(d, d); self.drop = nn.Dropout(drop)
        self.norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, d))
        nn.init.zeros_(self.off.weight); nn.init.zeros_(self.off.bias)

    def forward(self, feats, naive, scatter, refs, covs):
        q = naive + self.pos
        d = naive.shape[1]
        num = torch.zeros(NC, d, device=DEVICE); den = torch.zeros(NC, 1, device=DEVICE)
        for f, ref, cov in zip(feats, refs, covs):
            idx = cov.nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            qc = q[idx]
            off = torch.tanh(self.off(qc)).view(-1, self.k, 2) * self.off_scale
            pts = (ref[idx][:, None, :] + off).clamp(-1.2, 1.2)
            samp = F.grid_sample(f[None], pts[None], align_corners=False)[0]
            v = self.drop(self.vproj(samp.permute(1, 2, 0)))
            e = self.lg(qc).exp()
            num.index_add_(0, idx, (e[..., None] * v).sum(1))
            den.index_add_(0, idx, e.sum(1, keepdim=True))
        fused = num / den.clamp_min(1e-6)
        return fused + self.ffn(self.norm(fused))


@torch.no_grad()
def encode(enc, rgb, plan):
    out = []
    for tp in plan:
        t = np.asarray(P.a2p.erp_to_pinhole_tile(rgb, tp.yaw_deg, tp.pitch_deg, HFOV, TILE))
        x = torch.from_numpy(t).float().permute(2, 0, 1)[None] / 255.0
        out.append(P.dense(enc, normalize_tiles(x.to(DEVICE)))[0].half().cpu())   # fp16 CPU cache
    return out


def dev(feats):
    return [t.to(DEVICE).float() for t in feats]


def gt_field(lab):
    return torch.from_numpy(P.label_to_grid(lab, HF, WF).reshape(-1))


def run_encoder(tag, enc, tr, va, scatter, refs, covs):
    P.enc_patch = enc.patch
    cache = {"tr": [], "va": []}
    for sp, fl in [("tr", tr), ("va", va)]:
        for f in fl:
            rgb, lab = P.load_rgb_label(f)
            cache[sp].append((encode(enc, rgb, P.plan), gt_field(lab)))

    def naive_probe():
        Xtr, ytr, Xva, yva = [], [], [], []
        for feats, y in cache["tr"]:
            nf, cov = naive_field(dev(feats), scatter); Xtr.append(nf[cov].cpu()); ytr.append(y[cov.cpu()])
        for feats, y in cache["va"]:
            nf, cov = naive_field(dev(feats), scatter); Xva.append(nf[cov].cpu()); yva.append(y[cov.cpu()])
        Xtr, ytr = torch.cat(Xtr), torch.cat(ytr); Xva, yva = torch.cat(Xva), torch.cat(yva)
        Xtr, ytr = P.subsample(Xtr, ytr, 300000, SEED)
        torch.manual_seed(SEED); clf = nn.Linear(Xtr.shape[1], P.N_CLASS).to(DEVICE)
        opt = torch.optim.Adam(clf.parameters(), 1e-3, weight_decay=1e-4)
        lf = nn.CrossEntropyLoss(ignore_index=P.IGNORE); Xt, yt = Xtr.to(DEVICE).float(), ytr.to(DEVICE)
        for _ in range(800):
            opt.zero_grad(); lf(clf(Xt), yt).backward(); opt.step()
        with torch.no_grad():
            pr = clf(Xva.to(DEVICE).float()).argmax(1).cpu()
        return P.miou_acc(pr, yva)[0]

    torch.manual_seed(SEED)
    fuse = DeformFusion(enc.dim).to(DEVICE); decd = nn.Linear(enc.dim, P.N_CLASS).to(DEVICE)
    opt = torch.optim.AdamW(list(fuse.parameters()) + list(decd.parameters()), 1e-3, weight_decay=1e-2)
    lf = nn.CrossEntropyLoss(ignore_index=P.IGNORE)
    g = torch.Generator().manual_seed(SEED)
    fuse.train()
    for ep in range(EPOCHS):
        for i in torch.randperm(len(cache["tr"]), generator=g).tolist():
            feats, y = cache["tr"][i]; fd = dev(feats); yd = y.to(DEVICE)
            nf, cov = naive_field(fd, scatter)
            out = fuse(fd, nf, scatter, refs, covs)
            opt.zero_grad(); lf(decd(out[cov]), yd[cov]).backward(); opt.step()
    fuse.eval()
    inter = torch.zeros(P.N_CLASS); union = torch.zeros(P.N_CLASS)
    with torch.no_grad():
        for feats, y in cache["va"]:
            fd = dev(feats); nf, cov = naive_field(fd, scatter)
            pred = decd(fuse(fd, nf, scatter, refs, covs)).argmax(1).cpu()
            cm = cov.cpu(); mm = (y != P.IGNORE) & cm
            for c in range(1, P.N_CLASS):
                inter[c] += ((pred == c) & (y == c) & mm).sum(); union[c] += (((pred == c) | (y == c)) & mm).sum()
    deform = float(np.mean([(inter[c] / union[c]).item() for c in range(1, P.N_CLASS) if union[c] > 0]))
    nv = naive_probe()
    print(f"{tag:8s}  naive-scatter={nv:.3f}   deformable={deform:.3f}   Δ={deform-nv:+.3f}", flush=True)
    del cache; torch.cuda.empty_cache()


def main():
    P.configure("stanford2d3d"); P.TILE = TILE
    P.plan = P.a2p.plan_tiles("band", HFOV, HFOV, 0.25, pmax_deg=45.0)
    scatter, refs, covs = precompute_geom(P.plan)
    files = data.list_erps("stanford2d3d")
    def area(f): return f.split("extracted_data/")[1].split("/")[0]
    tr = [f for f in files if "5" not in area(f)][:TR_PANOS]
    va = [f for f in files if "5" in area(f)][:40]
    print(f"deformable field fusion (fair-retry): field={HF}x{WF} tiles={len(P.plan)} K=4 "
          f"tr={len(tr)} va={len(va)} ep={EPOCHS} seed={SEED}\n{'encoder':8s}  {'naive':>13}   {'deformable':>11}", flush=True)
    for tag, kw in [("frozen", dict(lora_rank=0)), ("LoRA", dict(adapter_path=T.CKPT))]:
        enc = PanoEncoder(model_id=P.MODEL, **kw).to(DEVICE).eval()
        run_encoder(tag, enc, tr, va, scatter, refs, covs)
        del enc; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
