"""Per-task eval of the COMMON SSL encoder (ckpt_ssl_lora, FIXED) vs frozen, paired with a
ZOO of decoder heads drawn from real SOTA models (not DPT):
  Linear  - Segmenter-Linear / SETR-Naive  (1x1 conv -> upsample)
  PUP     - SETR-PUP                        (progressive conv upsampling)
  UPerNet - UPerNet PPM head                (pyramid pooling + fuse)
  Mask    - Segmenter-Mask / Mask2Former    (class-query transformer -> masks; seg only)

Encoder fixed; only the decoder trains -> isolates encoder feature quality per task/decoder.
Stanford2D3D, E2P tiles hfov65, area5 val, tile-pixel metrics @128.
  seg -> 13-cls mIoU | normal -> mean ang err° | depth -> |Δlog| & δ<1.25

Run: OPENCV_IO_ENABLE_OPENEXR=1 CUDA_VISIBLE_DEVICES=0 conda run -n pano python scripts/multitask_eval.py
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
import probe_seg_dinov3 as P  # noqa: E402
import train_ssl as T  # noqa: E402
import probe_normal as PN  # noqa: E402
from encoder import PanoEncoder, normalize_tiles  # noqa: E402

DEVICE = P.DEVICE
TILE, HFOV, SEED, GH = 512, 65.0, 0, 128
EPOCHS = int(os.environ.get("EPOCHS", 15))
TR, VA = int(os.environ.get("TR", 80)), int(os.environ.get("VA", 30))
LOG125 = math.log(1.25)


# ----------------------------------------------------------------- decoder zoo
def up(x, s=2):
    return F.interpolate(x, scale_factor=s, mode="bilinear", align_corners=False)


class Linear(nn.Module):                                   # Segmenter-Linear / SETR-Naive
    def __init__(self, d, c):
        super().__init__(); self.h = nn.Conv2d(d, c, 1)

    def forward(self, x):
        return F.interpolate(self.h(x), size=GH, mode="bilinear", align_corners=False)


class PUP(nn.Module):                                      # SETR-PUP
    def __init__(self, d, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(d, 256, 3, padding=1), nn.GroupNorm(16, 256), nn.GELU(), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1), nn.GroupNorm(16, 128), nn.GELU(), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, c, 1))

    def forward(self, x):
        return self.net(x)


class UPerNet(nn.Module):                                  # UPerNet PPM head (single level)
    def __init__(self, d, c, scales=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(s), nn.Conv2d(d, 128, 1),
                                                   nn.GroupNorm(8, 128), nn.GELU()) for s in scales])
        self.fuse = nn.Sequential(
            nn.Conv2d(d + 128 * len(scales), 256, 3, padding=1), nn.GroupNorm(16, 256), nn.GELU(), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1), nn.GroupNorm(16, 128), nn.GELU(), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, c, 1))

    def forward(self, x):
        h, w = x.shape[-2:]
        feats = [x] + [F.interpolate(s(x), size=(h, w), mode="bilinear", align_corners=False) for s in self.stages]
        return self.fuse(torch.cat(feats, 1))


class Mask(nn.Module):                                     # Segmenter-Mask / Mask2Former-style
    def __init__(self, d, c, depth=2):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(c, d) * 0.02)
        layer = nn.TransformerDecoderLayer(d, 4, d * 2, batch_first=True, dropout=0.0)
        self.dec = nn.TransformerDecoder(layer, depth)
        self.proj = nn.Linear(d, d); self.scale = d ** -0.5

    def forward(self, x):
        b, d, h, w = x.shape
        patch = x.flatten(2).transpose(1, 2)               # (B,N,d)
        q = self.dec(self.cls[None].expand(b, -1, -1), patch)  # (B,c,d)
        m = torch.einsum("bnd,bcd->bcn", self.proj(patch), q) * self.scale
        return F.interpolate(m.reshape(b, -1, h, w), size=GH, mode="bilinear", align_corners=False)


ZOO = {"Linear": Linear, "PUP": PUP, "UPerNet": UPerNet, "Mask": Mask}
TASK_DECODERS = {"seg": ["Linear", "PUP", "UPerNet", "Mask"],
                 "normal": ["Linear", "PUP", "UPerNet"], "depth": ["Linear", "PUP", "UPerNet"]}
OUT_CH = {"seg": None, "normal": 3, "depth": 1}            # seg set at runtime


# ----------------------------------------------------------------- data
def load_depth(f):
    d = np.array(Image.open(data.s2d3d_gt_path(f, "depth")).resize((1024, 512), Image.NEAREST)).astype(np.float32)
    v = (d > 0) & (d < 65535); med = np.median(d[v]) if v.any() else 1.0
    return (d / med), v.astype(np.float32)


@torch.no_grad()
def encode_pano(enc, f):
    rgb = np.array(Image.open(f).convert("RGB").resize((1024, 512), Image.BILINEAR))
    seg = P.load_rgb_label(f)[1]; nrm, nval = PN.load_rgb_normal(f)[1:]; dn, dval = load_depth(f)
    feats, gts = [], []
    for tp in P.plan:
        tile = np.asarray(P.a2p.erp_to_pinhole_tile(rgb, tp.yaw_deg, tp.pitch_deg, HFOV, TILE))
        x = torch.from_numpy(tile).float().permute(2, 0, 1)[None] / 255.0
        feats.append(P.dense(enc, normalize_tiles(x.to(DEVICE)))[0].half().cpu())
        w = lambda a, ch=1: PN.warp_to_grid(a, tp.yaw_deg, tp.pitch_deg, HFOV, GH, GH, ch)
        nm = w(nrm, 3); nm = nm / np.clip(np.linalg.norm(nm, axis=2, keepdims=True), 1e-6, None)
        dd = w(dn[:, :, None])[:, :, 0]
        gts.append(dict(seg=torch.from_numpy(w(seg[:, :, None])[:, :, 0].astype(np.int64)),
                        nrm=torch.from_numpy(nm).float(), nval=torch.from_numpy(w(nval[:, :, None])[:, :, 0] > 0.5),
                        dlog=torch.from_numpy(np.log(np.clip(dd, 1e-3, None))).float(),
                        dval=torch.from_numpy((w(dval[:, :, None])[:, :, 0] > 0.5) & (dd > 1e-3))))
    return feats, gts


def loss_of(task, out, gts):
    if task == "seg":
        return F.cross_entropy(out, torch.stack([g["seg"] for g in gts]).to(DEVICE), ignore_index=P.IGNORE)
    if task == "normal":
        y = torch.stack([g["nrm"] for g in gts]).permute(0, 3, 1, 2).to(DEVICE)
        m = torch.stack([g["nval"] for g in gts]).to(DEVICE)
        c = (F.normalize(out, dim=1) * y).sum(1)
        return (1 - c)[m].mean() if m.any() else None
    y = torch.stack([g["dlog"] for g in gts]).to(DEVICE); m = torch.stack([g["dval"] for g in gts]).to(DEVICE)
    return F.l1_loss(out[:, 0][m], y[m]) if m.any() else None


def train_eval(task, name, dim, ctr, cva):
    c = P.N_CLASS if task == "seg" else OUT_CH[task]
    torch.manual_seed(SEED); dec = ZOO[name](dim, c).to(DEVICE)
    opt = torch.optim.AdamW(dec.parameters(), 1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(SEED)
    for _ in range(EPOCHS):
        for i in torch.randperm(len(ctr), generator=g).tolist():
            feats, gts = ctr[i]; opt.zero_grad()
            for s in range(0, len(feats), 8):
                fb = torch.stack([f.float() for f in feats[s:s + 8]]).to(DEVICE)
                ls = loss_of(task, dec(fb), gts[s:s + 8])
                if ls is not None:
                    (ls * fb.shape[0] / len(feats)).backward()
            opt.step()
    return evaluate(task, dec, cva)


@torch.no_grad()
def evaluate(task, dec, cva):
    dec.eval()
    if task == "seg":
        inter = torch.zeros(P.N_CLASS); union = torch.zeros(P.N_CLASS)
        for feats, gts in cva:
            for f, gt in zip(feats, gts):
                pr = dec(f.float()[None].to(DEVICE)).argmax(1)[0].cpu(); y = gt["seg"]; mm = y != P.IGNORE
                for c in range(1, P.N_CLASS):
                    inter[c] += ((pr == c) & (y == c) & mm).sum(); union[c] += (((pr == c) | (y == c)) & mm).sum()
        return float(np.mean([(inter[c] / union[c]).item() for c in range(1, P.N_CLASS) if union[c] > 0]))
    if task == "normal":
        tot, n = 0.0, 0
        for feats, gts in cva:
            for f, gt in zip(feats, gts):
                p = F.normalize(dec(f.float()[None].to(DEVICE)), dim=1)[0].permute(1, 2, 0).cpu()
                m = gt["nval"]; cos = (p * gt["nrm"]).sum(-1).clamp(-1, 1)
                tot += torch.rad2deg(torch.arccos(cos[m])).sum().item(); n += int(m.sum())
        return tot / max(n, 1)
    err, dac, n = 0.0, 0.0, 0
    for feats, gts in cva:
        for f, gt in zip(feats, gts):
            p = dec(f.float()[None].to(DEVICE))[0, 0].cpu(); m = gt["dval"]; e = (p[m] - gt["dlog"][m]).abs()
            err += e.sum().item(); dac += (e < LOG125).float().sum().item(); n += int(m.sum())
    return (err / max(n, 1), dac / max(n, 1))


def main():
    P.configure("stanford2d3d"); P.TILE = TILE
    P.plan = P.a2p.plan_tiles("band", HFOV, HFOV, 0.25, pmax_deg=45.0)
    files = data.list_erps("stanford2d3d")
    def area(f): return f.split("extracted_data/")[1].split("/")[0]
    tr = [f for f in files if "5" not in area(f)][:TR]; va = [f for f in files if "5" in area(f)][:VA]
    print(f"multitask × decoder-zoo: common SSL encoder vs frozen (encoder FIXED) | tr={len(tr)} va={len(va)} ep={EPOCHS}\n", flush=True)
    res = {}
    for tag, kw in [("frozen", dict(lora_rank=0)), ("SSL-LoRA", dict(adapter_path=T.CKPT))]:
        enc = PanoEncoder(model_id=P.MODEL, **kw).to(DEVICE).eval(); P.enc_patch = enc.patch
        ctr = [encode_pano(enc, f) for f in tr]; cva = [encode_pano(enc, f) for f in va]
        for task, decs in TASK_DECODERS.items():
            for name in decs:
                res[(tag, task, name)] = train_eval(task, name, enc.dim, ctr, cva)
                print(f"  [{tag}] {task:6s} {name:8s} -> {res[(tag, task, name)]}", flush=True)
        del enc, ctr, cva; torch.cuda.empty_cache()

    print("\n=== PER-TASK × DECODER: frozen vs common SSL encoder ===")
    for task, decs in TASK_DECODERS.items():
        unit = {"seg": "mIoU↑", "normal": "ang°↓", "depth": "|Δlog|↓ / δ↑"}[task]
        print(f"\n[{task}]  ({unit})\n{'decoder':9s} {'frozen':>16} {'SSL-LoRA':>16} {'Δ':>10}")
        for name in decs:
            f, l = res[("frozen", task, name)], res[("SSL-LoRA", task, name)]
            if task == "depth":
                print(f"{name:9s} {f[0]:7.3f}/{f[1]:.2f}   {l[0]:7.3f}/{l[1]:.2f}   {l[0]-f[0]:+.3f}/{l[1]-f[1]:+.2f}")
            else:
                d = l - f
                print(f"{name:9s} {f:16.3f} {l:16.3f} {d:+10.3f}")


if __name__ == "__main__":
    main()
