"""Fine-tune a conv seg decoder on top of the encoder (frozen / scratch-LoRA / SSL-LoRA)
and report full-ERP mIoU on Stanford2D3D area-5 — to place us on the SOTA scale and test
whether the SSL-pretrained adapter helps *fine-tuned accuracy* (predicted: barely, since
accuracy is teacher-bounded; the adapter's value is consistency).

E2P tiles (hfov65, 3-ring) -> encoder -> conv decoder -> per-tile logits -> stitched to a
(128x256) ERP grid (logit-summed over overlaps) -> argmax -> mIoU vs ERP GT.

Run: CUDA_VISIBLE_DEVICES=0 conda run -n pano python scripts/fine_tune_seg.py <frozen|lora_scratch|lora_ssl>
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from py360convert import e2p

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data  # noqa: E402
import probe_seg_dinov3 as P  # noqa: E402
import train_ssl as T  # noqa: E402
from encoder import PanoEncoder, normalize_tiles  # noqa: E402

DEVICE = P.DEVICE
COND = sys.argv[1] if len(sys.argv) > 1 else "frozen"
TILE, SEED = 512, 0
HS, WS = 128, 256                      # ERP stitch resolution
EPOCHS = int(os.environ.get("EPOCHS", 8))
TR_PANOS = int(os.environ.get("TR_PANOS", 150))
CHUNK = 8


class SegHead(nn.Module):
    def __init__(self, d, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(d, 256, 3, padding=1), nn.GroupNorm(16, 256), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 256, 3, padding=1), nn.GroupNorm(16, 256), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, c, 1))

    def forward(self, x):
        return self.net(x)             # (B,d,32,32) -> (B,c,128,128)


def build_encoder():
    if COND == "frozen":
        enc = PanoEncoder(model_id=P.MODEL, lora_rank=0)
    elif COND == "lora_scratch":
        enc = PanoEncoder(model_id=P.MODEL, lora_rank=16)
    elif COND == "lora_ssl":
        enc = PanoEncoder(model_id=P.MODEL, adapter_path=T.CKPT)
        for n, p in enc.named_parameters():
            if "lora_" in n:
                p.requires_grad = True
    else:
        raise ValueError(COND)
    return enc.to(DEVICE)


def tiles_labels(rgb, lab, plan, hfov):
    tiles, labs = [], []
    for tp in plan:
        t = np.asarray(P.a2p.erp_to_pinhole_tile(rgb, tp.yaw_deg, tp.pitch_deg, hfov, TILE))
        tiles.append(torch.from_numpy(t).float().permute(2, 0, 1) / 255.0)
        gl = P.e2p_label(lab, tp.yaw_deg, tp.pitch_deg, hfov, TILE)
        gl = np.array(Image.fromarray(gl.astype(np.uint8)).resize((128, 128), Image.NEAREST))
        labs.append(torch.from_numpy(gl.astype(np.int64)))
    return torch.stack(tiles), torch.stack(labs)


def coord128(h, w, yaw, pitch, hfov):
    uy = np.broadcast_to(np.arange(w, dtype=np.float32)[None], (h, w))
    vy = np.broadcast_to(np.arange(h, dtype=np.float32)[:, None], (h, w))
    um = e2p(uy[:, :, None], hfov, yaw, pitch, out_hw=(128, 128), mode="nearest")[:, :, 0]
    vm = e2p(vy[:, :, None], hfov, yaw, pitch, out_hw=(128, 128), mode="nearest")[:, :, 0]
    uf = np.clip((um / w * WS).astype(int), 0, WS - 1)
    vf = np.clip((vm / h * HS).astype(int), 0, HS - 1)
    return (vf * WS + uf).reshape(-1)


def main():
    P.configure("stanford2d3d"); P.TILE = TILE
    enc = build_encoder()
    frozen_enc = (COND == "frozen")
    if frozen_enc:
        enc.eval()
    head = SegHead(enc.dim, P.N_CLASS).to(DEVICE)
    plan = P.a2p.plan_tiles("band", 65.0, 65.0, 0.25, pmax_deg=45.0)
    hfov = 65.0

    files = data.list_erps("stanford2d3d")
    def area(f): return f.split("extracted_data/")[1].split("/")[0]
    tr_f = [f for f in files if "5" not in area(f)][:TR_PANOS]
    va_f = [f for f in files if "5" in area(f)][:40]
    print(f"COND={COND} N_CLASS={P.N_CLASS} tiles/pano={len(plan)} tr={len(tr_f)} va={len(va_f)} "
          f"epochs={EPOCHS} enc_trainable={sum(p.numel() for p in enc.parameters() if p.requires_grad)/1e6:.2f}M",
          flush=True)

    pg = [{"params": head.parameters(), "lr": 1e-3}]
    if not frozen_enc:
        pg.append({"params": [p for p in enc.parameters() if p.requires_grad], "lr": 1e-4})
    opt = torch.optim.AdamW(pg, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(ignore_index=P.IGNORE)

    cache = [(np.array(Image.open(f).convert("RGB").resize((1024, 512), Image.BILINEAR)),
              P.load_rgb_label(f)[1]) for f in tr_f]
    g = torch.Generator().manual_seed(SEED)
    for ep in range(EPOCHS):
        order = torch.randperm(len(cache), generator=g).tolist()
        tot, nb = 0.0, 0
        for i in order:
            rgb, lab = cache[i]
            tiles, labs = tiles_labels(rgb, lab, plan, hfov)
            opt.zero_grad()
            for s in range(0, tiles.shape[0], CHUNK):
                tb = normalize_tiles(tiles[s:s + CHUNK].to(DEVICE))
                yb = labs[s:s + CHUNK].to(DEVICE)
                if not (yb != P.IGNORE).any():                  # all-void chunk -> CE would be nan
                    continue
                with torch.set_grad_enabled(not frozen_enc):
                    feat = enc(tb)
                logit = head(feat)
                loss = lossf(logit, yb) * (tb.shape[0] / tiles.shape[0])
                loss.backward(); tot += loss.item(); nb += 1
            opt.step()
        print(f"  ep{ep} loss={tot/max(nb,1):.3f}", flush=True)

    # eval: stitch to ERP (HS,WS)
    head.eval()
    inter = torch.zeros(P.N_CLASS); union = torch.zeros(P.N_CLASS)
    for f in va_f:
        rgb = np.array(Image.open(f).convert("RGB").resize((1024, 512), Image.BILINEAR))
        lab = P.load_rgb_label(f)[1]
        gt = np.array(Image.fromarray(lab.astype(np.uint8)).resize((WS, HS), Image.NEAREST)).astype(np.int64)
        acc = torch.zeros(HS * WS, P.N_CLASS); cnt = torch.zeros(HS * WS, 1)
        for tp in plan:
            t = np.asarray(P.a2p.erp_to_pinhole_tile(rgb, tp.yaw_deg, tp.pitch_deg, hfov, TILE))
            x = normalize_tiles(torch.from_numpy(t).float().permute(2, 0, 1)[None].to(DEVICE) / 255.0)
            with torch.no_grad():
                lg = head(enc(x))[0]                    # (C,128,128)
            cid = torch.from_numpy(coord128(512, 1024, tp.yaw_deg, tp.pitch_deg, hfov))
            flat = lg.permute(1, 2, 0).reshape(-1, P.N_CLASS).cpu()
            acc.index_add_(0, cid, flat); cnt.index_add_(0, cid, torch.ones(flat.shape[0], 1))
        pred = acc.argmax(1).numpy().reshape(HS, WS)
        m = gt != P.IGNORE
        for c in range(1, P.N_CLASS):
            pc, gc = (pred == c) & m, (gt == c) & m
            inter[c] += (pc & gc).sum(); union[c] += (pc | gc).sum()
    ious = [(inter[c] / union[c]).item() for c in range(1, P.N_CLASS) if union[c] > 0]
    miou = float(np.mean(ious))
    print(f"\n=== {COND}: full-ERP({HS}x{WS}) mIoU = {miou:.4f}  ({len(ious)} classes) ===", flush=True)


if __name__ == "__main__":
    main()
