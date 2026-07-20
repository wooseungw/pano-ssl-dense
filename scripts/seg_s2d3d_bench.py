"""Official Stanford2D3D-Panoramic (13-class) segmentation benchmark — places our
frozen-DINOv3 + E2P + decoder system on the SOTA scale (Trans4PASS / SGAT4PASS / HoHoNet).

Protocol (docs/SOTA_BENCHMARK_PLAN.md §A, verified vs Trans4PASS/SGAT4PASS source): official area
folds; dataset-AGGREGATED, class-unweighted mIoU over the 13 classes (void=0 ignored) — inter/union
accumulated over ALL test panos, then per-class IoU, then mean over 13 (absent class -> IoU 0).
E2P FULL-SPHERE tiles (hfov65 3-ring + pole caps: every valid GT cell gets >=1 tile; band-only left
the top/bottom ~12.5deg uncovered -> valid ceiling/floor pixels scored as false-negative void = a
real mIoU understatement) -> encoder -> conv decoder -> per-tile logits -> coverage-normalized logit
STITCH to the ERP grid -> argmax. GT is loaded at the eval resolution (P.WORK_HW = EVAL_HW). Encoder
is FROZEN (a loaded SSL adapter is FIXED, not trained here); the ~2.4M decoder is the only trained
module — our honest param-efficient probe.

NOTE (validity): SSL-adapter rows are TRANSDUCTIVE — the adapter's SSL pretraining pool includes the
test areas (no fold holdout), unlike the ImageNet/LVD-pretrained SOTA baselines. Report the FROZEN
row as the clean headline; caveat SSL/scaled rows accordingly.

Encoders via ENC_ADAPTER: unset = frozen DINOv3; else an adapter dir (SSL-810 / TC3 / 21.8k-scaled).
Saves runs/<stamp>_seg_s2d3d_<tag>_f<fold>/ : config + head weights + GT-alongside viz + vs-SOTA table.

Run: ENC_ADAPTER= FOLD=1 EPOCHS=20 CUDA_VISIBLE_DEVICES=1 python scripts/seg_s2d3d_bench.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import geometry  # noqa: E402
import data  # noqa: E402
import probe_seg_dinov3 as P  # noqa: E402
import runlog  # noqa: E402
from encoder import PanoEncoder, normalize_tiles  # noqa: E402

DEVICE = P.DEVICE
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAPTER = os.environ.get("ENC_ADAPTER", "").strip()      # "" -> frozen DINOv3
TAG = os.environ.get("TAG") or (os.path.basename(ADAPTER.rstrip("/")) if ADAPTER else "frozen")
FOLD = int(os.environ.get("FOLD", 1))
HFOV, TILE, HEAD_OUT = 65.0, 512, 128
EH, EW = (lambda s: (int(s[0]), int(s[1])))(os.environ.get("EVAL_HW", "512,1024").split(","))
# STITCH res must MATCH tile patch density (~2 deg/cell); the ERP is far finer, so stitching
# at EVAL res leaves holes (coverage collapse -> void). Stitch coverage-complete, then upsample.
SH, SW = (lambda s: (int(s[0]), int(s[1])))(os.environ.get("STITCH_HW", "128,256").split(","))
TILE_OUT = int(os.environ.get("TILE_OUT", 256))          # per-tile logit res for the stitch
EPOCHS = int(os.environ.get("EPOCHS", 20))
TR_PANOS = int(os.environ.get("TR_PANOS", 2000))         # headline: full non-test complement (~1040)
VA_PANOS = int(os.environ.get("VA_PANOS", 1000))         # headline: ALL test panos (area5 = 373)
CHUNK, NUM_WORKERS, SEED = 8, int(os.environ.get("NUM_WORKERS", 6)), 0

# Stanford2D3D official 3-fold — VERIFIED vs Trans4PASS/SGAT4PASS dataloaders (area NUMBERS in the
# TEST set; train = complement): fold1 test=area5; fold2 test=area2+4 (hardest); fold3 test=area1+3+6.
FOLD_TEST = {1: set("5"), 2: set("24"), 3: set("136")}

# Verified SOTA (docs/SOTA_BENCHMARK_PLAN.md §A) — (method, fold1 mIoU%, 3-fold avg mIoU%).
SOTA = [("SGAT4PASS-S (2023, SOTA)", 56.4, 55.3), ("Trans4PASS+ -S (2023)", 53.6, 53.7),
        ("Trans4PASS-S (2022)", 53.3, 52.1), ("HoHoNet (2021)", 53.9, 52.0)]

PLAN = None                                              # set in main() after configure


class SegHead(nn.Module):
    """(B,D,32,32) patch features -> (B,C,128,128) logits (4x bilinear upsample)."""

    def __init__(self, d, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(d, 256, 3, padding=1), nn.GroupNorm(16, 256), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 256, 3, padding=1), nn.GroupNorm(16, 256), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, c, 1))

    def forward(self, x):
        return self.net(x)


def area_num(f):
    a = f.split("extracted_data/")[1].split("/")[0]      # e.g. 'area_5a'
    return a.replace("area_", "")[0]                      # '5'


def build_encoder():
    enc = (PanoEncoder(model_id=P.MODEL, adapter_path=ADAPTER) if ADAPTER
           else PanoEncoder(model_id=P.MODEL, lora_rank=0))
    return enc.to(DEVICE).eval()                          # frozen: features cached once (decoder-only probe)


def render_pano(f):
    """CPU: (T,3,512,512) RGB tiles + (T,128,128) label tiles for one pano."""
    rgb = np.array(Image.open(f).convert("RGB").resize((1024, 512), Image.BILINEAR))
    lab = P.load_rgb_label(f)[1]
    tiles, labs = [], []
    for tp in PLAN:
        t = np.asarray(P.a2p.erp_to_pinhole_tile(rgb, tp.yaw_deg, tp.pitch_deg, HFOV, TILE))
        tiles.append(torch.from_numpy(t).float().permute(2, 0, 1) / 255.0)
        gl = P.e2p_label(lab, tp.yaw_deg, tp.pitch_deg, HFOV, TILE)
        gl = np.array(Image.fromarray(gl.astype(np.uint8)).resize((HEAD_OUT, HEAD_OUT), Image.NEAREST))
        labs.append(torch.from_numpy(gl.astype(np.int64)))
    return torch.stack(tiles), torch.stack(labs)


@torch.no_grad()
def encode(enc, tiles):
    outs = []
    for s in range(0, tiles.shape[0], CHUNK):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            fe = P.dense(enc, normalize_tiles(tiles[s:s + CHUNK].to(DEVICE)))   # (b,D,32,32)
        outs.append(fe.float().cpu())
    return torch.cat(outs)


def build_cache(enc, files, want_lab_full):
    """Encode each pano's tiles ONCE (frozen encoder). Bounded prefetch -> flat RAM.
    Returns list of (feat fp16 (T,D,32,32), labs_tile int16 (T,128,128) or gt_full int)."""
    cache = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        it = iter(files)
        inflight = deque()
        for _ in range(NUM_WORKERS * 2):
            try:
                inflight.append(ex.submit(render_pano, next(it)))
            except StopIteration:
                break
        fi = 0
        while inflight:
            fut = inflight.popleft()
            try:
                inflight.append(ex.submit(render_pano, next(it)))
            except StopIteration:
                pass
            tiles, labs = fut.result()
            feat = encode(enc, tiles).half()
            if want_lab_full:
                gt = P.load_rgb_label(files[fi])[1]                      # (512,1024) full GT
                cache.append((feat, torch.from_numpy(gt.astype(np.int64))))
            else:
                cache.append((feat, labs.to(torch.int16)))
            fi += 1
    return cache


def coord_map(yaw, pitch):
    """Each of (TILE_OUT,TILE_OUT) tile pixels -> its STITCH-grid (SH,SW) cell id. Image-independent."""
    return torch.from_numpy(geometry.coord_cell_map(512, 1024, yaw, pitch, HFOV, TILE_OUT, SH, SW))


@torch.no_grad()
def predict_erp(head, feat, cids):
    """Stitch per-tile logits at the coverage-complete STITCH grid, bilinear-upsample the logit
    field to the EVAL grid, argmax. Returns (EH,EW) pred + stitch coverage fraction."""
    acc = torch.zeros(SH * SW, P.N_CLASS); cnt = torch.zeros(SH * SW, 1)
    for ti in range(feat.shape[0]):
        lg = head(feat[ti:ti + 1].float().to(DEVICE))                    # (1,C,128,128)
        lg = F.interpolate(lg, (TILE_OUT, TILE_OUT), mode="bilinear", align_corners=False)[0]
        flat = lg.permute(1, 2, 0).reshape(-1, P.N_CLASS).cpu()
        acc.index_add_(0, cids[ti], flat); cnt.index_add_(0, cids[ti], torch.ones(flat.shape[0], 1))
    field = (acc / cnt.clamp_min(1.0)).reshape(1, SH, SW, P.N_CLASS).permute(0, 3, 1, 2)  # (1,C,SH,SW)
    field = F.interpolate(field, (EH, EW), mode="bilinear", align_corners=False)          # -> eval res (dense)
    return field[0].argmax(0).numpy(), float((cnt.squeeze(1) > 0).float().mean())


def main():
    torch.manual_seed(SEED)
    global PLAN
    P.configure("stanford2d3d"); P.TILE = TILE
    P.WORK_HW = (EH, EW)                                  # load GT at eval resolution (native-ish)
    enc = build_encoder(); P.enc_patch = enc.patch
    PLAN = P.a2p.plan_tiles("full_sphere", HFOV, HFOV, 0.25)   # 3-ring + pole caps -> full coverage

    files = data.list_erps("stanford2d3d")
    test = FOLD_TEST[FOLD]
    tr = [f for f in files if area_num(f) not in test][:TR_PANOS]
    va = [f for f in files if area_num(f) in test][:VA_PANOS]
    print(f"Seg-S2D3D fold{FOLD} enc={TAG} tiles/pano={len(PLAN)} tr={len(tr)} va={len(va)} "
          f"eval={EH}x{EW} ep={EPOCHS} N_CLASS={P.N_CLASS}", flush=True)

    t0 = time.time()
    ctr = build_cache(enc, tr, want_lab_full=False)
    cva = build_cache(enc, va, want_lab_full=True)
    cids = [coord_map(tp.yaw_deg, tp.pitch_deg) for tp in PLAN]
    print(f"encoded {len(tr)}+{len(va)} panos ({time.time()-t0:.0f}s)", flush=True)

    torch.manual_seed(SEED); head = SegHead(enc.dim, P.N_CLASS).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), 1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(ignore_index=P.IGNORE)
    g = torch.Generator().manual_seed(SEED)
    for ep in range(EPOCHS):
        head.train(); tot, nb = 0.0, 0
        for i in torch.randperm(len(ctr), generator=g).tolist():
            feat, labs = ctr[i]
            opt.zero_grad()
            for s in range(0, feat.shape[0], CHUNK):
                yb = labs[s:s + CHUNK].long().to(DEVICE)
                if not (yb != P.IGNORE).any():
                    continue
                lg = head(feat[s:s + CHUNK].float().to(DEVICE))
                loss = lossf(lg, yb) * (yb.shape[0] / feat.shape[0])
                loss.backward(); tot += loss.item(); nb += 1
            opt.step()
        if (ep + 1) % 5 == 0 or ep == EPOCHS - 1:
            print(f"  ep{ep+1}/{EPOCHS} loss={tot/max(nb,1):.3f}", flush=True)

    head.eval()
    inter = torch.zeros(P.N_CLASS); union = torch.zeros(P.N_CLASS)
    cov_sum = 0.0
    for feat, gt_full in cva:
        pred, stitch_cov = predict_erp(head, feat, cids)
        cov_sum += stitch_cov
        gt = np.array(Image.fromarray(gt_full.numpy().astype(np.uint8)).resize((EW, EH), Image.NEAREST)).astype(np.int64)
        m = gt != P.IGNORE
        for c in range(1, P.N_CLASS):
            pc, gc = (pred == c) & m, (gt == c) & m
            inter[c] += (pc & gc).sum(); union[c] += (pc | gc).sum()
    present = [c for c in range(1, P.N_CLASS) if union[c] > 0]
    ious = {P.S2D3D_CLASSES[c - 1]: ((inter[c] / union[c]).item() if union[c] > 0 else 0.0)
            for c in range(1, P.N_CLASS)}                 # absent -> 0.0: keep denom = 13 (SOTA convention)
    miou = float(np.mean(list(ious.values())))            # dataset-aggregated mean over ALL 13 classes
    coverage = cov_sum / max(len(cva), 1)
    dec_M = sum(p.numel() for p in head.parameters()) / 1e6
    ours = (f"OURS (frozen DINOv3 + {dec_M:.2f}M dec)" if not ADAPTER
            else f"OURS (DINOv3 + 0.59M SSL-LoRA[{TAG}] + {dec_M:.2f}M dec)")

    print(f"\n=== Seg-S2D3D fold{FOLD} | {ours} | stitch {SH}x{SW}->eval {EH}x{EW} mIoU = {miou*100:.2f} "
          f"| {len(present)}/13 classes | stitch coverage {coverage*100:.1f}% ===", flush=True)
    print(f"{'method':46s} fold1  3fold", flush=True)
    print(f"{ours:46s} {miou*100:5.1f}   —", flush=True)
    for name, f1, f3 in SOTA:
        print(f"{name:46s} {f1:5.1f}  {f3:5.1f}", flush=True)
    leak = "" if not ADAPTER else " · SSL row is TRANSDUCTIVE (adapter pool incl. test areas)"
    print(f"caveats: our eval fold{FOLD}, {EH}x{EW} (SOTA ~1024x2048, native 2048x4096), decoder-only on "
          f"FROZEN features, {len(tr)}-pano train / {len(va)} test; SOTA = full fine-tune.{leak}", flush=True)
    if len(present) < 13:
        print(f"WARNING: only {len(present)}/13 classes present in the {len(va)} test panos — raise VA_PANOS.", flush=True)

    run = runlog.create_run(f"seg_s2d3d_{TAG}_f{FOLD}", {
        "benchmark": "Stanford2D3D-Panoramic 13-class seg", "fold": FOLD, "encoder": ADAPTER or "frozen",
        "tag": TAG, "eval_hw": [EH, EW], "stitch_hw": [SH, SW], "tile_out": TILE_OUT, "epochs": EPOCHS,
        "tr_panos": len(tr), "va_panos": len(va), "mIoU": miou, "per_class_IoU": ious,
        "classes_present": len(present), "valid_pixel_coverage": coverage, "decoder_M": dec_M,
        "transductive": bool(ADAPTER),
        "sota": [{"method": n, "fold1": f1, "3fold": f3} for n, f1, f3 in SOTA],
        "protocol_caveats": "our eval (single fold, not 3-fold-avg); decoder-only on frozen features; "
        f"eval {EH}x{EW} vs SOTA ~1024x2048; SSL rows transductive={bool(ADAPTER)}"})
    torch.save(head.state_dict(), os.path.join(run, "weights", "seghead.pt"))

    pal = runlog.seg_palette(P.N_CLASS)
    for i, (feat, gt_full) in enumerate(cva[:3]):
        pred, _ = predict_erp(head, feat, cids)
        rgb = np.array(Image.open(va[i]).convert("RGB").resize((EW, EH), Image.BILINEAR)).astype(np.float32) / 255.0
        gt = np.array(Image.fromarray(gt_full.numpy().astype(np.uint8)).resize((EW, EH), Image.NEAREST)).astype(np.int64)
        runlog.save_seg_sample(run, "s2d3d", i, rgb, gt, {TAG: pred}, pal, scale=1)
        runlog.save_panel(run, "s2d3d", i, [("input ERP", rgb),
                          ("GT (13-cls)", runlog.colorize(gt, pal)),
                          (f"pred: {TAG} mIoU={miou*100:.1f}", runlog.colorize(pred, pal))])
    print(f"saved -> {run} (config + weights + viz + SOTA table)", flush=True)


if __name__ == "__main__":
    main()
