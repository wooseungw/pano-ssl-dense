"""Shared benchmark scaffold for Stanford2D3D dense-prediction SOTA comparison.

Both `depth_s2d3d_bench.py` (metric 360 depth) and `normal_s2d3d_bench.py` (surface
normals) build on this — the two dense tasks BESIDES segmentation that Stanford2D3D
carries aligned GT for (`seg_s2d3d_bench.py` covers segmentation). All three share one
protocol so numbers are comparable across tasks and encoders:

  * official Stanford2D3D 3-fold splits (fold1 test=area5, fold2=area2+4, fold3=area1+3+6);
  * FULL-SPHERE E2P tiling (hfov65 rings + pole caps) so every valid GT cell gets a tile;
  * FROZEN encoder — features cached once; only a small conv head trains (param-efficient probe);
  * COVERAGE-COMPLETE stitch — stitch per-tile fields at STITCH_HW (patch density ~2 deg/cell),
    then bilinear-upsample the field to EVAL_HW. Stitching directly at eval res leaves holes
    (coverage collapse -> the 21.7%-coverage / 15%-mIoU artifact in SOTA_BENCHMARK_PLAN §2.6).

Swap the model under test with env vars (mirrors seg_s2d3d_bench.py):
  ENC_ADAPTER=<dir>   an SSL LoRA adapter to evaluate (else frozen DINOv3);
  MODEL=<hf id>       a different HF backbone entirely (DINOv2, MAE, ...).
SSL-adapter rows are TRANSDUCTIVE (the adapter's SSL pretrain pool includes the test areas),
unlike the ImageNet/LVD-pretrained published baselines -> report the FROZEN row as the clean
headline and caveat adapter/backbone rows accordingly.
"""
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from py360convert import e2p

import anyres_e2p as a2p
from encoder import PanoEncoder, normalize_tiles

# --------------------------------------------------------------------- config
MODEL: str = os.environ.get("MODEL", "facebook/dinov3-vitb16-pretrain-lvd1689m")
ADAPTER: str = os.environ.get("ENC_ADAPTER", "").strip()          # "" -> frozen DINOv3
TAG: str = os.environ.get("TAG") or (
    os.path.basename(ADAPTER.rstrip("/")) if ADAPTER else "frozen")
FOLD: int = int(os.environ.get("FOLD", 1))
DEVICE: str = os.environ.get("DEV", "cuda" if torch.cuda.is_available() else "cpu")

HFOV, TILE, HEAD_OUT = 65.0, 512, 128                             # tile fov / render px / head px
OVERLAP: float = float(os.environ.get("OVERLAP", 0.25))
EH, EW = (lambda s: (int(s[0]), int(s[1])))(os.environ.get("EVAL_HW", "512,1024").split(","))
SH, SW = (lambda s: (int(s[0]), int(s[1])))(os.environ.get("STITCH_HW", "128,256").split(","))
TILE_OUT: int = int(os.environ.get("TILE_OUT", 256))             # per-tile field res for the stitch
EPOCHS: int = int(os.environ.get("EPOCHS", 20))
TR_PANOS: int = int(os.environ.get("TR_PANOS", 100000))          # default: full non-test complement
VA_PANOS: int = int(os.environ.get("VA_PANOS", 100000))          # default: all test panos
CHUNK, SEED = int(os.environ.get("CHUNK", 8)), int(os.environ.get("SEED", 0))

# Stanford2D3D official 3-fold (area NUMBERS in the TEST set; train = complement) — verified vs
# Trans4PASS/SGAT4PASS dataloaders in SOTA_BENCHMARK_PLAN §2.5.
FOLD_TEST = {1: set("5"), 2: set("24"), 3: set("136")}


def area_num(f: str) -> str:
    """extracted_data/area_5a/... -> '5' (the fold key)."""
    return f.split("extracted_data/")[1].split("/")[0].replace("area_", "")[0]


def split_files(files: List[str], fold: int) -> Tuple[List[str], List[str]]:
    """Official fold split: (train complement, test areas), capped by TR_PANOS/VA_PANOS."""
    test = FOLD_TEST[fold]
    tr = [f for f in files if area_num(f) not in test][:TR_PANOS]
    va = [f for f in files if area_num(f) in test][:VA_PANOS]
    return tr, va


def build_encoder() -> PanoEncoder:
    """FROZEN encoder under test: ENC_ADAPTER dir if set (SSL adapter), else frozen MODEL."""
    enc = (PanoEncoder(model_id=MODEL, adapter_path=ADAPTER) if ADAPTER
           else PanoEncoder(model_id=MODEL, lora_rank=0))
    return enc.to(DEVICE).eval()


def build_plan() -> List[a2p.TilePlan]:
    """Full-sphere tile schedule: cos(phi) rings + pole caps -> every valid GT cell covered."""
    return a2p.plan_tiles("full_sphere", HFOV, HFOV, OVERLAP)


class DenseHead(nn.Module):
    """(B,D,32,32) patch features -> (B,C,128,128) dense field (4x bilinear upsample).

    Same architecture as seg_s2d3d_bench.SegHead — a single honest conv head (NOT the
    multitask_eval decoder zoo); output channels C are the only per-task difference
    (C=1 log-depth, C=3 normal, C=N_CLASS seg)."""

    def __init__(self, d: int, c: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(d, 256, 3, padding=1), nn.GroupNorm(16, 256), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 256, 3, padding=1), nn.GroupNorm(16, 256), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, c, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _DeformAttn(nn.Module):
    """Single-scale deformable self-attention (Deformable-DETR style, pure-torch via grid_sample).

    Each of the H*W patch-token queries predicts n_points sampling offsets per head around its OWN
    grid location and attends to the bilinearly-sampled values -> a learned, distortion-adaptive
    receptive field (vs a fixed 3x3 conv). Offsets init to zero => starts at the reference point."""

    def __init__(self, dim: int, n_heads: int = 4, n_points: int = 4) -> None:
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.nh, self.np, self.hd = n_heads, n_points, dim // n_heads
        self.offset = nn.Linear(dim, n_heads * n_points * 2)
        self.attn = nn.Linear(dim, n_heads * n_points)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(self, x: torch.Tensor, ref: torch.Tensor, hw) -> torch.Tensor:
        b, n, d = x.shape
        h, w = hw
        v = self.value(x).reshape(b, n, self.nh, self.hd)
        vmap = v.permute(0, 2, 3, 1).reshape(b * self.nh, self.hd, h, w)          # (b*nh, hd, H, W)
        off = self.offset(x).reshape(b, n, self.nh, self.np, 2)
        aw = self.attn(x).reshape(b, n, self.nh, self.np).softmax(-1)
        scale = torch.tensor([2.0 / w, 2.0 / h], device=x.device, dtype=x.dtype)  # 1 px -> normalized
        loc = ref[:, :, None, None, :] + off * scale                             # (b,n,nh,np,2) in [-1,1]
        grid = loc.permute(0, 2, 1, 3, 4).reshape(b * self.nh, n, self.np, 2)
        samp = F.grid_sample(vmap, grid, mode="bilinear", align_corners=False)    # (b*nh, hd, n, np)
        samp = samp.reshape(b, self.nh, self.hd, n, self.np)
        out = (samp * aw.permute(0, 2, 1, 3)[:, :, None]).sum(-1)                 # (b,nh,hd,n)
        return self.out(out.permute(0, 3, 1, 2).reshape(b, n, d))


class ThinDeformDecoder(nn.Module):
    """Thin deformable-attention transformer decoder: (B,D,32,32) -> (B,C,128,128). Drop-in for
    DenseHead. n_layers deformable self-attn + FFN blocks on the tile's patch-token grid (each token
    samples a distortion-adaptive neighborhood), then a light conv upsample to head resolution."""

    def __init__(self, d_in: int, c: int, dim: int = 256, n_layers: int = 2,
                 n_heads: int = 4, n_points: int = 4, ffn: int = 2) -> None:
        super().__init__()
        self.proj = nn.Conv2d(d_in, dim, 1)
        self.norm1 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
        self.attn = nn.ModuleList([_DeformAttn(dim, n_heads, n_points) for _ in range(n_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
        self.ffn = nn.ModuleList([nn.Sequential(nn.Linear(dim, dim * ffn), nn.GELU(),
                                                nn.Linear(dim * ffn, dim)) for _ in range(n_layers)])
        self.up = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1), nn.GroupNorm(16, dim), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(dim, dim, 3, padding=1), nn.GroupNorm(16, dim), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(dim, c, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        f = self.proj(x)
        ys = torch.linspace(-1 + 1.0 / h, 1 - 1.0 / h, h, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1 + 1.0 / w, 1 - 1.0 / w, w, device=x.device, dtype=x.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        ref = torch.stack([gx, gy], -1).reshape(1, h * w, 2).expand(b, -1, -1)    # (b,N,2) grid_sample (x,y)
        t = f.flatten(2).transpose(1, 2)                                          # (b, N, dim)
        for i in range(len(self.attn)):
            t = t + self.attn[i](self.norm1[i](t), ref, (h, w))
            t = t + self.ffn[i](self.norm2[i](t))
        return self.up(t.transpose(1, 2).reshape(b, -1, h, w))


def make_head(d: int, c: int) -> nn.Module:
    """Decoder factory: DECODER=conv (DenseHead, default) | deform (ThinDeformDecoder)."""
    return ThinDeformDecoder(d, c) if os.environ.get("DECODER") == "deform" else DenseHead(d, c)


def render_tiles(rgb: np.ndarray, plan: List[a2p.TilePlan]) -> torch.Tensor:
    """(H,W,3) uint8 ERP -> (T,3,TILE,TILE) float[0,1] pinhole tiles for the plan."""
    ts = []
    for tp in plan:
        t = np.asarray(a2p.erp_to_pinhole_tile(rgb, tp.yaw_deg, tp.pitch_deg, HFOV, TILE))
        ts.append(torch.from_numpy(t).float().permute(2, 0, 1) / 255.0)
    return torch.stack(ts)


def warp_to_grid(arr: np.ndarray, yaw: float, pitch: float, gh: int, gw: int, ch: int) -> np.ndarray:
    """e2p-warp an (H,W,ch) ERP map to a tile (NEAREST), sample (gh,gw) centers -> (gh,gw,ch).

    Sampling nearest keeps discrete GT (labels, valid masks) exact and does NOT rotate
    vector-valued GT (normals) — so tile-space normals stay in the pano/world frame, which
    is what lets overlapping-tile predictions be averaged without per-tile frame rotation."""
    w = e2p(arr.astype(np.float32), HFOV, yaw, pitch, out_hw=(TILE, TILE), mode="nearest")
    if w.ndim == 2:
        w = w[:, :, None]
    cy = ((np.arange(gh) + 0.5) * TILE / gh).astype(int)
    cx = ((np.arange(gw) + 0.5) * TILE / gw).astype(int)
    return w[np.ix_(cy, cx)].reshape(gh, gw, ch)


@torch.no_grad()
def encode(enc: PanoEncoder, tiles: torch.Tensor) -> torch.Tensor:
    """(T,3,TILE,TILE) tiles -> (T,D,32,32) features (bf16 autocast on cuda), fp32 on CPU.

    Uses the encoder's forward path, so a loaded SSL adapter IS applied (student features);
    frozen when no adapter -> the clean headline."""
    outs = []
    for s in range(0, tiles.shape[0], CHUNK):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            fe = enc(normalize_tiles(tiles[s:s + CHUNK].to(DEVICE)))
        outs.append(fe.float().cpu())
    return torch.cat(outs)


# --- GPU E2P rendering: move the ERP->pinhole tiling off the CPU (py360convert) onto the GPU.
# The tile geometry is image-independent, so per plan we precompute ONE sampling grid per tile
# (continuous, seam-safe: cos/sin longitude sampled *through* py360 so the convention matches e2p
# to feature-cosine ~0.995), then render every pano with a single batched F.grid_sample on GPU.
GRIDS: torch.Tensor = None                                       # (T,TILE,TILE,2) RGB-render grid
GRIDS_GT: torch.Tensor = None                                    # (T,HEAD_OUT,HEAD_OUT,2) GT-warp grid


def _grids_for(plan: List[a2p.TilePlan], out_size: int) -> torch.Tensor:
    """(T,out_size,out_size,2) normalized ERP sampling grids, one per tile. Built by e2p-sampling a
    smooth [cos(lon),sin(lon),v] ERP (bilinear -> sub-pixel + seam-safe) and reconstructing
    continuous (u,v) — matches py360 e2p geometry within bf16 feature noise. Image-independent."""
    xx, yy = np.meshgrid(np.arange(EW), np.arange(EH))
    lon = xx / EW * 2 * np.pi
    coord = np.stack([np.cos(lon), np.sin(lon), yy.astype(np.float32)], -1).astype(np.float32)
    grids = []
    for tp in plan:
        s = e2p(coord, HFOV, tp.yaw_deg, tp.pitch_deg, out_hw=(out_size, out_size), mode="bilinear")
        u = (np.arctan2(s[:, :, 1], s[:, :, 0]) / (2 * np.pi)) % 1.0 * EW
        gx = u / (EW - 1) * 2 - 1
        gy = s[:, :, 2] / (EH - 1) * 2 - 1
        grids.append(np.stack([gx, gy], -1).astype(np.float32))
    return torch.from_numpy(np.stack(grids)).to(DEVICE)


def build_sample_grids(plan: List[a2p.TilePlan]) -> torch.Tensor:
    """Precompute GPU sampling grids once after build_plan(): GRIDS for RGB tile rendering (TILE res)
    and GRIDS_GT for warping GT maps to HEAD_OUT tiles. Moves all per-image e2p off the CPU."""
    global GRIDS, GRIDS_GT
    GRIDS = _grids_for(plan, TILE)
    GRIDS_GT = _grids_for(plan, HEAD_OUT)
    return GRIDS


@torch.no_grad()
def encode_erp(enc: PanoEncoder, rgb: np.ndarray) -> torch.Tensor:
    """(EH,EW,3) uint8 ERP -> (T,D,32,32) features. Renders all tiles with GPU grid_sample (no CPU
    py360), then encodes. Requires build_sample_grids(plan) to have run. CPU only does image I/O."""
    erp = torch.from_numpy(rgb).float().permute(2, 0, 1)[None].to(DEVICE) / 255.0   # (1,3,EH,EW)
    outs = []
    for s in range(0, GRIDS.shape[0], CHUNK):
        g = GRIDS[s:s + CHUNK]                                                       # (b,TILE,TILE,2)
        tiles = F.grid_sample(erp.expand(g.shape[0], -1, -1, -1), g,
                              mode="bilinear", align_corners=False)                   # (b,3,TILE,TILE)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            fe = enc(normalize_tiles(tiles))
        outs.append(fe.float().cpu())
    return torch.cat(outs)


@torch.no_grad()
def warp_gt_gpu(m: np.ndarray, mode: str = "nearest") -> torch.Tensor:
    """(EH,EW,C) ERP GT map -> (T,HEAD_OUT,HEAD_OUT,C) warped to every tile via GPU grid_sample.
    Replaces the per-tile CPU warp_to_grid for training GT. NEAREST by default (matches warp_to_grid;
    keeps labels/masks/depth-edges crisp). GRIDS_GT samples the same ray centers warp_to_grid does."""
    t = torch.from_numpy(np.ascontiguousarray(m)).float().permute(2, 0, 1)[None].to(DEVICE)  # (1,C,EH,EW)
    outs = []
    for s in range(0, GRIDS_GT.shape[0], CHUNK):
        g = GRIDS_GT[s:s + CHUNK]
        w = F.grid_sample(t.expand(g.shape[0], -1, -1, -1), g, mode=mode, align_corners=False)
        outs.append(w.permute(0, 2, 3, 1).cpu())
    return torch.cat(outs)                                                            # (T,HEAD_OUT,HEAD_OUT,C)


def coord_map(yaw: float, pitch: float) -> torch.Tensor:
    """Each (TILE_OUT,TILE_OUT) tile pixel -> its STITCH-grid (SH,SW) cell id. Image-independent.

    (u,v) are expressed on a reference 1024x512 ERP; the stitch grid is coarse (SH,SW) so
    tiles fully cover it (coverage-complete), and the assembled field is upsampled later."""
    uy = np.broadcast_to(np.arange(1024, dtype=np.float32)[None], (512, 1024))
    vy = np.broadcast_to(np.arange(512, dtype=np.float32)[:, None], (512, 1024))
    um = e2p(uy[:, :, None], HFOV, yaw, pitch, out_hw=(TILE_OUT, TILE_OUT), mode="nearest")[:, :, 0]
    vm = e2p(vy[:, :, None], HFOV, yaw, pitch, out_hw=(TILE_OUT, TILE_OUT), mode="nearest")[:, :, 0]
    uf = np.clip((um / 1024 * SW).astype(int), 0, SW - 1)
    vf = np.clip((vm / 512 * SH).astype(int), 0, SH - 1)
    return torch.from_numpy((vf * SW + uf).reshape(-1))


@torch.no_grad()
def stitch_field(head: nn.Module, feat: torch.Tensor, cids: List[torch.Tensor],
                 out_ch: int) -> Tuple[torch.Tensor, float, np.ndarray]:
    """Coverage-weighted MEAN stitch of per-tile head fields -> (out_ch,EH,EW) field,
    coverage fraction, and an (EH,EW) bool covered-mask (cells no tile reached -> False).

    Every pano tile shares one optical center, so the same ERP direction has one radial depth /
    one world-frame normal across tiles -> averaging overlaps is unbiased (and denoises). Stitch
    at the coverage-complete (SH,SW) grid, then bilinear-upsample the field to (EH,EW). Uncovered
    cells hold a zero vector -> the caller must drop them via the returned mask, not score them.
    The covered mask uses the SAME bilinear geometry as the field and a >0.999 threshold, so it
    drops not just uncovered cells but the boundary ring whose bilinear stencil touches one (those
    pixels' field values are pulled toward the uncovered zero -> a directional bias on the metric
    depth board if scored). Interior pixels have coverage exactly 1.0 and are kept.
    Caller interprets the field per task (depth: exp of channel 0; normal: renormalize)."""
    acc = torch.zeros(SH * SW, out_ch)
    cnt = torch.zeros(SH * SW, 1)
    for ti in range(feat.shape[0]):
        o = head(feat[ti:ti + 1].float().to(DEVICE))                      # (1,C,head_px,head_px)
        o = F.interpolate(o, (TILE_OUT, TILE_OUT), mode="bilinear", align_corners=False)[0]
        flat = o.permute(1, 2, 0).reshape(-1, out_ch).cpu()
        acc.index_add_(0, cids[ti], flat)
        cnt.index_add_(0, cids[ti], torch.ones(flat.shape[0], 1))
    field = (acc / cnt.clamp_min(1.0)).reshape(1, SH, SW, out_ch).permute(0, 3, 1, 2)
    field = F.interpolate(field, (EH, EW), mode="bilinear", align_corners=False)
    covered = F.interpolate((cnt.squeeze(1) > 0).float().reshape(1, 1, SH, SW), (EH, EW),
                            mode="bilinear", align_corners=False)[0, 0].numpy() > 0.999
    return field[0], float((cnt.squeeze(1) > 0).float().mean()), covered
