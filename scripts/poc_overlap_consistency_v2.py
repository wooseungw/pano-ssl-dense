"""POC v2: disentangle quantization vs real distortion, profile edge falloff,
and find the best layer to distill. Adds to v1:
  (A) SUBPIXEL B-feature sampling (grid_sample) vs patch-mean nearest match
      -> if subpixel >> patch-mean, the v1 gap was quantization, not distortion.
  (B) corresponded-cos vs distance-from-tile-center (the anisotropic edge nuisance).
  (C) layer sweep: which DINOv2 block has the most overlap-consistent features.
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import anyres_e2p as a2p  # noqa: E402
import py360convert  # noqa: E402
from transformers import AutoModel  # noqa: E402

ERP_GLOB = "/data/1_personal/4_SWWOO/SUN360/test/RGB/*.jpg"
MODEL_ID = "facebook/dinov2-base"
PATCH, TILE = 14, 518
GRID = TILE // PATCH
HFOV, YAW_A, YAW_B, PITCH = 90.0, 0.0, 60.0, 0.0
N_ERP = 12
THRESH_PX = 4.0
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def render(erp_np, yaw):
    tile = np.asarray(a2p.erp_to_pinhole_tile(erp_np, yaw, PITCH, HFOV, TILE))
    h, w = erp_np.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    coord = np.stack([xx, yy, np.zeros_like(xx)], -1).astype(np.float32)
    cmap = py360convert.e2p(coord, HFOV, yaw, PITCH, out_hw=(TILE, TILE), mode="nearest")[..., :2]
    return tile, cmap  # cmap: (TILE,TILE,2) erp (x,y) per tile pixel


@torch.no_grad()
def feats(model, tile):
    x = torch.from_numpy(tile.copy()).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    x = ((x - MEAN) / STD).to(DEVICE)
    hs = model(pixel_values=x, output_hidden_states=True).hidden_states  # tuple L+1
    # return per-layer normalized token grids (D, GRID, GRID) for grid_sample
    grids = {}
    for li in (3, 6, 9, 12):
        t = F.normalize(hs[li][0, 1:, :], dim=-1)            # (N, D)
        grids[li] = t.t().reshape(t.shape[1], GRID, GRID)    # (D, GRID, GRID)
    return grids


def main():
    torch.manual_seed(0)
    model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
    files = sorted(glob.glob(ERP_GLOB))[:N_ERP]
    print(f"device={DEVICE} model={MODEL_ID} grid={GRID}x{GRID} erps={len(files)}")

    pm, sp = [], []                      # patch-mean cos, subpixel cos (layer 12)
    layer_cos = {li: [] for li in (3, 6, 9, 12)}
    edge_bins = {0: [], 1: [], 2: []}    # near-center .. near-edge

    # A patch-center pixel coords
    ci = (np.arange(GRID) * PATCH + PATCH // 2)
    cyx = np.stack(np.meshgrid(ci, ci, indexing="ij"), -1).reshape(-1, 2)  # (N,2)=(row,col)

    for f in files:
        erp = np.array(Image.open(f).convert("RGB"))
        ta, ca = render(erp, YAW_A)
        tb, cb = render(erp, YAW_B)
        ga, gb = feats(model, ta), feats(model, tb)

        # A patch-center ERP coords
        a_erp = ca[cyx[:, 0], cyx[:, 1]]                      # (N,2) erp xy
        # B nearest pixel (downsampled stride 2) for each A center
        bs = cb[::2, ::2].reshape(-1, 2)                      # (M,2)
        bpix = np.stack(np.meshgrid(np.arange(0, TILE, 2), np.arange(0, TILE, 2), indexing="ij"), -1)
        bpix = bpix.reshape(-1, 2)                            # (M,2)=(row,col)
        A = torch.from_numpy(a_erp).float().to(DEVICE)
        B = torch.from_numpy(bs).float().to(DEVICE)
        d = torch.cdist(A, B)                                 # (N,M)
        dmin, nn = d.min(1)
        valid = (dmin < THRESH_PX).cpu().numpy()
        idx = np.where(valid)[0]
        if len(idx) == 0:
            continue
        b_rc = bpix[nn.cpu().numpy()[idx]]                   # matched B pixel (row,col)

        # --- subpixel B feature via grid_sample (align_corners=False) ---
        nx = (b_rc[:, 1] + 0.5) / TILE * 2 - 1               # col -> x
        ny = (b_rc[:, 0] + 0.5) / TILE * 2 - 1               # row -> y
        grid = torch.tensor(np.stack([nx, ny], -1), dtype=torch.float32, device=DEVICE).view(1, 1, -1, 2)

        fa12 = ga[12].reshape(ga[12].shape[0], -1).t()[idx]  # (n,D) A patch feats
        fb12_grid = gb[12].unsqueeze(0)                      # (1,D,G,G)
        fb_sp = F.grid_sample(fb12_grid, grid, mode="bilinear", align_corners=False)
        fb_sp = F.normalize(fb_sp[0, :, 0, :].t(), dim=-1)   # (n,D)
        sp.append((fa12 * fb_sp).sum(-1).mean().item())

        # patch-mean nearest (B patch index from matched pixel)
        b_patch = (b_rc // PATCH)
        b_lin = (b_patch[:, 0] * GRID + b_patch[:, 1])
        fb12 = gb[12].reshape(gb[12].shape[0], -1).t()       # (N,D)
        pm.append((fa12 * fb12[b_lin]).sum(-1).mean().item())

        # layer sweep (patch-mean)
        for li in (3, 6, 9, 12):
            fa = ga[li].reshape(ga[li].shape[0], -1).t()[idx]
            fb = gb[li].reshape(gb[li].shape[0], -1).t()
            layer_cos[li].append((fa * fb[b_lin]).sum(-1).mean().item())

        # edge profile: A center column distance from tile center (518/2)
        col = cyx[idx, 1]
        dist = np.abs(col - TILE / 2)
        b0, b1 = np.quantile(dist, [1 / 3, 2 / 3])
        cos_sp = (fa12 * fb_sp).sum(-1).cpu().numpy()
        edge_bins[0].append(cos_sp[dist <= b0].mean())
        edge_bins[1].append(cos_sp[(dist > b0) & (dist <= b1)].mean())
        edge_bins[2].append(cos_sp[dist > b1].mean())

    s = lambda x: f"{np.mean(x):.3f}+/-{np.std(x):.3f}"
    print(f"\n[A] patch-mean cos : {s(pm)}")
    print(f"    SUBPIXEL  cos  : {s(sp)}   (quantization gain = {np.mean(sp)-np.mean(pm):+.3f})")
    print(f"\n[B] edge profile (subpixel cos, near-center -> near-edge):")
    print(f"    center {np.mean(edge_bins[0]):.3f} | mid {np.mean(edge_bins[1]):.3f} | edge {np.mean(edge_bins[2]):.3f}"
          f"   (center-edge drop = {np.mean(edge_bins[0])-np.mean(edge_bins[2]):+.3f})")
    print(f"\n[C] layer sweep (patch-mean cos): " + "  ".join(f"L{li}={np.mean(v):.3f}" for li, v in layer_cos.items()))


if __name__ == "__main__":
    main()
