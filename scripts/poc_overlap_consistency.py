"""POC: does a FROZEN planar SSL encoder (DINOv2) already produce geometrically
consistent dense features in the OVERLAP of two adjacent AnyRes-E2P tiles?

Tests three things on SUN360 ERP panoramas:
  1. premise        : corresponded-patch cosine  >> random-patch cosine ?
  2. warp necessity : corresponded-patch cosine  >> naive same-index cosine ?
                      (the survey's forbidden ||F_a(u,v)-F_b(u,v)|| is "same index")
  3. pose-free hope : top-1 retrieval acc of the true match among all B patches
                      (high => correspondence is self-recoverable without metadata)

Correspondence is established WITHOUT deriving a homography: we render, alongside
each tile, a "coordinate map" by e2p-sampling an ERP whose pixels encode their own
(x,y) index (nearest interp). Two tile patches correspond iff they sampled the same
ERP location. This is convention-free and exact.
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
PATCH = 14
TILE = 518            # 37x37 patches, DINOv2 native
GRID = TILE // PATCH  # 37
HFOV = 90.0
YAW_A, YAW_B = 0.0, 60.0   # 30deg azimuth overlap
PITCH = 0.0
N_ERP = 12
MATCH_THRESH_PX = 6.0      # erp-pixel tolerance for a patch correspondence
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def render_tile_and_coords(erp_np: np.ndarray, yaw: float):
    """Return (tile_uint8 HxWx3, patch_coords GRIDxGRIDx2 = mean ERP (x,y) per patch)."""
    tile = a2p.erp_to_pinhole_tile(erp_np, yaw_deg=yaw, pitch_deg=PITCH, hfov_deg=HFOV, out_size=TILE)
    h, w = erp_np.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    coord_erp = np.stack([xx, yy, np.zeros_like(xx)], axis=-1).astype(np.float32)
    cmap = py360convert.e2p(coord_erp, HFOV, yaw, PITCH, out_hw=(TILE, TILE), mode="nearest")[..., :2]
    # mean ERP coordinate over each 14x14 patch block -> (GRID, GRID, 2)
    pc = cmap.reshape(GRID, PATCH, GRID, PATCH, 2).mean(axis=(1, 3))
    return np.asarray(tile), pc


@torch.no_grad()
def dense_feats(model: torch.nn.Module, tile_uint8: np.ndarray) -> torch.Tensor:
    """Frozen DINOv2 patch features -> (GRID*GRID, D), L2-normalized."""
    x = torch.from_numpy(tile_uint8).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = x.to(DEVICE)
    out = model(pixel_values=x).last_hidden_state[0, 1:, :]  # drop CLS -> (GRID*GRID, D)
    return F.normalize(out, dim=-1)


def correspond(pc_a: np.ndarray, pc_b: np.ndarray):
    """For each A patch, nearest B patch in ERP-coordinate space + validity mask."""
    a = pc_a.reshape(-1, 2)
    b = pc_b.reshape(-1, 2)
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)  # (Na, Nb)
    nn = d.argmin(axis=1)
    valid = d[np.arange(len(a)), nn] < MATCH_THRESH_PX
    return nn, valid


def main():
    torch.manual_seed(0)
    model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
    files = sorted(glob.glob(ERP_GLOB))[:N_ERP]
    print(f"device={DEVICE} model={MODEL_ID} tiles={GRID}x{GRID} erps={len(files)}")

    corr, naive, rand, retr, n_overlap = [], [], [], [], []
    for f in files:
        erp = np.array(Image.open(f).convert("RGB"))
        ta, pca = render_tile_and_coords(erp, YAW_A)
        tb, pcb = render_tile_and_coords(erp, YAW_B)
        fa = dense_feats(model, ta)            # (N, D)
        fb = dense_feats(model, tb)
        nn, valid = correspond(pca, pcb)
        if valid.sum() == 0:
            continue
        idx = np.where(valid)[0]
        match = nn[idx]
        fa_v = fa[idx]
        fb_m = fb[match]
        sim_all = fa_v @ fb.T                   # (n_valid, N) cosine (normalized)
        corr.append((fa_v * fb_m).sum(-1).mean().item())
        naive.append((fa_v * fb[idx]).sum(-1).mean().item())  # same index = wrong geometry
        perm = torch.randperm(fb.shape[0])[: len(idx)].to(DEVICE)
        rand.append((fa_v * fb[perm]).sum(-1).mean().item())
        retr.append((sim_all.argmax(1).cpu().numpy() == match).mean().item())
        n_overlap.append(int(valid.sum()))

    def stat(x):
        a = np.array(x)
        return f"{a.mean():.3f} +/- {a.std():.3f}"

    print(f"\noverlap patches/img : {np.mean(n_overlap):.0f} / {GRID*GRID}")
    print(f"[1] corresponded cos : {stat(corr)}   <- warp-correct match")
    print(f"    naive same-idx   : {stat(naive)}   <- survey's forbidden ||F(u,v)-F(u,v)||")
    print(f"    random cos       : {stat(rand)}   <- chance floor")
    print(f"[3] top-1 retrieval  : {stat(retr)}   <- self-recoverable w/o pose?")
    print(f"\nlift (corr - random) = {np.mean(corr)-np.mean(rand):.3f} ; "
          f"warp gain (corr - naive) = {np.mean(corr)-np.mean(naive):.3f}")


if __name__ == "__main__":
    main()
