"""Matterport3D skybox -> equirectangular (ERP) loader.

MP3D `matterport_skybox_images` ships 6 cube faces per panorama viewpoint:
    <house>/matterport_skybox_images/<pano_id>_skybox{0..5}_sami.jpg   (square, 1024x1024)
skybox0 = Up (ceiling), skybox5 = Down (floor), skybox1..4 = the horizontal ring.

We stitch to ERP with py360convert.c2e. py360convert's `list` cube format is ordered
[Front, Right, Back, Left, Up, Down] with x=right, y=up, z=front (side faces upright,
+longitude to the right). The MP3D->py360 mapping below was derived from PanoBasic's
demo_matterport.m + py360convert 1.0.4 `xyzcube`, and VERIFIED on real MP3D pixels
(house 17DRP5sb8fy): the two pole rotations make the zenith/nadir seams continuous
(wrong rotations produce a visible pinwheel at the floor/ceiling).

    Front=skybox2, Right=skybox3, Back=skybox4, Left=skybox1   (no rotation)
    Up   = np.rot90(skybox0, k=3)   (90 deg clockwise)
    Down = np.rot90(skybox5, k=1)   (90 deg counter-clockwise)

RGB only: MP3D depth/normal/semantic are per-view perspective / mesh-based, not
ERP-aligned, so dense GT needs separate rendering (not handled here).
"""
from __future__ import annotations

import glob
import os
from typing import List, Optional, Tuple

import numpy as np
import py360convert
from PIL import Image

from data import ROOT

MP3D_SCANS = os.path.join(ROOT, "matterport3d", "v1", "scans")

# skybox index -> py360convert [F, R, B, L, U, D] slot, and np.rot90 k per slot.
_FACE_ORDER: Tuple[int, ...] = (2, 3, 4, 1, 0, 5)   # F, R, B, L, U, D
_FACE_ROT: Tuple[int, ...] = (0, 0, 0, 0, 3, 1)      # rot90 k applied to each


def list_mp3d(limit: Optional[int] = None, house: Optional[str] = None) -> List[str]:
    """Return skybox0 jpg paths (one per MP3D panorama viewpoint) on disk.

    Robust to both the canonical layout (v1/scans/<house>/matterport_skybox_images/)
    and a double-nested extraction (v1/scans/<house>/<house>/matterport_skybox_images/).
    `house` restricts to one house id. Each returned path identifies a full 6-face set;
    pass it to skybox_to_erp / load_erp.
    """
    pat = "*_skybox0_sami.jpg"
    hs = house or "*"
    fs = glob.glob(os.path.join(MP3D_SCANS, hs, "matterport_skybox_images", pat))
    fs += glob.glob(os.path.join(MP3D_SCANS, hs, "*", "matterport_skybox_images", pat))
    # dedup by (house_id, pano_id) so canonical + double-nested extractions of the
    # same house don't yield the panorama twice.
    seen, out = set(), []
    for f in sorted(fs):
        key = mp3d_ids(f)
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out[:limit] if limit else out


def face_paths(skybox0_path: str) -> List[str]:
    """The 6 face paths (skybox0..5) for the panorama identified by a skybox0 path."""
    return [skybox0_path.replace("_skybox0_sami", f"_skybox{i}_sami") for i in range(6)]


def mp3d_ids(skybox0_path: str) -> Tuple[str, str]:
    """(house_id, pano_id) parsed from a skybox0 path."""
    sbdir = os.path.dirname(skybox0_path)                 # .../<house>/matterport_skybox_images
    house_id = os.path.basename(os.path.dirname(sbdir))
    pano_id = os.path.basename(skybox0_path).replace("_skybox0_sami.jpg", "")
    return house_id, pano_id


def skybox_to_erp(skybox0_path: str, out_h: int = 1024, out_w: int = 2048,
                  mode: str = "bilinear") -> np.ndarray:
    """Stitch a panorama's 6 skybox faces into an ERP RGB array (out_h, out_w, 3) uint8.

    out_w must be a multiple of 8 (py360convert requirement); faces must be square.
    """
    faces = [np.asarray(Image.open(p).convert("RGB")) for p in face_paths(skybox0_path)]
    h, w = faces[0].shape[:2]
    if h != w:
        raise ValueError(f"skybox faces must be square, got {faces[0].shape} for {skybox0_path}")
    cube = [np.rot90(faces[idx], k) for idx, k in zip(_FACE_ORDER, _FACE_ROT)]
    erp = py360convert.c2e(cube, out_h, out_w, mode=mode, cube_format="list")
    return np.clip(erp, 0, 255).astype(np.uint8)


def load_erp(skybox0_path: str, out_h: int = 1024, out_w: int = 2048) -> np.ndarray:
    """Alias matching the (path -> ERP np.uint8) contract used elsewhere in the repo."""
    return skybox_to_erp(skybox0_path, out_h, out_w)
