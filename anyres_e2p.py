"""ERP to pinhole tile generation for AnyRes-style panorama preprocessing.

This is a standalone copy of the core AnyRes-E2P renderer from PanoLLaVA/CORA.
It generates one global ERP context image plus a grid of perspective tiles with
metadata needed for view fusion or inverse projection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

try:
    from py360convert import e2p as erp_to_persp

    _HAS_PY360 = True
except Exception:
    _HAS_PY360 = False
    erp_to_persp = None  # type: ignore[assignment]


def deg2rad(degrees: float) -> float:
    return degrees * math.pi / 180.0


def yaw_pitch_to_xyz(yaw_deg: float, pitch_deg: float) -> Tuple[float, float, float]:
    """Convert yaw and pitch in degrees to a unit vector."""
    yaw_rad = deg2rad(yaw_deg)
    pitch_rad = deg2rad(pitch_deg)
    cos_pitch = math.cos(pitch_rad)
    return (
        cos_pitch * math.cos(yaw_rad),
        math.sin(pitch_rad),
        cos_pitch * math.sin(yaw_rad),
    )


def letterbox_square(img: Image.Image, size: int, fill: Tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """Resize the longest edge to size and pad the shorter edge to a square."""
    width, height = img.size
    scale = size / max(width, height)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    resized = img.resize((new_width, new_height), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(resized, ((size - new_width) // 2, (size - new_height) // 2))
    return canvas


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def resize_to_vit(tensor: torch.Tensor, vit_size: Optional[int]) -> torch.Tensor:
    if vit_size is None:
        return tensor
    return torch.nn.functional.interpolate(
        tensor.unsqueeze(0),
        size=(vit_size, vit_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def norm_angle_180(angle: float) -> float:
    """Normalize an angle to [-180, 180)."""
    return ((angle + 180.0) % 360.0) - 180.0


def make_yaw_centers_standard(
    hfov_deg: float,
    overlap: float,
    yaw_start: float = -180.0,
    yaw_end: float = 180.0,
    phase_deg: float = 0.0,
) -> List[float]:
    """Build open-interval yaw centers."""
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1).")
    step = hfov_deg * (1.0 - overlap)
    if step <= 0.0:
        raise ValueError("Invalid overlap leading to non-positive step.")

    centers: List[float] = []
    cur = yaw_start + hfov_deg / 2.0 + phase_deg
    while cur < yaw_end - hfov_deg / 2.0 + 1e-9:
        centers.append(norm_angle_180(cur))
        cur += step
    return centers


def make_yaw_centers_closed_loop(
    hfov_deg: float,
    overlap: float,
    start_deg: float = -180.0,
    seam_phase_deg: float = 0.0,
) -> List[float]:
    """Build yaw centers with uniform spacing that exactly wraps 360 degrees."""
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1).")
    step_raw = hfov_deg * (1.0 - overlap)
    if step_raw <= 0.0:
        raise ValueError("Invalid overlap leading to non-positive step.")

    n_tiles = max(1, math.ceil(360.0 / step_raw))
    step = 360.0 / n_tiles

    centers: List[float] = []
    cur = start_deg + seam_phase_deg
    for _ in range(n_tiles):
        centers.append(norm_angle_180(cur))
        cur += step

    unique: List[float] = []
    for center in centers:
        if all(abs(norm_angle_180(center - existing)) > 1e-6 for existing in unique):
            unique.append(center)
    return sorted(unique, key=lambda x: ((x + 360.0) % 360.0))


def resolve_yaw_geometry(
    hfov_deg: float,
    render_overlap: float,
    closed_loop: bool = True,
) -> Tuple[int, float]:
    """Return the realized yaw tile count and physical adjacent overlap."""
    if hfov_deg <= 0.0:
        return 1, 0.0

    render_overlap = float(max(0.0, min(render_overlap, 0.999999)))
    step_raw = hfov_deg * (1.0 - render_overlap)
    if step_raw <= 0.0:
        return 1, 0.0

    if closed_loop:
        n_tiles = max(1, math.ceil(360.0 / step_raw))
        step = 360.0 / n_tiles
    else:
        n_tiles = max(1, int(math.floor(360.0 / step_raw)))
        step = step_raw

    phys_overlap = max(0.0, 1.0 - step / hfov_deg)
    return n_tiles, float(phys_overlap)


def maybe_add_seam_center(centers: List[float], seam_center: float = -180.0) -> List[float]:
    """Force-add a center at the ERP seam if not already present."""

    def angle_diff(a: float, b: float) -> float:
        return abs(norm_angle_180(a - b))

    if all(angle_diff(center, seam_center) > 1e-6 for center in centers):
        return [norm_angle_180(seam_center)] + centers
    return centers


def make_pitch_centers(
    vfov_deg: float,
    overlap: float,
    pitch_min: float,
    pitch_max: float,
) -> List[float]:
    """Build sliding pitch centers within a vertical range."""
    if abs(pitch_max - pitch_min) < 1e-6:
        return [pitch_min]
    if pitch_min >= pitch_max:
        raise ValueError("pitch_min must be smaller than pitch_max.")

    step = vfov_deg * (1.0 - overlap)
    if step <= 0.0:
        raise ValueError("Invalid overlap leading to non-positive step.")

    centers: List[float] = []
    cur = pitch_min + vfov_deg / 2.0
    while cur <= pitch_max - vfov_deg / 2.0 + 1e-9:
        centers.append(cur)
        cur += step
    return centers


@dataclass(frozen=True)
class TileMeta:
    tile_id: int
    yaw_deg: float
    pitch_deg: float
    hfov_deg: float
    vfov_deg: float
    center_xyz: Tuple[float, float, float]
    is_cap: bool = False


@dataclass(frozen=True)
class AnyResPack:
    global_image: torch.Tensor
    tiles: torch.Tensor
    metas: List[TileMeta]
    global_meta: Dict[str, object]
    yaw_geometry: Dict[str, object] = field(default_factory=dict)


def compute_vfov_from_hfov(hfov_deg: float, out_size: int) -> float:
    """Return vertical FOV for square pinhole output."""
    _ = out_size
    return hfov_deg


def erp_to_pinhole_tile(
    erp: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    hfov_deg: float,
    out_size: int,
    vfov_deg: Optional[float] = None,
) -> Image.Image:
    """Render one pinhole tile from an equirectangular panorama array.

    vfov_deg=None -> square FOV (vfov=hfov); otherwise an independent vertical FOV.
    """
    if not _HAS_PY360:
        raise ImportError("py360convert required: `pip install py360convert opencv-python`")

    if erp.dtype != np.uint8:
        erp_u8 = (np.clip(erp, 0, 1) * 255).astype(np.uint8) if erp.max() <= 1.0 else erp.astype(np.uint8)
    else:
        erp_u8 = erp

    fov = hfov_deg if vfov_deg is None else (hfov_deg, vfov_deg)
    bgr = erp_u8[:, :, ::-1]
    persp = erp_to_persp(bgr, fov, yaw_deg, pitch_deg, out_hw=(out_size, out_size))
    rgb = persp[:, :, ::-1]
    return Image.fromarray(rgb)


@dataclass(frozen=True)
class TilePlan:
    yaw_deg: float
    pitch_deg: float
    is_cap: bool = False


def _ring_yaw_count(hfov_deg: float, overlap: float, pitch_deg: float, pitch_step: float) -> int:
    """Tiles to wrap a latitude ring with cos(phi) correction (sized at the ring's
    equator-side edge so the whole band is covered)."""
    phi_w = max(0.0, abs(pitch_deg) - pitch_step / 2.0)
    return max(1, math.ceil(360.0 * math.cos(math.radians(phi_w)) / (hfov_deg * (1.0 - overlap))))


def _ring_yaws(n_tiles: int, yaw_phase_deg: float) -> List[float]:
    """Uniform 360/N closed-loop yaw centers (tile centered on the +/-180 seam at phase 0),
    so the last tile overlaps the first across the seam exactly like any interior pair."""
    return [norm_angle_180(-180.0 + j * 360.0 / n_tiles + yaw_phase_deg) for j in range(n_tiles)]


def plan_tiles(mode: str, hfov_deg: float, vfov_deg: float, overlap: float,
               pmax_deg: float = 60.0, yaw_phase_deg: float = 0.0) -> List[TilePlan]:
    """Explicit-mode tile schedule with cos(phi) yaw correction and seamless wrap.

    mode='full_sphere': pitch interval [-90,90] with +/-90 pole caps (N=1 each).
    mode='band'       : equator-centered symmetric rings within [-pmax,pmax], no caps.
    """
    dphi = vfov_deg * (1.0 - overlap)
    if dphi <= 0.0:
        raise ValueError("overlap too large (non-positive pitch step).")

    if mode == "full_sphere":
        n = max(1, math.ceil(180.0 / dphi)); step = 180.0 / n
        centers = [-90.0 + i * step for i in range(n + 1)]
    elif mode == "band":
        m = max(1, math.ceil(pmax_deg / dphi)); step = pmax_deg / m
        centers = [i * step for i in range(-m, m + 1)]
    else:
        raise ValueError(f"unknown mode {mode!r} (use 'full_sphere' or 'band').")

    tiles: List[TilePlan] = []
    for phi in centers:
        if mode == "full_sphere" and abs(abs(phi) - 90.0) < 1e-6:
            tiles.append(TilePlan(0.0, phi, is_cap=True))
            continue
        n_yaw = _ring_yaw_count(hfov_deg, overlap, phi, step)
        tiles += [TilePlan(y, phi) for y in _ring_yaws(n_yaw, yaw_phase_deg)]
    return tiles


def build_anyres_from_erp(
    erp_img: Image.Image,
    base_size: int = 336,
    tile_render_size: int = 672,
    vit_size: Optional[int] = None,
    hfov_deg: float = 90.0,
    overlap: float = 0.2,
    mode: Optional[str] = None,          # None = legacy band; "full_sphere" | "band" = cos-phi schedule
    vfov_deg: Optional[float] = None,    # new modes: independent vertical FOV (default = hfov, square)
    pmax_deg: float = 60.0,              # band mode: max |pitch| of ring centers
    closed_loop_yaw: bool = True,  # SSL needs seam overlap: wrap 360 so +/-180 tiles overlap
    yaw_phase_deg: float = 0.0,
    include_seam_center: bool = False,
    pitch_min: float = -45.0,
    pitch_max: float = 45.0,
    pitch_full_span: bool = False,
    cap_eps: float = 0.5,
) -> AnyResPack:
    """Generate global ERP context plus perspective tiles from an ERP panorama."""
    global_img = letterbox_square(erp_img.convert("RGB"), base_size)
    global_tensor = resize_to_vit(pil_to_tensor(global_img), vit_size)

    if pitch_full_span:
        pitch_min = max(pitch_min, -90.0 + cap_eps)
        pitch_max = min(pitch_max, 90.0 - cap_eps)

    erp_np = np.array(erp_img.convert("RGB"))

    if mode is not None:                                   # explicit improved schedule (cos-phi)
        vfov = float(vfov_deg) if vfov_deg is not None else float(hfov_deg)
        plan = plan_tiles(mode, hfov_deg, vfov, overlap, pmax_deg=pmax_deg, yaw_phase_deg=yaw_phase_deg)
        tile_tensors: List[torch.Tensor] = []
        metas: List[TileMeta] = []
        for tid, tp in enumerate(plan):
            tile = erp_to_pinhole_tile(erp_np, tp.yaw_deg, tp.pitch_deg, hfov_deg,
                                       tile_render_size, vfov_deg=vfov)
            tile_tensors.append(resize_to_vit(pil_to_tensor(tile), vit_size))
            metas.append(TileMeta(tid, tp.yaw_deg, tp.pitch_deg, float(hfov_deg), vfov,
                                  yaw_pitch_to_xyz(tp.yaw_deg, tp.pitch_deg), tp.is_cap))
        if tile_tensors:
            tiles = torch.stack(tile_tensors, dim=0)
        else:
            s = vit_size or tile_render_size
            tiles = torch.empty(0, 3, s, s)
        rings = sorted({round(tp.pitch_deg, 3) for tp in plan})
        yaw_geometry = {
            "mode": mode, "hfov_deg": float(hfov_deg), "vfov_deg": vfov,
            "overlap": float(overlap), "pmax_deg": (float(pmax_deg) if mode == "band" else None),
            "n_tiles": len(plan), "n_caps": int(sum(tp.is_cap for tp in plan)),
            "ring_pitches_deg": rings,
            "ring_tile_counts": [int(sum(1 for tp in plan if abs(tp.pitch_deg - p) < 1e-6)) for p in rings],
            "closed_loop_yaw": True,
        }
        return AnyResPack(
            global_image=global_tensor, tiles=tiles, metas=metas,
            global_meta={"kind": "global_letterbox", "base_size": base_size, "vit_size": vit_size},
            yaw_geometry=yaw_geometry,
        )

    vfov_deg = compute_vfov_from_hfov(hfov_deg, tile_render_size)

    if closed_loop_yaw:
        yaws = make_yaw_centers_closed_loop(hfov_deg, overlap, start_deg=-180.0, seam_phase_deg=yaw_phase_deg)
    else:
        yaws = make_yaw_centers_standard(
            hfov_deg,
            overlap,
            yaw_start=-180.0,
            yaw_end=180.0,
            phase_deg=yaw_phase_deg,
        )

    if include_seam_center:
        yaws = maybe_add_seam_center(yaws, seam_center=-180.0)

    pitches = make_pitch_centers(vfov_deg, overlap, pitch_min=pitch_min, pitch_max=pitch_max)

    tile_tensors: List[torch.Tensor] = []
    metas: List[TileMeta] = []
    tile_id = 0
    for pitch in pitches:
        for yaw in yaws:
            tile = erp_to_pinhole_tile(
                erp_np,
                yaw_deg=yaw,
                pitch_deg=pitch,
                hfov_deg=hfov_deg,
                out_size=tile_render_size,
            )
            tile_tensor = resize_to_vit(pil_to_tensor(tile), vit_size)
            tile_tensors.append(tile_tensor)
            metas.append(
                TileMeta(
                    tile_id=tile_id,
                    yaw_deg=yaw,
                    pitch_deg=pitch,
                    hfov_deg=hfov_deg,
                    vfov_deg=vfov_deg,
                    center_xyz=yaw_pitch_to_xyz(yaw, pitch),
                )
            )
            tile_id += 1

    if tile_tensors:
        tiles = torch.stack(tile_tensors, dim=0)
    else:
        size = vit_size or tile_render_size
        tiles = torch.empty(0, 3, size, size)

    _, phys_overlap = resolve_yaw_geometry(hfov_deg, overlap, closed_loop=closed_loop_yaw)
    global_meta: Dict[str, object] = {
        "kind": "global_letterbox",
        "base_size": base_size,
        "vit_size": vit_size,
    }
    yaw_geometry: Dict[str, object] = {
        "hfov_deg": float(hfov_deg),
        "render_overlap": float(overlap),
        "phys_overlap": float(phys_overlap),
        "n_tiles": int(len(yaws)),
        "yaw_centers_deg": [float(yaw) for yaw in yaws],
        "closed_loop_yaw": bool(closed_loop_yaw),
    }

    return AnyResPack(
        global_image=global_tensor,
        tiles=tiles,
        metas=metas,
        global_meta=global_meta,
        yaw_geometry=yaw_geometry,
    )

