"""DINOv3 frozen-feature segmentation probe: ERP-direct vs AnyRes-E2P pinhole.

Tests the project thesis: a planar encoder (DINOv3) should segment ERP panoramas
better through low-distortion perspective tiles than from the raw distorted ERP.
Labels are warped with the SAME E2P geometry but NEAREST interpolation, so RGB
tiles and label tiles stay pixel-aligned. Linear-probe mIoU; each regime gets its
own head.

Datasets:
  structured3d : single-channel NYU-40 semantic (41 cls incl void 0)
  stanford2d3d : RGB-encoded -> 13 cls via assets/semantic_labels.json (14 incl void)

Train/val is GROUP-disjoint (s3d: by scene; s2d3d: by area, val = area 5 = the
canonical fold) to avoid same-building leakage. With MATCH_PATCHES, the larger
regime (E2P, ~7.5x more patches) is randomly subsampled to the ERP patch count so
the linear probe sees equal training data (controls the data-volume confound).

Run: CUDA_VISIBLE_DEVICES=1 conda run -n pano python scripts/probe_seg_dinov3.py \
       [structured3d|stanford2d3d] [n_groups]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from py360convert import e2p

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import anyres_e2p as a2p  # noqa: E402
import data  # noqa: E402
from encoder import PanoEncoder, normalize_tiles  # noqa: E402

MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DATASET = "structured3d"         # set by configure()
WORK_HW = (512, 1024)            # common working ERP (H,W) for both regimes/datasets
HFOV, OVERLAP, PMAX, TILE = 90.0, 0.2, 45.0, 512
MATCH_PATCHES = True
IGNORE = 0
DEVICE = os.environ.get("DEV", "cuda" if torch.cuda.is_available() else "cpu")

S2D3D_CLASSES = ["beam", "board", "bookcase", "ceiling", "chair", "clutter",
                 "column", "door", "floor", "sofa", "table", "wall", "window"]


def build_s2d3d_lut():
    labels = json.load(open(f"{data.ROOT}/stanford2d3d/semantic_labels.json"))
    name2id = {c: i + 1 for i, c in enumerate(S2D3D_CLASSES)}   # void/UNK -> 0
    lut = np.zeros(len(labels), dtype=np.int64)
    for i, s in enumerate(labels):
        lut[i] = name2id.get(s.split("_")[0], 0)
    return lut


S2D3D_LUT = None
N_CLASS, ROOMS, N_GROUPS = 41, 1, 40          # set by configure()


def configure(dataset, n_groups=None):
    """Set dataset-dependent globals; call before using the pipeline."""
    global DATASET, S2D3D_LUT, N_CLASS, ROOMS, N_GROUPS
    DATASET = dataset
    if dataset == "stanford2d3d":
        S2D3D_LUT = build_s2d3d_lut()
        N_CLASS, def_groups, ROOMS = len(S2D3D_CLASSES) + 1, 7, 8
    elif dataset == "densepass":
        N_CLASS, def_groups, ROOMS = 20, 100, 1        # Cityscapes 19 + void(0)
    else:
        N_CLASS, def_groups, ROOMS = 41, 40, 1
    N_GROUPS = n_groups or def_groups


def dense(enc, x):
    """(B,3,H,W) normalized -> (B,D,gh,gw); pos-embed interpolated for any HxW."""
    b, _, h, w = x.shape
    gh, gw = h // enc.patch, w // enc.patch
    try:
        out = enc.backbone(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state
    except TypeError:
        out = enc.backbone(pixel_values=x).last_hidden_state
    patches = out[:, out.shape[1] - gh * gw:, :]
    return patches.transpose(1, 2).reshape(b, enc.dim, gh, gw)


def label_to_grid(lbl, gh, gw):
    return np.array(Image.fromarray(lbl.astype(np.uint8)).resize((gw, gh), Image.NEAREST)).astype(np.int64)


def e2p_label(lbl, yaw, pitch, hfov, size):
    """Warp single-channel label to a pinhole tile with NEAREST interp (same
    geometry as anyres_e2p.erp_to_pinhole_tile -> aligns with the RGB tile)."""
    out = e2p(lbl[:, :, None].astype(np.float32), hfov, yaw, pitch,
              out_hw=(size, size), mode="nearest")
    return np.round(out[:, :, 0]).astype(np.int64)


def load_rgb_label(f):
    """-> rgb (Hw,Ww,3 uint8) and integer class map (Hw,Ww), both at WORK_HW."""
    h, w = WORK_HW
    if DATASET == "densepass":
        # 400x2048 vertical-FoV strip -> pad into a 2:1 ERP centered at the equator,
        # so e2p geometry is well-defined (pole/pad regions become void=0, ignored).
        sh = round(400 * w / 2048)
        img = np.array(Image.open(f).convert("RGB").resize((w, sh), Image.BILINEAR))
        lbl = np.array(Image.open(data.densepass_gt_path(f)))
        lbl = np.where(lbl < 19, lbl + 1, 0).astype(np.uint8)          # 0..18->1..19, 255->0
        lbl = np.array(Image.fromarray(lbl).resize((w, sh), Image.NEAREST))
        top = (h - sh) // 2
        rgb = np.zeros((h, w, 3), np.uint8); rgb[top:top + sh] = img
        lab = np.zeros((h, w), np.int64); lab[top:top + sh] = lbl
        return rgb, lab
    rgb = np.array(Image.open(f).convert("RGB").resize((w, h), Image.BILINEAR))
    if DATASET == "structured3d":
        lab = np.array(Image.open(data.s3d_gt_path(f, "semantic")))
    else:
        sem = np.array(Image.open(data.s2d3d_gt_path(f, "semantic"))).astype(np.int64)
        idx = sem[:, :, 0] * 65536 + sem[:, :, 1] * 256 + sem[:, :, 2]
        lab = np.where(idx < len(S2D3D_LUT), S2D3D_LUT[np.clip(idx, 0, len(S2D3D_LUT) - 1)], 0)
    lab = np.array(Image.fromarray(lab.astype(np.uint8)).resize((w, h), Image.NEAREST)).astype(np.int64)
    return rgb, lab


@torch.no_grad()
def feats_erp(enc, rgb, lab):
    x = torch.from_numpy(rgb).float().permute(2, 0, 1)[None] / 255.0
    feat = dense(enc, normalize_tiles(x.to(DEVICE)))[0]
    d, gh, gw = feat.shape
    g = label_to_grid(lab, gh, gw)
    return feat.reshape(d, -1).t().cpu(), torch.from_numpy(g.reshape(-1))


@torch.no_grad()
def feats_e2p(enc, rgb, lab):
    if DATASET == "densepass":                       # equator ring only (content is a band)
        n = a2p._ring_yaw_count(HFOV, OVERLAP, 0.0, 90.0)
        plan = [a2p.TilePlan(y, 0.0) for y in a2p._ring_yaws(n, 0.0)]
    else:
        plan = a2p.plan_tiles("band", HFOV, HFOV, OVERLAP, pmax_deg=PMAX)
    fs, ls = [], []
    for tp in plan:
        tile = np.asarray(a2p.erp_to_pinhole_tile(rgb, tp.yaw_deg, tp.pitch_deg, HFOV, TILE))
        x = torch.from_numpy(tile).float().permute(2, 0, 1)[None] / 255.0
        feat = dense(enc, normalize_tiles(x.to(DEVICE)))[0]
        d, gh, gw = feat.shape
        gl = label_to_grid(e2p_label(lab, tp.yaw_deg, tp.pitch_deg, HFOV, TILE), gh, gw)
        fs.append(feat.reshape(d, -1).t().cpu())
        ls.append(torch.from_numpy(gl.reshape(-1)))
    return torch.cat(fs), torch.cat(ls), len(plan)


def miou_acc(pred, gt):
    m = gt != IGNORE
    pred, gt = pred[m], gt[m]
    acc = (pred == gt).float().mean().item()
    ious = []
    for c in range(1, N_CLASS):
        g = gt == c
        if g.sum() == 0:
            continue
        p = pred == c
        u = (p | g).sum().item()
        ious.append((p & g).sum().item() / u if u else 0.0)
    return (float(np.mean(ious)) if ious else 0.0), acc, len(ious)


def linear_probe(Xtr, ytr, Xva, yva, steps=800):
    Xtr, ytr = Xtr.to(DEVICE).float(), ytr.to(DEVICE)
    Xva, yva = Xva.to(DEVICE).float(), yva.to(DEVICE)
    clf = torch.nn.Linear(Xtr.shape[1], N_CLASS).to(DEVICE)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = torch.nn.CrossEntropyLoss(ignore_index=IGNORE)
    for _ in range(steps):
        opt.zero_grad()
        lossf(clf(Xtr), ytr).backward()
        opt.step()
    with torch.no_grad():
        pred = clf(Xva).argmax(1)
    return miou_acc(pred.cpu(), yva.cpu())


def subsample(feat, lab, n, seed=0):
    if feat.shape[0] <= n:
        return feat, lab
    idx = torch.randperm(feat.shape[0], generator=torch.Generator().manual_seed(seed))[:n]
    return feat[idx], lab[idx]


def grouped():
    if DATASET == "densepass":                       # 100 distinct locations -> split by image
        allf = data.list_densepass()
        panos = [(str(i), f) for i, f in enumerate(allf)]
        groups = [g for g, _ in panos]
        return panos, groups, set(groups[:int(len(groups) * 0.7)])
    if DATASET == "structured3d":
        allf = data.list_structured3d()
        def key(f): return f.split("Structured3D/scene_")[1][:5]
    else:
        allf = data.list_erps("stanford2d3d")
        def key(f): return f.split("extracted_data/")[1].split("/")[0]
    by = {}
    for f in allf:
        by.setdefault(key(f), []).append(f)
    groups = sorted(by)[:N_GROUPS]
    panos = [(g, f) for g in groups for f in by[g][:ROOMS]]
    if DATASET == "structured3d":
        train = set(groups[:int(len(groups) * 0.7)])
    else:
        train = {g for g in groups if "5" not in g}          # val = area_5* (canonical)
    return panos, groups, train


def main():
    configure(sys.argv[1] if len(sys.argv) > 1 else "structured3d",
              int(sys.argv[2]) if len(sys.argv) > 2 else None)
    panos, groups, train = grouped()
    print(f"device={DEVICE} model={MODEL} dataset={DATASET} N_CLASS={N_CLASS}")
    print(f"groups={len(groups)} (train={len(train)}/val={len(groups) - len(train)}, group-disjoint) "
          f"panos={len(panos)} rooms/group={ROOMS} match_patches={MATCH_PATCHES} "
          f"hfov={HFOV} overlap={OVERLAP} tile={TILE} work_erp={WORK_HW}")
    enc = PanoEncoder(model_id=MODEL, lora_rank=0).to(DEVICE).eval()

    bins = {r: {"tr": ([], []), "va": ([], [])} for r in ("erp", "e2p")}
    n_tiles = 0
    for i, (g, f) in enumerate(panos):
        rgb, lab = load_rgb_label(f)
        ef, el = feats_erp(enc, rgb, lab)
        pf, pl, n_tiles = feats_e2p(enc, rgb, lab)
        sp = "tr" if g in train else "va"
        bins["erp"][sp][0].append(ef); bins["erp"][sp][1].append(el)
        bins["e2p"][sp][0].append(pf); bins["e2p"][sp][1].append(pl)
        if (i + 1) % 10 == 0 or i == len(panos) - 1:
            print(f"  [{i+1}/{len(panos)}] last={g} [{sp}] e2p_tiles={n_tiles}")

    cat = torch.cat
    out = {r: {k: (cat(bins[r][k][0]), cat(bins[r][k][1])) for k in ("tr", "va")}
           for r in ("erp", "e2p")}
    note = ""
    if MATCH_PATCHES:
        for k in ("tr", "va"):
            out["e2p"][k] = subsample(*out["e2p"][k], out["erp"][k][0].shape[0])
        note = " [patch-matched]"
    print(f"\nE2P band tiles/pano={n_tiles}  train patches: "
          f"ERP={out['erp']['tr'][0].shape[0]} E2P={out['e2p']['tr'][0].shape[0]} | "
          f"val: ERP={out['erp']['va'][0].shape[0]} E2P={out['e2p']['va'][0].shape[0]}")

    res = {r: linear_probe(*out[r]["tr"], *out[r]["va"]) for r in ("erp", "e2p")}
    print(f"\n{'regime':18s} {'mIoU':>7} {'pixAcc':>7} {'#cls':>5}")
    print(f"{'ERP-direct':18s} {res['erp'][0]:7.3f} {res['erp'][1]:7.3f} {res['erp'][2]:5d}")
    print(f"{'E2P-pinhole(avg)':18s} {res['e2p'][0]:7.3f} {res['e2p'][1]:7.3f} {res['e2p'][2]:5d}")
    d = res["e2p"][0] - res["erp"][0]
    print(f"\nE2P - ERP mIoU delta = {d:+.3f}{note}  ({'E2P better' if d > 0 else 'ERP better'})")


if __name__ == "__main__":
    main()
