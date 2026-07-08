"""Qualitative val-set visualization for F-2: mean vs attn fusion, side by side.

Per val pano (docs/figures/viz_fusion_f2/pano_<i>.png), five panels:
  ERP RGB | GT seg | mean-fusion pred | attn-fusion pred | fix/break map
The fix/break map is the paired question made visible — on cells where the two
fusions DISAGREE: green = attn fixed a mean error, red = attn broke a mean-correct
cell, yellow = both wrong (differently). Prints the aggregate fixed/broken counts.

Run (after both paired runs saved): CUDA_VISIBLE_DEVICES=<n> conda run -n pano \
    python scripts/viz_fusion_f2.py [n_panos=6]
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob  # noqa: E402

import probe_seg_dinov3 as P  # noqa: E402
import runlog  # noqa: E402
import train_fusion_f2 as F2  # noqa: E402
from encoder import PanoEncoder  # noqa: E402
from fusion import SetFusion, masked_mean, pack_sets  # noqa: E402

DEVICE = P.DEVICE
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "figures", "viz_fusion_f2")


def load_branch(kind: str, dim: int):
    ck = torch.load(os.path.join(ROOT, "runs", f"ckpt_fusion_f2_{kind}", "fusion_f2.pt"),
                    map_location="cpu")
    head = torch.nn.Linear(dim, P.N_CLASS)
    head.load_state_dict(ck["head"])
    fusion = None
    if ck["fusion"] is not None:
        fusion = SetFusion(dim=dim)
        fusion.load_state_dict(ck["fusion"])
        fusion.to(DEVICE).eval()
    return fusion, head.to(DEVICE).eval()


def palette(n: int) -> np.ndarray:
    rng = np.random.RandomState(0)
    pal = rng.uniform(0.15, 0.95, size=(n, 3))
    pal[0] = 0.0                                          # void/ignore = black
    return pal


@torch.no_grad()
def predict_grid(fusion, head, f, g, m, cov, hf, wf):
    """Covered-cell predictions painted back onto the (hf, wf) grid (-1 = uncovered)."""
    logits = head(fusion(f, g, m) if fusion is not None else masked_mean(f, m))
    grid = np.full(hf * wf, -1, np.int64)
    grid[cov.numpy()] = logits.argmax(1).cpu().numpy()
    return grid.reshape(hf, wf)


def main() -> None:
    n_panos = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    P.configure("structured3d"); P.TILE = 512
    F2.D.DATASET, F2.D.OVERLAP, F2.D.TILE = "structured3d", F2.OVERLAP, 512
    enc = PanoEncoder(model_id=P.MODEL, adapter_path=F2.ADAPTER).to(DEVICE).eval()
    P.enc_patch = enc.patch
    cfg = F2.build_config(65.0)
    fusion_a, head_a = load_branch("attn", enc.dim)
    _, head_m = load_branch("mean", enc.dim)

    allf = P.data.list_structured3d()
    by_scene: dict = {}
    for fpath in allf:
        by_scene.setdefault(fpath.split("scene_")[1][:5], []).append(fpath)
    scenes = sorted(by_scene)
    val_files = [fp for s in scenes[-max(1, len(scenes) // 10):] for fp in by_scene[s]]

    hf, wf = P.WORK_HW[0] // P.enc_patch, P.WORK_HW[1] // P.enc_patch
    pal = palette(P.N_CLASS)
    os.makedirs(OUT, exist_ok=True)
    # per-sample input/gt/pred trios also go into the latest attn run dir (user convention:
    # comparable with GT in ONE folder; both branch preds share that folder)
    attn_runs = sorted(glob.glob(os.path.join(runlog.RUNS, "*_f2_fusion_attn")))
    run_dir = attn_runs[-1] if attn_runs else runlog.create_run("f2_fusion_attn", {"note": "viz-only"})
    tot_fix = tot_break = tot_diff = 0

    for i, fpath in enumerate(val_files[:n_panos]):
        rgb, lab = P.load_rgb_label(fpath)
        feats = F2.encode_tiles(enc, F2.render_cfg_tiles(rgb, cfg))
        f, g, m = pack_sets(cfg["cid"], feats.reshape(-1, feats.shape[-1]),
                            cfg["geo"], cfg["ncell"], F2.MAX_COV)
        cov = m.any(1)
        fd, gd, md = f[cov].to(DEVICE), g[cov].to(DEVICE), m[cov].to(DEVICE)
        pm = predict_grid(None, head_m, fd, gd, md, cov, hf, wf)
        pa = predict_grid(fusion_a, head_a, fd, gd, md, cov, hf, wf)
        gt = P.label_to_grid(lab, hf, wf)

        valid = (gt != P.IGNORE) & (pm >= 0)
        diff = (pm != pa) & valid
        fixed = diff & (pa == gt) & (pm != gt)
        broken = diff & (pm == gt) & (pa != gt)
        tot_fix += int(fixed.sum()); tot_break += int(broken.sum()); tot_diff += int(diff.sum())

        def colorize(p):
            c = pal[np.clip(p, 0, P.N_CLASS - 1)]
            c[p < 0] = 0.25                               # uncovered = gray
            return c

        fb = rgb.astype(np.float32) / 255.0
        fb_small = fb[::P.enc_patch, ::P.enc_patch][:hf, :wf] * 0.4
        fb_map = fb_small.copy()
        fb_map[fixed] = (0.1, 0.9, 0.1)
        fb_map[broken] = (0.95, 0.15, 0.15)
        fb_map[diff & ~fixed & ~broken] = (0.9, 0.9, 0.2)

        fig, axes = plt.subplots(5, 1, figsize=(12, 14))
        for ax, img, title in zip(
                axes,
                [fb, colorize(gt), colorize(pm), colorize(pa), fb_map],
                ["input ERP", "GT", "fusion=mean pred", "fusion=attn pred",
                 f"attn vs mean — green: fixed {int(fixed.sum())}, red: broken {int(broken.sum())}, "
                 f"yellow: both-wrong {int((diff & ~fixed & ~broken).sum())}"]):
            ax.imshow(img, interpolation="nearest", aspect="auto")
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        fig.tight_layout()
        out = os.path.join(OUT, f"pano_{i}.png")
        fig.savefig(out, dpi=110)
        plt.close(fig)
        if i < 3:                                        # designated samples -> run viz/ (GT alongside)
            runlog.save_seg_sample(run_dir, "s3d", i, rgb.astype(np.float32) / 255.0,
                                   gt, {"mean": pm, "attn": pa}, pal)
        print(f"saved {out}", flush=True)

    print(f"\naggregate over {n_panos} panos: attn-vs-mean differing cells={tot_diff}  "
          f"FIXED={tot_fix}  BROKEN={tot_break}  net={tot_fix - tot_break:+d}", flush=True)


if __name__ == "__main__":
    main()
