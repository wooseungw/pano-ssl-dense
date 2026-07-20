"""Extended frozen-backbone comparison incl CLIP & SAM: E2P@50deg seg mIoU on DensePASS.

All backbones see the SAME square E2P tiles (resized to each model's native input),
features patch-matched to a common count + seeded head, so the only variable is the
encoder. DINO/CLIP/SAM have different APIs -> a small per-family adapter.

Run: CUDA_VISIBLE_DEVICES=0 conda run -n pano python scripts/sweep_backbones_ext.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import probe_seg_dinov3 as P  # noqa: E402

DEVICE = P.DEVICE
FOV, OVERLAP, SEED, PMAX = 50.0, 0.25, 0, 35.0
N_TR, N_VA = 40000, 15000          # common patch budget across backbones
DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures", "sweep_backbones_ext")
NORMS = {"imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
         "clip": ([0.4815, 0.4578, 0.4082], [0.2686, 0.2613, 0.2758])}
MODELS = [("dino", "DINOv2", "facebook/dinov2-base"),
          ("dino", "DINOv2-reg", "facebook/dinov2-with-registers-base"),
          ("dino", "DINOv3", "facebook/dinov3-vitb16-pretrain-lvd1689m"),
          ("clip", "CLIP-B/16", "openai/clip-vit-base-patch16"),
          ("pe", "PE-Core-B16\n(SAM3 enc)", "vit_pe_core_base_patch16_224"),
          ("sam", "SAM-B", "facebook/sam-vit-base")]


class Backbone:
    def __init__(self, fam, mid):
        self.fam = fam
        if fam == "dino":
            from transformers import AutoModel
            self.m = AutoModel.from_pretrained(mid).to(DEVICE).eval()
            self.patch, self.dim = self.m.config.patch_size, self.m.config.hidden_size
            self.size, self.norm = (512 // self.patch) * self.patch, "imagenet"
        elif fam == "clip":
            from transformers import CLIPVisionModel
            self.m = CLIPVisionModel.from_pretrained(mid).to(DEVICE).eval()
            self.patch, self.dim = self.m.config.patch_size, self.m.config.hidden_size
            self.size, self.norm = (512 // self.patch) * self.patch, "clip"
        elif fam == "sam":
            from transformers import SamModel
            self.m = SamModel.from_pretrained(mid).vision_encoder.to(DEVICE).eval()
            self.patch, self.dim = 16, self.m.config.hidden_size
            self.size, self.norm = 1024, "imagenet"
        elif fam == "pe":                                  # SAM3 backbone, via timm
            import timm
            self.m = timm.create_model(mid, pretrained=True, num_classes=0,
                                       dynamic_img_size=True).to(DEVICE).eval()
            self.patch, self.dim, self.size, self.norm = 16, self.m.num_features, 512, "pe"
            cfg = self.m.pretrained_cfg
            NORMS["pe"] = (cfg.get("mean", NORMS["imagenet"][0]), cfg.get("std", NORMS["imagenet"][1]))
        for p in self.m.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def dense(self, tile_u8):
        x = torch.from_numpy(tile_u8).float().permute(2, 0, 1)[None].to(DEVICE) / 255.0
        if x.shape[-1] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        m, s = NORMS[self.norm]
        x = (x - torch.tensor(m, device=DEVICE).view(1, 3, 1, 1)) / torch.tensor(s, device=DEVICE).view(1, 3, 1, 1)
        if self.fam == "pe":                               # timm forward_features
            o = self.m.forward_features(x)
            gh = gw = self.size // self.patch
            patches = o[:, o.shape[1] - gh * gw:, :]
            return patches.transpose(1, 2).reshape(self.dim, gh, gw)
        kw = {"interpolate_pos_encoding": True} if self.fam in ("dino", "clip") else {}
        try:
            o = self.m(pixel_values=x, **kw).last_hidden_state
        except TypeError:
            o = self.m(pixel_values=x).last_hidden_state
        if o.dim() == 4:                                   # SAM spatial (B,H,W,C)
            if o.shape[-1] == self.dim:
                o = o.permute(0, 3, 1, 2)
            return o[0]
        gh = gw = self.size // self.patch
        patches = o[:, o.shape[1] - gh * gw:, :]
        return patches.transpose(1, 2).reshape(self.dim, gh, gw)


def head(Xtr, ytr, Xva, yva, steps=800):
    torch.manual_seed(SEED)
    return P.linear_probe(Xtr, ytr, Xva, yva, steps=steps)[0]


def collect(bb, cache):
    plan = P.a2p.plan_tiles("band", FOV, FOV, OVERLAP, pmax_deg=PMAX)
    tf, tl, vf, vl = [], [], [], []
    for sp, (rgb, lab) in cache:
        for tp in plan:
            tile = np.asarray(P.a2p.erp_to_pinhole_tile(rgb, tp.yaw_deg, tp.pitch_deg, FOV, 512))
            feat = bb.dense(tile)
            d, gh, gw = feat.shape
            gl = P.label_to_grid(P.e2p_label(lab, tp.yaw_deg, tp.pitch_deg, FOV, 512), gh, gw)
            (tf if sp == "tr" else vf).append(feat.reshape(d, -1).t().cpu())
            (tl if sp == "tr" else vl).append(torch.from_numpy(gl.reshape(-1)))
    Tf, Tl = P.subsample(torch.cat(tf), torch.cat(tl), N_TR, SEED)
    Vf, Vl = P.subsample(torch.cat(vf), torch.cat(vl), N_VA, SEED)
    return Tf, Tl, Vf, Vl


def main():
    P.configure("densepass")
    panos, groups, train = P.grouped()
    cache = [("tr" if g in train else "va", P.load_rgb_label(f)) for g, f in panos]
    print(f"densepass E2P@{FOV} backbones (patch budget tr={N_TR}/va={N_VA}, seeded) seed={SEED}\n"
          f"{'backbone':14s} {'patch':>5} {'dim':>4} {'in':>5} {'mIoU':>7}", flush=True)
    rows = []
    for fam, name, mid in MODELS:
        try:
            bb = Backbone(fam, mid)
            miou = head(*collect(bb, cache))
            rows.append((name, bb.patch, miou))
            print(f"{name:14s} {bb.patch:5d} {bb.dim:4d} {bb.size:5d} {miou:7.3f}", flush=True)
            del bb; torch.cuda.empty_cache()
        except Exception as ex:
            print(f"{name:14s}  FAIL {type(ex).__name__}: {str(ex)[:70]}", flush=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(rows))
    colors = ["#c44", "#e80", "#48c", "#4a4", "#849"][:len(rows)]
    ax.bar(x, [r[2] for r in rows], 0.6, color=colors)
    for i, r in enumerate(rows):
        ax.text(i, r[2] + .005, f"{r[2]:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([f"{r[0]}\n(p{r[1]})" for r in rows])
    ax.set_ylabel("E2P@50 seg mIoU"); ax.set_ylim(0, max(r[2] for r in rows) * 1.18)
    ax.set_title("DensePASS: frozen backbone comparison incl CLIP & SAM (E2P@50, patch-matched)")
    out = os.path.join(DOCS, "backbone_compare_ext_densepass.png")
    fig.savefig(out, dpi=120, bbox_inches="tight"); print("saved", out, flush=True)


if __name__ == "__main__":
    main()
