"""Stage-0 ceiling probe: does GLOBAL spatial context have headroom on the fused ERP field?

`adaptive_field_deform.py` learned per-cell *sampling* (deformable) and TIED naive averaging —
because the E2P correspondence is exact + full-coverage, so "where to look" has nothing to fix.
But that fusion was per-cell LOCAL (each cell saw only its covering tiles). The untested axis is
GLOBAL context propagation: let a confidently-featured region inform an ambiguous one across the
ERP field (seam-wrapping). This probe trains, SUPERVISED, a global refiner (circular-pad conv +
longitude/latitude axial attention + spherical PE) on the naive blend field and compares to the
naive-blend linear probe — the SAME baseline deform tied.

Decision gate: refine - naive > +0.01 (above the deform tie noise) => global context has headroom,
build the label-free SSL neck (Stage 1). Otherwise STOP — a neck cannot add accuracy here either.

Reuses the deform harness (geometry, naive_field, encode, 250-pano fp16 cache, frozen vs LoRA).
Run: CUDA_VISIBLE_DEVICES=0 conda run -n pano python scripts/field_refine_probe.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data  # noqa: E402
import probe_seg_dinov3 as P  # noqa: E402
import train_ssl as T  # noqa: E402
import adaptive_field_deform as DF  # noqa: E402  (shared harness)
from adaptive_field_deform import HF, WF, NC, naive_field, precompute_geom, encode, dev, gt_field, pos2d  # noqa: E402
from encoder import PanoEncoder  # noqa: E402

DEVICE = P.DEVICE
SEED = 0
EPOCHS = int(os.environ.get("EPOCHS", 20))
TR_PANOS = int(os.environ.get("TR_PANOS", 250))
LAYERS = int(os.environ.get("LAYERS", 2))
HEADS = int(os.environ.get("HEADS", 8))


# --------------------------------------------------------------------------- model
class AxialAttn(nn.Module):
    """Zero-init residual attention along ONE field axis; lat/lon position injected into the
    query/key space. out_proj zero-init => contributes 0 at start (identity)."""

    def __init__(self, d, heads, drop=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        nn.init.zeros_(self.attn.out_proj.weight); nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, x, pos, axis):                  # x, pos: (HF, WF, d)
        z = self.norm(x) + pos
        if axis == "lat":                             # attend over HF within each column
            z = z.transpose(0, 1); o, _ = self.attn(z, z, z)
            return x + o.transpose(0, 1)
        o, _ = self.attn(z, z, z)                     # attend over WF within each row (seam via periodic lon PE)
        return x + o


class RefineBlock(nn.Module):
    """All residual branches zero-init => block(x) == x at start, so the refiner begins AT the
    naive blend field and can only IMPROVE on it. A tie therefore means global context adds
    nothing — never an optimization artifact pushing the field below naive."""

    def __init__(self, d, heads, drop=0.2):
        super().__init__()
        self.ln1 = nn.LayerNorm(d); self.dw = nn.Conv2d(d, d, 3, groups=d)
        self.ln2 = nn.LayerNorm(d)
        self.pw = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, d))
        self.lon = AxialAttn(d, heads, drop); self.lat = AxialAttn(d, heads, drop)
        nn.init.zeros_(self.dw.weight); nn.init.zeros_(self.dw.bias)
        nn.init.zeros_(self.pw[-1].weight); nn.init.zeros_(self.pw[-1].bias)

    def forward(self, x, pos):                         # x, pos: (HF, WF, d)
        z = (self.ln1(x) + pos).permute(2, 0, 1)[None]
        z = F.pad(z, (1, 1, 0, 0), mode="circular")   # wrap longitude (seam)
        z = F.pad(z, (0, 0, 1, 1), mode="replicate")  # clamp latitude (poles)
        x = x + self.dw(z)[0].permute(1, 2, 0)        # local context (0 at init)
        x = x + self.pw(self.ln2(x) + pos)            # channel MLP (0 at init)
        x = self.lon(x, pos, "lon")                   # longitude context (0 at init)
        x = self.lat(x, pos, "lat")                   # latitude context (0 at init)
        return x


class GlobalRefiner(nn.Module):
    def __init__(self, d, layers=LAYERS, heads=HEADS):
        super().__init__()
        self.register_buffer("pos", pos2d(d).view(HF, WF, d))   # sinusoidal lat/lon
        self.blocks = nn.ModuleList([RefineBlock(d, heads) for _ in range(layers)])

    def forward(self, naive):                          # naive: (NC, d) -> (NC, d); == naive at init
        x = naive.view(HF, WF, -1)
        for blk in self.blocks:
            x = blk(x, self.pos)
        return x.reshape(NC, -1)


# --------------------------------------------------------------------------- probe
def run_encoder(tag, enc, tr, va, scatter):
    P.enc_patch = enc.patch
    cache = {"tr": [], "va": []}
    for sp, fl in [("tr", tr), ("va", va)]:
        for f in fl:
            rgb, lab = P.load_rgb_label(f)
            cache[sp].append((encode(enc, rgb, P.plan), gt_field(lab)))

    def eval_miou(ref, decd):
        ref.eval()
        inter = torch.zeros(P.N_CLASS); union = torch.zeros(P.N_CLASS)
        with torch.no_grad():
            for feats, y in cache["va"]:
                nf, cov = naive_field(dev(feats), scatter)
                pred = decd(ref(nf)).argmax(1).cpu()
                cm = cov.cpu(); mm = (y != P.IGNORE) & cm
                for c in range(1, P.N_CLASS):
                    inter[c] += ((pred == c) & (y == c) & mm).sum(); union[c] += (((pred == c) | (y == c)) & mm).sum()
        ref.train()
        return float(np.mean([(inter[c] / union[c]).item() for c in range(1, P.N_CLASS) if union[c] > 0]))

    def train_decode(layers):
        """Train GlobalRefiner(layers)+linear decoder jointly. IDENTICAL protocol for every
        `layers`, so layers=0 (refiner == identity == naive blend field) is the matched baseline
        and Δ isolates the global-context refiner alone. Reports BEST val mIoU over epochs:
        identity-init starts AT naive, so if global context helps, val peaks ABOVE naive BEFORE
        any overfit collapse — best-val reads the true ceiling regardless of late overfitting."""
        torch.manual_seed(SEED)
        ref = GlobalRefiner(enc.dim, layers=layers).to(DEVICE)
        decd = nn.Linear(enc.dim, P.N_CLASS).to(DEVICE)
        opt = torch.optim.AdamW(list(ref.parameters()) + list(decd.parameters()), 1e-3, weight_decay=1e-2)
        lf = nn.CrossEntropyLoss(ignore_index=P.IGNORE)
        g = torch.Generator().manual_seed(SEED)
        init = eval_miou(ref, decd)                           # ep -1: identity == naive (sanity)
        best, traj = init, []
        for _ in range(EPOCHS):
            for i in torch.randperm(len(cache["tr"]), generator=g).tolist():
                feats, y = cache["tr"][i]; yd = y.to(DEVICE)
                nf, cov = naive_field(dev(feats), scatter)
                opt.zero_grad(); lf(decd(ref(nf)[cov]), yd[cov]).backward(); opt.step()
            m = eval_miou(ref, decd); traj.append(m); best = max(best, m)
        print(f"    L{layers} init={init:.3f} val/ep=[{' '.join(f'{m:.3f}' for m in traj)}] best={best:.3f}", flush=True)
        return best

    nv = train_decode(0)
    refine = train_decode(LAYERS)
    gate = "✅ headroom" if refine - nv > 0.01 else "❌ tie (stop)"
    print(f"{tag:8s}  naive(L0)={nv:.3f}   global-refine(L{LAYERS})={refine:.3f}   Δ={refine-nv:+.3f}   {gate}", flush=True)
    del cache; torch.cuda.empty_cache()


def main():
    P.configure("stanford2d3d"); P.TILE = DF.TILE
    P.plan = P.a2p.plan_tiles("band", DF.HFOV, DF.HFOV, 0.25, pmax_deg=45.0)
    scatter, _, _ = precompute_geom(P.plan)
    files = data.list_erps("stanford2d3d")
    def area(f): return f.split("extracted_data/")[1].split("/")[0]
    tr = [f for f in files if "5" not in area(f)][:TR_PANOS]
    va = [f for f in files if "5" in area(f)][:40]
    print(f"global-context refiner ceiling probe: field={HF}x{WF} tiles={len(P.plan)} "
          f"layers={LAYERS} heads={HEADS} tr={len(tr)} va={len(va)} ep={EPOCHS}", flush=True)
    print("gate: Δ>+0.01 => global context has headroom (deform tied here) => build SSL neck\n", flush=True)
    for tag, kw in [("frozen", dict(lora_rank=0)), ("LoRA", dict(adapter_path=T.CKPT))]:
        enc = PanoEncoder(model_id=P.MODEL, **kw).to(DEVICE).eval()
        run_encoder(tag, enc, tr, va, scatter)
        del enc; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
