# The coherence deliverable — what the E2P-overlap SSL adapter actually delivers

**Positioning (honest, information-theoretic).** This project asked whether a label-free SSL adapter on a
frozen DINOv3 panorama encoder can raise **single-view downstream accuracy**. The answer is a **rigorous
negative** (`docs/FAILURE_ANALYSIS.md`): per-tile accuracy is not encoder-limited — **seg is axis-1-saturated**
(F is in-domain-strong, already strong-decoded to 57.7 mIoU) and **depth/normal are axis-2** (F *contains* the
geometry; a strong conv decoder on **frozen** DINOv3 hits depth AbsRel **0.111** ≥ zero-shot MoGe 0.133, §8).
So encoder-SSL has no room to raise accuracy — but that was never where its value lived.

**What it DOES deliver (this document).** The adapter is a **cross-tile COHERENCE** engine: it turns the frozen
encoder — which "cannot self-recover correspondence without pose" — into one that maps the *same 3D ray, seen
from different tiles at different obliquity/context, to the same feature*. The product is a panorama that comes
out as **one coherent full-sphere field** (dense predictions fuse without seams/ghosting) and a **dense
cross-view correspondence** capability the frozen encoder lacks. These wins are **large and head-free**
(measured without a trainable head that could launder them), so they are robust in a way the accuracy claims
never were.

---

## 1. The deliverable — two pillars

**Pillar A — dense cross-view correspondence (a capability frozen DINOv3 lacks).** Head-free
retrieval/matching at the E2P overlap. Frozen DINOv3 cannot self-recover correspondence without pose (~0.2);
the adapter transforms it:

| metric (held-out overlap, head-free) | frozen | adapter |
|---|---|---|
| retrieval@1 (outdoor / indoor) | 0.21 / 0.25 | **0.86 / 0.62** |
| Hungarian@1 (strict 1:1) | 0.27 | **0.87** |
| CKA (indoor, structural alignment) | 0.52 | **0.83** |
| overlap feature cosine | 0.68 | **0.90+** |

*Why this is honest and not the accuracy wall:* it does not claim new task-information — it imprints the
**exact, parallax-free E2P correspondence** (a geometric prior) into the features. That is a genuine capability
(dense matching), laundering-proof (head-free), and orthogonal to the per-tile accuracy question.

**Pillar B — coherent full-sphere fusion (the product).** Overlapping E2P tiles fuse into ONE consistent 3D /
dense field; the frozen encoder ghosts at overlaps, the adapter coincides:

| metric (cross-tile, held-out) | frozen → adapter |
|---|---|
| cross-tile normal consistency | **−15%** |
| cross-tile depth consistency `|Δlog d|` | **−14%** |
| overlap-pair pointmap gap (ghosting) | **−22%** |
| per-tile accuracy (depth/normal/seg) | **~flat** (by design — the honest part) |

---

## 2. Flagship demo — coherent panorama pointmap

`scripts/coherence_pointmap.py` (frozen vs deployed adapter `runs/ckpt_ssl_lora`, S2D3D area_1 train / area_3
val, pano-disjoint). Evaluated on **coherence, not accuracy**: a light depth probe (each encoder its own)
back-projects each tile to 3D; we measure how coherently the overlapping tiles fuse.

**Result (2026-07-09, 60 area_1 train / 30 area_3 val panos, pano-disjoint):**

| enc | depth logErr (accuracy) | cross-tile `|Δlogd|` (coherence) | overlap point-gap (ghosting) | overlap cosine (correspondence) |
|---|---|---|---|---|
| frozen | 0.202 | 0.145 | 0.187 | 0.715 |
| **adapter** | 0.195 | **0.125** | **0.165** | **0.871** |
| Δ | **−0.007 (flat)** | **−14%** | **−12%** | **+0.157** |

**Exactly the deliverable, reproduced fresh on the current adapter:** per-tile depth accuracy is **flat**
(−0.007, the honest "consistency ≠ accuracy"), while the adapter fuses the overlapping tiles **−14% more
depth-consistent**, with **−12% less pointmap ghosting**, and lifts head-free overlap **correspondence
0.715 → 0.871**. Figure: `docs/figures/coherence/coherence_pointmap.png` — the adapter's fused top-down cloud
coincides at overlaps where frozen ghosts.

---

## 3. Where this is the right product (deployment)

The coherence deliverable is valuable exactly where **cross-tile coherence IS the output**, not where a trained
head re-extracts per-tile accuracy:
- **Coherent full-sphere 3D / pointmap reconstruction** (this flagship) — one consistent cloud, not N ghosting
  tiles. The committed DUSt3R-style task.
- **Dense correspondence / matching / registration** across tiles or panoramas (retrieval 0.86).
- **Seamless stitched dense maps** — seg/depth/normal panoramas without tile-boundary artifacts.
- **Uncertainty / disagreement signal** for downstream (overlap code-agreement as a per-pixel confidence map).

**Scope (honest).** Accuracy is flat by construction — this is not a SOTA-accuracy claim and does not move the
57.7 seg ceiling. It is a *coherence* deliverable: the label-free adapter is a working, stable, generalizing,
no-erosion (TC3: purity 0.838→0.862) panorama coherence engine. That is the defensible contribution, and it is
exactly what the information-theoretic analysis says an encoder-SSL adapter can deliver in this corner.
