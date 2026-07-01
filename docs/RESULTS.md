# Pano-SSL-Dense — E2P-Overlap SSL: Results

**Goal.** Transfer a frozen planar vision encoder (DINOv3) into a panorama feature
extractor by turning the overlap between adjacent E2P (equirectangular→perspective)
tiles into a free, geometrically-exact, label-free self-supervision signal, and adapt
the encoder with a light LoRA — without retraining a panorama encoder from scratch.

**TL;DR.** The E2P-overlap SSL is a **cross-view *consistency* adapter, not an *accuracy*
adapter.** It dramatically and generalizably improves every consistency-type metric
(overlap cosine, dense correspondence retrieval, Hungarian matching, CKA, cross-tile
normal agreement) while leaving per-tile task *accuracy* essentially unchanged (semantic
linear-probe mIoU, normal angular error). Its value lives wherever cross-tile
coherence/matching is the deliverable — not where a trainable head re-extracts accuracy.

![capstone](figures/viz_summary/ssl_summary.png)

---

## 1. Setup

| | |
|---|---|
| Backbone | DINOv3 ViT-B/16 (`facebook/dinov3-vitb16-pretrain-lvd1689m`), frozen |
| Adapter | LoRA r=16, α=32 on attention `q_proj,v_proj` → **0.590 M** trainable params |
| Teacher | adapter-disabled forward of the same model (no second copy) |
| Tiling | E2P square tiles 512², overlap 0.25; indoor hfov 65° (3 pitch rings), outdoor hfov 50° (equator ring) |
| SSL data (label-free) | Structured3D (300) + Stanford2D3D-train (300) + DensePASS-train (70, ×3 oversampled) |
| Eval (held-out) | DensePASS-val (outdoor) & Stanford2D3D area-5 (indoor) |

**Loss.** `L = warp-equivariance (cosine F_A(p)≈F_B(Hp), obliquity-weighted, eroded
overlap, stop-grad target) + distill (per-token cosine + relational Gram to the frozen
teacher) + VICReg (variance floor γ=0.04 + covariance)`, with the warp weight warmed up
0→1 over the first quarter of training.

**Training health** (3 epochs, ~2010 steps, ~70 min, 1 GPU): warp loss **0.31 → 0.12**
(overlap cosine 0.69→0.88), distill bounded ~0.06 (teacher preserved), student effective
rank tracks the teacher (no collapse), VICReg dormant.

---

## 2. The diagnostic that shaped the experiment

Before training we found that **linear-probe mIoU is blind to cross-tile inconsistency**:
frozen DINOv3 already disagrees across overlapping tiles on **~24–28 %** of overlap cells,
but a re-trained linear head *launders* it (blended-overlap features classify as well as
single-tile — averaging is a free ensemble). A FOV sweep showed the disagreement is
*largest* at the distortion-optimal narrow FOV, and that the overlap **ensemble** beats a
single tile by up to **+0.119 mIoU** (DensePASS@50). That ensemble headroom — recoverable
or not — became the thing the SSL aims at, evaluated with **laundering-proof** metrics.

---

## 3. Main result: frozen vs LoRA-SSL

![frozen vs lora](figures/viz_ssl_result/ssl_result.png)

### 3.1 Semantic (ERP-stitched linear probe) — **null**

| dataset | metric | frozen | LoRA | Δ |
|---|---|---|---|---|
| DensePASS@50 | single-best mIoU | 0.326 | 0.338 | **+0.012** (10 % of headroom) |
| | overlap-blend mIoU | 0.445 | 0.449 | +0.004 |
| | cross-tile disagree | 0.323 | 0.321 | −0.003 |
| Stanford2D3D@65 | single-best mIoU | 0.577 | 0.576 | **−0.001** |
| | overlap-blend mIoU | 0.611 | 0.605 | −0.006 |
| | cross-tile disagree | 0.292 | 0.283 | −0.009 |

Directionally correct but trivial in magnitude — accuracy is bounded by the frozen
teacher's content, which the distill anchor deliberately preserves, and the linear head
launders the consistency the SSL adds.

### 3.2 Feature consistency (held-out overlap) — **large & generalizes**

| domain | frozen cos | LoRA cos | Δ |
|---|---|---|---|
| outdoor | 0.680 | **0.914** | **+0.234** |
| indoor | 0.723 | **0.876** | **+0.153** |

### 3.3 Head-free capabilities — **the SSL pays off**

Head-free overlap metrics (no trainable head to launder anything):

| domain | enc | corr | rand | lift | retrieval@1 | sem-match@1 |
|---|---|---|---|---|---|---|
| outdoor | frozen | 0.678 | 0.435 | 0.243 | 0.208 | 0.903 |
| | **LoRA** | 0.896 | 0.498 | **0.398** | **0.855** | 0.960 |
| indoor | frozen | 0.714 | 0.469 | 0.245 | 0.247 | 0.839 |
| | **LoRA** | 0.866 | 0.543 | **0.323** | **0.620** | 0.941 |

The gain is **correspondence-specific, not collapse** (lift rises while the random-pair
baseline barely moves), and dense correspondence **retrieval is transformed** — frozen
DINOv3 "cannot self-recover correspondence without pose" (≈0.2); the adapter reaches
**0.86 (outdoor) / 0.62 (indoor)**.

![cosine distributions](figures/viz_consistency/viz_cosine_dist.png)

### 3.4 Consistency beyond cosine — Hungarian & CKA

![metrics](figures/viz_metrics/viz_metrics.png)

| domain | enc | corr | lift | retrieval@1 | **Hungarian@1** | **CKA** |
|---|---|---|---|---|---|---|
| outdoor | frozen | 0.678 | 0.243 | 0.208 | 0.266 | 0.898 |
| | **LoRA** | 0.896 | 0.398 | 0.865 | **0.865** | 0.924 |
| indoor | frozen | 0.711 | 0.246 | 0.246 | 0.312 | **0.517** |
| | **LoRA** | 0.864 | 0.324 | 0.607 | **0.660** | **0.828** |

- **Hungarian@1** (strict optimal 1:1 assignment) confirms the retrieval gain is not a
  greedy-collision artifact.
- **CKA** (orthogonal/scale-invariant → structural alignment, not a global-rotation cosine
  bump) reveals a nuance cosine missed: **indoor frozen CKA is only 0.517** — the two views
  were *structurally misaligned* — and the SSL jumps it to **0.828 (+0.31)**. Outdoor CKA
  was already 0.90 (band texture self-similar) so little headroom. The biggest *structural*
  alignment gain is indoor, the opposite of where cosine/retrieval headroom looked biggest.

### 3.5 Geometric: surface-normal probe (Stanford2D3D) — accuracy flat, consistency up

| enc | normal angular error ↓ | cross-tile normal consistency ↓ |
|---|---|---|
| frozen | 57.25° | 35.01° |
| **LoRA** | 57.84° (flat) | **29.89° (−15 %)** |

Even on a head-bearing geometric task the pattern holds: **per-tile accuracy unchanged,
cross-tile consistency improved** — and unlike semantic argmax, the consistency survives
the probe. (Linear normal probe is weak in absolute terms ~57°; the controlled
frozen-vs-LoRA delta is the signal.)

![normal consistency](figures/viz_consistency_explain/viz_normal_consistency.png)

### 3.6 Geometric fusion: cross-tile pointmap (the committed DUSt3R-style task)

E2P tiles share one optical center (no parallax), so the same surface point is seen along
the **same ray** by both tiles — fusion ghosting reduces to **depth disagreement** along
that ray. We predict per-pano-normalized log-depth (linear probe), back-project each patch
to 3D (`depth × shared ray dir`), and fuse.

| metric | frozen | LoRA | Δ |
|---|---|---|---|
| depth accuracy (log err) ↓ | 0.195 | 0.189 | flat |
| cross-tile depth consistency `|Δlog d|` ↓ | 0.146 | **0.126** | **−14 %** |
| overlap-pair point gap ↓ | 0.498 | **0.388** | **−22 %** |

Same pattern: per-tile depth accuracy flat, **cross-tile consistency improves** — the
adapter makes multi-tile pointmaps **fuse more coherently**. (The fused cloud is visually
noisy because the depth probe is linear; the gap reduction is clearer quantitatively. A
stronger decoder would sharpen the picture.)

![pointmap fusion](figures/pointmap_fusion/pointmap_fusion.png)

### 3.7 Fine-tuned — placing it on the SOTA scale

To compare against task leaderboards (which fine-tune end-to-end with heavy decoders), we
trained a 2-conv seg decoder on E2P-tile features and stitched to a full-ERP prediction
(Stanford2D3D area-5, full-ERP mIoU @128×256, 150 train panos / 8 epochs — a POC, not a
SOTA-tuned run):

| condition | full-ERP mIoU | note |
|---|---|---|
| frozen encoder + decoder | 0.496 | encoder frozen, decoder only |
| scratch-LoRA + decoder | 0.510 | LoRA learned on the task from scratch |
| **SSL-LoRA + decoder** | **0.513** | our consistency adapter as init |
| *published SOTA* | *0.53 – 0.60* | Trans4PASS 0.53 / 360Mapper 0.54 / SGAT4PASS·SphereUFormer ~0.55–0.60 (full-res, full fine-tune, UDA) |

Two findings: **(1)** unfreezing the adapter helps over frozen (+0.014–0.017); **(2)
SSL-init ≈ scratch-LoRA (0.513 vs 0.510, a tie)** — the consistency pretraining gives **no
accuracy edge** even when you fine-tune with a real decoder and measure on the SOTA scale.
"Consistency ≠ accuracy" holds end-to-end. We sit 2–9 pts below SOTA, explained by the
intentionally-light setup (coarse 128×256 eval, 2-conv decoder, LoRA-only not full
fine-tune, 150-pano POC) — not a capability ceiling; the relative verdicts are the signal.

### 3.8 Merging tiles into one field — adaptive fusion & resolution

To cut the per-tile decode cost, fuse the overlapping tile patch-features into a **single ERP
feature field** and decode **once**. Resolution sweep (frozen, naive scatter-average, 150 panos):

| field | cells | coverage | mIoU |
|---|---|---|---|
| 32×64 | 2 048 | 0.87 | 0.550 |
| **64×128** | 8 192 | 0.82 | **0.557** |
| 128×256 | 32 768 | 0.49 | 0.550 |

**64×128 is the sweet spot** — it matches the tile patch density (~2°/cell). Finer (128×256)
is *finer than the 32×32-patch tiles*, so coverage collapses to 0.49 (the field develops
holes) and mIoU drops. The fused field at 64×128 (~0.557, near per-tile pooled ~0.58) is the
**efficiency win: one decode pass instead of one per tile**.

- **Adaptive obliquity-weighting** (down-weight oblique edge patches) > naive uniform by a
  small **+0.006**.
- **Geometry-guided deformable cross-attention** (learned sampling offset + attention over
  tiles) **ties naive**: at 60 panos it overfit and lost −0.03; a fair retry at 250 panos +
  regularization → ±0.001. The E2P overlap is *exact and parallax-free*, so the geometric
  reference is already pixel-accurate — deformable's "learn where to look" has nothing to fix.
- **LoRA ≈ frozen** again (consistency ≠ accuracy).

Takeaway: high-res naive/obliquity **field averaging** is the practical single-decode fusion;
a *learned* fusion head pays off only with uncertain correspondence (parallax multi-view) or
at much larger data scale.

### 3.9 Common SSL encoder × SOTA decoders (per task) — and a de-risk

We paired the **common SSL encoder** (fixed) with decoder heads from real models —
Segmenter-Linear, SETR-PUP, UPerNet (PPM), Segmenter-Mask — on seg/normal/depth (encoder
fixed, only the decoder trains, so this isolates feature quality). A single-split run hinted
the strongest decoder (UPerNet) might convert some consistency into accuracy (seg +0.024
mIoU). A **multi-seed paired de-risk** (4 seeds, per-seed random 180-pano subset) did **not**
confirm it:

| task (UPerNet) | frozen | SSL-LoRA | Δ (mean±std) | consistent |
|---|---|---|---|---|
| seg mIoU↑ | 0.543±0.028 | 0.547±0.020 | +0.004±0.010 | 2/4 (noise) |
| normal °↓ | 56.32±2.09 | 55.63±1.43 | −0.70±1.47 | 3/4 (within noise) |
| depth \|Δlog\|↓ | 0.156±0.004 | 0.153±0.005 | −0.003±0.002 | **4/4** |
| depth δ<1.25↑ | 0.776±0.010 | 0.783±0.013 | +0.008±0.005 | **4/4** |

The apparent UPerNet **seg gain was a single-split artifact** (shrank to +0.004, |Δ|≪std, 2/4
seeds — with more data even frozen rose 0.517→0.543, closing the gap). The only robust SSL
accuracy gain is on **depth** — small but consistent (4/4 seeds, |Δ|>std). So **"consistency
≠ accuracy" holds across decoder families**; a stronger decoder does not reliably cash
consistency into accuracy, and the de-risk caught a false-positive. (Segmenter-Mask underfit
at this scale and is excluded.) Scripts: `multitask_eval.py`, `verify_upernet.py`.

---

## 4. Unified characterization

Across semantic, head-free, and geometric evaluations the result is one story:

| axis | metric | frozen → LoRA |
|---|---|---|
| **consistency** (improves) | overlap cosine | +0.15 … +0.23 |
| | retrieval@1 / Hungarian@1 | 0.21→0.86 / 0.27→0.87 |
| | CKA (indoor) | 0.52→0.83 |
| | cross-tile normal consistency | −15 % |
| | cross-tile depth consistency (fusion) | −14 % |
| **accuracy** (unchanged) | semantic single-tile mIoU | +0.01 / ~0 |
| | normal angular error | flat |
| | depth log-error | flat |

The thesis **mechanism is fully validated** — a working, stable, generalizing, label-free
panorama adapter that does exactly what it was designed to do. The thesis **value** is
**consistency/coherence**, and its natural home is tasks where coherence *is* the
deliverable (dense correspondence, cross-tile DUSt3R-style fusion, coherent stitched
output) rather than where a trainable head re-extracts accuracy.

---

## 5. Qualitative figures

| figure | shows |
|---|---|
| `viz_correspondence.png` | feature-NN match lines on an overlapping pair: frozen 4/14 → LoRA 14/14 correct |
| `viz_cosine_heatmap_{outdoor,indoor}.png` | per-patch overlap cosine overlaid on the overlap region (input tiles + ERP footprints shown) |
| `viz_feature_pca{,_indoor}.png` | PCA-3 stitched feature panorama — seam coherence across tile boundaries |
| `viz_normal_consistency.png` | predicted-normal maps + cross-tile angular-disagreement heatmap |
| `viz_metrics.png`, `viz_cosine_dist.png` | consistency-metric bars and corr/rand distributions |
| `ssl_summary.png`, `ssl_result.png` | capstone (consistency↑ vs accuracy-flat) and per-domain summary |

![correspondence](figures/viz_consistency_explain/viz_correspondence.png)

---

## 6. Reproducibility

| script | purpose |
|---|---|
| `scripts/train_ssl.py` | LoRA overlap-SSL training (warm-up, VICReg floor, erank monitor) → `ckpt_ssl_lora/` |
| `scripts/eval_ssl.py` | frozen vs LoRA: single-best / blend mIoU + disagreement |
| `scripts/diag_seam.py` | D1: cross-tile disagreement / laundering diagnostic |
| `scripts/diag_cos.py` | held-out overlap feature cosine |
| `scripts/diag_overlap_match.py` | head-free corr/rand/lift/retrieval/sem-match |
| `scripts/diag_consistency_metrics.py` | Hungarian@1 + linear CKA |
| `scripts/probe_normal.py` | surface-normal accuracy + cross-tile consistency |
| `scripts/pointmap_fusion.py` | depth probe → back-project → cross-tile pointmap consistency + fusion viz |
| `scripts/fine_tune_seg.py` | trainable decoder fine-tune (frozen / scratch-LoRA / SSL-LoRA) → full-ERP mIoU vs SOTA |
| `scripts/adaptive_field.py` | naive vs obliquity-weighted ERP feature-field fusion |
| `scripts/adaptive_field_deform.py` | geometry-guided deformable cross-attention fusion (vs naive) |
| `scripts/field_res_sweep.py` | field resolution sweep (32×64 / 64×128 / 128×256) |
| `scripts/viz_*.py` | all figures (→ `docs/figures/<script>/`) |

Run with `CUDA_VISIBLE_DEVICES=<n> conda run -n pano python scripts/<name>.py`.

---

## 7. Limitations & next

- Single seed / single split per dataset; absolute linear-probe numbers are modest
  (probes are intentionally light to isolate frozen-feature quality).
- Normal probe is linear (~57° abs error) — a stronger decoder would sharpen the geometric
  story; the controlled delta is what we report.
- **Pointmap fusion (done, §3.6):** the correspondence/consistency capability *does*
  translate — cross-tile depth consistency −14 %, overlap-pair point gap −22 %, accuracy
  flat. Confirms the consistency-adapter characterization on the committed geometric task.
- **Next:** swap the linear depth probe for an MLP/DPT decoder to sharpen the fused cloud
  (make the ghosting reduction visually dramatic, not just quantitative), and scale the
  SSL pool / try a larger LoRA rank to test whether consistency gains keep growing.
