# SOTA Benchmark Plan — pano-ssl-dense vs published leaderboards

**Goal (user, 2026-07-03):** stop reporting internal-probe numbers; tune our champion pipeline
(frozen DINOv3 ViT-B/16 + 0.59M LoRA E2P-overlap SSL 'TC3' + dense head) to its best and
compare against published SOTA on **all four** benchmarks, under each benchmark's **official
protocol**. SOTA numbers below are from a scoping workflow (WebSearch); `[verify]` = single-source
/ from-memory, must be reconfirmed before any published table.

## 0. The shared truth (why none of our current numbers count)

Our current eval is a **diagnostic probe**, not a leaderboard entry: 32×64 ERP *cell* output,
scored on **covered cells only**, median-normalized depth, custom scene-disjoint splits. Every
benchmark below rejects that. The **shared upgrades** (build once, unblock 3 of 4):

- **U1 — full-resolution decoder.** DINOv3 ViT-B/16 emits 32×64 patch tokens at 512×1024. Replace
  the per-cell light head with a **DPT/UPerNet upsampling decoder** to native ERP resolution.
- **U2 — all-valid-pixel scoring.** Drop the "covered-cell" restriction (it inflates and is
  non-standard); score every valid pixel, excluding only the top/bottom ERP pole mask.
- **U3 — official split loaders** (S2D3D 3-fold for seg; areas 1-4,6 / area5 for depth).
- **U4 — metric-depth head** (depth/pointmap only): absolute meters, **no median align**, ~10 m cap.

Fusion: per §9.7, use **uniform masked-mean** E2P fusion (learned F-2 lost on both seg and depth).

## 1. Per-benchmark scope

### A. Semantic seg — Stanford2D3D (13-class panoramic)  ← highest ROI, closest to SOTA
- **Protocol:** official 3-fold CV on the 1413-pano subset; report **Fold-1 mIoU AND 3-fold avg**.
  Fold-1 = train areas 1,2,3,4,6 / test area 5. Full-ERP argmax, class-unweighted mIoU. Eval
  resolution ~1024×2048 / 1080×2160 `[verify exact]`.
- **SOTA (verified-web):** SGAT4PASS-S **56.4 / 55.3** (fold1/3fold, current SOTA) · Trans4PASS+ 53.6/53.7 ·
  Trans4PASS-S 53.3/52.1 · HoHoNet 53.9/52.0. SphereUFormer 72.2 `[verify — almost certainly a different protocol]`.
- **Our anchors (docs/RESULTS.md, OUR eval — area5, 150–180 panos, NOT the official 3-fold):**
  §3.7 POC (128×256, 2-conv, 150-pano): frozen 0.496 / scratch-LoRA 0.510 / SSL-LoRA 0.513.
  §3.8 fused-field (frozen, naive scatter-avg): 32×64 **0.550** · 64×128 **0.557** (sweet spot) ·
  per-tile pooled **~0.58**. §3.9 UPerNet multi-seed: frozen **0.543** / SSL-LoRA **0.547**.
  → With a real decoder we are ALREADY **~0.543–0.557**, i.e. **at/inside the 0.53–0.56 SOTA band**
  (≈ Trans4PASS+ 0.536, within ~1–2 pts of SGAT4PASS 0.564) — but on our eval, not yet the official fold.
- **⚠ Load-bearing honesty:** across §3.7/§3.8/§3.9 the **SSL-LoRA adapter ≈ frozen on accuracy**
  ("consistency ≠ accuracy", confirmed at 810-pool scale). So the SOTA-competitive system is
  **frozen DINOv3 + E2P tiling + fusion + decoder**; the SSL adapter's value is cross-view
  *consistency*, not mIoU. The **21.8k run (in progress) is the scale test** of whether SSL finally
  earns an accuracy edge — frame the comparison honestly on this.
- **Gap:** U2+U3 + resolution done right (U1: naive 128×256 DROPS to 0.550 from coverage holes —
  needs DPT upsampling, not finer scatter). Infra: **`scripts/fine_tune_seg.py` already targets
  full-ERP-vs-SOTA** — extend it, don't rebuild.
- **Story:** *frozen DINOv3 + E2P competitive with pano-seg SOTA at 0.59M trainable params*;
  SSL = consistency/efficiency bonus (+ pending scale test).

### B. 360 depth — Stanford2D3D  ← biggest scale/metric gap
- **Protocol:** fold-1 (areas 1-4,6 train / **area5 ~373 test**), **512×1024**, **METRIC meters,
  NO median align**, ~10 m cap, valid-pixel mask (poles excluded). Metrics: Abs Rel, Sq Rel, RMSE,
  RMSE(log), δ<1.25^{1,2,3}. ⚠ cross-paper numbers disagree by protocol — compare within one table.
- **SOTA (verified-web, Cross360 Table I, 512×1024 metric):** SGFormer **AbsRel 0.104 / RMSE 0.341 / δ1 90.0** ·
  UniFuse 0.112/0.356/87.1 · PanoFormer 0.112/0.395/88.7 · OmniFusion 0.115/0.381/86.7 ·
  Elite360D 0.118/0.376/88.7 · EGFormer 0.153/0.497/81.9. DAP (foundation) **AbsRel ~0.092** `[verify]`.
- **Gap:** U1+U2+U3+**U4 (metric head — the load-bearing change)**. Our median-norm output is
  scale-invariant and **cannot enter the metric board** without a scale/shift-recovery head.
- **Landing zone:** UniFuse–Elite360D band (AbsRel ~0.11–0.13) = **mid-pack, not SOTA**; foundation
  backbones (DAP ~0.092) top it — beating them likely needs unfreezing / large 360 pretrain
  (contradicts the frozen thesis). **Story:** honest mid-pack at tiny params.

### C. Semantic seg — DensePASS (Cityscapes-19, outdoor real)  ← biggest TASK mismatch
- **Protocol:** **Unsupervised Domain Adaptation.** Source = labeled **Cityscapes pinhole**;
  target = DensePASS (2500 unlabeled + **100 annotated test**), **400×2048**, 19-class mIoU.
- **SOTA (verified-web):** DATR-S **56.81** · Trans4PASS+ 56.38 (multi-scale) · DPPASS mid-50s `[verify]` ·
  360SFUDA++ (source-free) `[verify]`.
- **Gap:** whole new regime — needs **supervised Cityscapes training + a UDA adaptation loop**
  (adversarial / self-training / prototype). Our label-free pano SSL adapter is **not** a UDA method.
  Plus U1+U2. **Highest lift.** **Story:** SSL adapter as a distortion-robust UDA *component* (stretch).

### D. Pointmap / dense-3D — DUSt3R-style  ← collapses into depth
- No clean single-view **pano pointmap** leaderboard exists. pointmap = depth × ray, so it reduces
  to a **depth comparison**. Two regimes (pick one): **metric** (UniFuse/Elite360D/Cross360) or
  **scale-invariant** (DA360 — align before metric). Structured3D depth SOTA: Cross360 AbsRel **0.036** ·
  UniFuse 0.0535. Matterport3D: DA360 0.079 / PanoFormer 0.105 (verified-web).
- **Gap:** same as B (U1+U2+U4) + pick scale regime. **Report as depth**; the novel angle is E2P
  parallax-free cross-tile 3D consistency (our pointmap_fusion thesis), reported alongside depth.

## 2. Recommended sequencing (fastest credible result → hardest)

1. **Seg-S2D3D** — already **~0.543–0.557** (our eval) vs 0.564 SOTA; extend `fine_tune_seg.py` with
   U1+U2+U3 (official 3-fold + native res + verified 13-class). Best ROI, clearest story.
2. **Depth-S2D3D (+ Pointmap)** — one metric DPT head (U4) serves both; U1+U2+U3. Mid-pack target.
3. **DensePASS UDA** — full UDA pipeline; biggest lift; stretch / reframe as SSL-as-UDA-component.

**Build first (GPU-free, while 21.8k SSL pretrain runs):** U1 shared full-res decoder + U2 all-pixel
eval + U3 S2D3D loaders → the Seg-S2D3D official harness. This is benchmark-agnostic infra.

## 2.5 Verified Seg-S2D3D protocol + harness (`scripts/seg_s2d3d_bench.py`, review 2026-07-03)

Adversarial review (web-verified vs Trans4PASS/SGAT4PASS **source code**, HIGH confidence):
- **Folds CONFIRMED:** fold1 test=area5 · fold2 test=area2+4 (hardest) · fold3 test=area1+3+6. Our
  `FOLD_TEST={1:"5",2:"24",3:"136"}` is exactly right.
- **mIoU convention CONFIRMED:** dataset-AGGREGATED, class-unweighted — accumulate inter/union over
  all test panos, per-class IoU, mean over 13 (absent→0). Not per-image averaged.
- **Eval resolution:** Trans4PASS/SGAT4PASS report at **2048×1024** (W×H) full ERP (train crop
  1080²); HoHoNet at 512×1024 (its 53.9/52.0 row is thus lower-res). Native GT = 4096×2048.
- **SOTA rows CONFIRMED** vs SGAT4PASS arXiv Table 1; SphereUFormer 72.2 stays flagged (diff protocol).

Harness bugs found & FIXED: (1) band plan left ~14% valid ceiling/floor pixels uncovered → **now
full_sphere** (coverage reported, ~93%↑); (2) OURS row falsely labelled "0.59M LoRA" → **now honest**
(frozen DINOv3 + 2.36M decoder; SSL-LoRA is fixed/pretrained, not trained here); (3) GT loaded at
512×1024 → **now at EVAL_HW**; (4) mIoU over present-classes-only → **now all-13, absent→0**;
(5) **transductive leakage** — SSL adapters' pretrain pool includes test areas → **report FROZEN as
clean headline; SSL/scaled rows carry a transductive caveat.**

**Headline run config:** `ENC_ADAPTER= EVAL_HW=1024,2048 TILE_OUT=256 TR_PANOS=full VA_PANOS=all-area5
EPOCHS=20 FOLD=1` (frozen, clean) → then SSL-810/TC3/21.8k-scaled with the transductive caveat.

## 2.6 RESULTS — Seg-S2D3D fold-1 (2026-07-04, fixed harness)

stitch 128×256 → eval 512×1024, 800-pano train / all-area5(373) test, 20ep, **decoder-only (2.36M)**:

| encoder (trained = 2.36M decoder) | mIoU | note |
|---|---|---|
| **frozen DINOv3 + E2P** | **57.7** | clean (no leakage) |
| SSL-810 (geo) | 56.6 | transductive |
| SSL-TC3 (champion) | 56.3 | transductive |
| **SSL-21.8k-scaled** | **55.6** | transductive · LOWEST |
| SGAT4PASS (SOTA) / Trans4PASS+ / HoHoNet | 56.4 / 53.6 / 53.9 | full fine-tune, 2048×1024 |

**Finding 1 — frozen DINOv3 + E2P is SOTA-competitive at ~2.4M trainable params.** 57.7 is in/above the
SOTA band. NOT a clean "beats SOTA": our eval is 512×1024 (SOTA 2048×1024) + single fold-1 + single-seed;
coarser scoring inflates indoor mIoU. Honest claim = **competitive/near-SOTA, parameter-efficient**.

**Finding 2 — SSL does NOT beat frozen; scale doesn't help.** All SSL rows < frozen, and 21.8k-scaled is
the LOWEST. So (a) "erosion = data scarcity" is REJECTED for accuracy — scaling SSL to 21.8k bought no
edge; (b) "consistency ≠ accuracy" holds at scale; (c) stronger still — SSL rows carry a transductive
advantage (pretrain pool includes test areas) yet still lose. Caveat: single-seed (1–2 mIoU may be noise),
but the ordering frozen>810>tc3>scaled is suggestive. → multi-seed to firm up.

Bug fixed this round: eval conflated stitch-res with eval-res → coverage collapsed to 21.7% at 1024×2048
(mIoU 15%, artifact). Fix = stitch at coverage-complete 128×256, upsample logit field to eval res
(coverage 97.5%). See `scripts/seg_s2d3d_bench.py` predict_erp.

## 3. Numbers to VERIFY before any published table
- DAP AbsRel 0.0921 (S2D3D depth) — single search summary.
- HoHoNet exact S2D3D depth row.
- SphereUFormer 72.2 S2D3D seg — protocol mismatch suspected.
- DPPASS / 360SFUDA++ exact DensePASS mIoU.
- Cross360's own Stanford2D3D depth row (fetched table contradicted the paper's claim).
- RESULTS.md §3.7 anchor (0.496 / 0.510 / 0.513) — reconfirm against the file.
