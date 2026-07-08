# Semantic-Erosion Prevention & Panorama-Robust Features — literature synthesis

**Purpose:** map published methods to this project's pipeline — frozen DINOv3 ViT-B/16 +
~0.59M LoRA (attn q/v) + E2P-overlap cross-view consistency (`train_ssl.py`), champion
**TC3** (cross-view code-swap CE vs FIXED k-means prototypes of frozen-teacher features).

**Provenance / confidence.** Both axes come from deep-research passes (fan-out search → fetch →
3-vote adversarial verification). AXIS 1: **24 claims, all 3-0 unanimous**. AXIS 2 (follow-up pass):
**24 claims, 23 unanimous (3-0), 1 refuted, 1 medium-confidence (2-1)** — noted inline. All rows rest
on primary arXiv sources with verified identity; the one refuted claim (OOOPS RERP-as-primary) is
flagged, and the one non-frozen-guaranteed method (arXiv:2507.09216) is hedged.

---

## AXIS 1 — Preventing semantic erosion / representation collapse (VERIFIED)

Ranked by directness to our pipeline. Each: mechanism → why it prevents erosion → how it maps here.

### 1. Source prototypes/anchors from the FROZEN ENCODER, not a projector head
- **Paper:** *Clustering Properties of Self-Supervised Learning*, Weng et al., **ICML 2025**, arXiv:2501.18452.
- **Finding (3-0):** across ~11 SSL methods, **encoder** outputs have higher ARI / mean-silhouette and *keep improving* over training, while **projector-head embeddings degrade in later stages**.
- **Maps to us:** directly validates TC3 building its fixed k-means prototypes from **frozen-teacher encoder features**, and explains why a consistency loss applied in *projector/code space* (M1) can erode semantics. *Caveat:* their encoder is trained (not frozen); result is "encoder doesn't degrade," related-but-distinct from M1 where the encoder itself eroded.

### 2. TC3 is the SwAV/MSN swapped-prediction paradigm — and M1 collapsed into exactly the trivial solution those methods block
- **Papers:** *SwAV* Caron et al. **NeurIPS 2020** arXiv:2006.09882; *MSN* Assran et al. **ECCV 2022** arXiv:2204.07141.
- **Finding (3-0):** SwAV predicts one view's cluster code from another's representation and blocks all-same-code collapse via an **equipartition (Sinkhorn-Knopp, 3 iters) constraint**; MSN adds a **mean-entropy-max (ME-MAX)** regularizer forcing full prototype use. **M1 had neither** → cross-view agreement was satisfiable by code collapse.
- **Maps to us:** TC3 inherits the swapped-code mechanism but swaps *learned online-balanced* prototypes for *fixed frozen-teacher* ones — reaching the anti-collapse goal by a different route (immune to collapse, but can't adapt prototypes to the pano domain). Sinkhorn equipartition / ME-MAX are the exact constraints M1 lacked.

### 3. Prefer FIXED prototypes over learned ones — learned codebooks under-fill
- **Paper:** *On Partial Prototype Collapse in the DINO Family*, Govindarajan et al., **BMVC 2024**, arXiv:2410.14060.
- **Finding (3-0):** even with balancing, learned prototypes collapse to far fewer unique clusters than K (iBOT ViT-L/16 **8192→969**; DINO-vMF ViT-B/16 **65536→939**). Sharpening only prevents *total* collapse, not duplication; redundant prototypes absorb disproportionate mass.
- **Maps to us:** a strong empirical argument that a learned-prototype variant (M1-style) under-uses its codebook; TC3's **fixed** k-means prototypes are immune to this shortcut. *Correction carried from verify:* DINO-vMF used K=65536 (not 8192) — collapse is worse than first stated.

### 4. Anchor to frozen DINOv3; treat dense-feature collapse over long training as first-class
- **Paper:** *DINOv3*, Siméoni/Vo/Oquab et al., Meta AI, **2025**, arXiv:2508.10104.
- **Finding (3-0):** DINOv3's frozen **dense** features are SOTA; the paper names "**collapse of dense features with large models and long training**" as a "known yet unsolved" problem and introduces **Gram anchoring** — a Gram-matrix consistency loss pulling the student's patch-similarity matrix toward an early-iteration teacher's — to fix it.
- **Maps to us:** validates frozen DINOv3 as teacher; **Gram anchoring is the closest published analogue to our M1 dense-feature erosion** and a candidate guardrail on E2P-overlap features. *Caveats:* dense gains are largest for the 1B model (we use distilled ViT-B/16); adapting Gram anchoring to a frozen-teacher + LoRA-only regime is untested. (Note: our `losses.py` already has a `gram_anchor` implementation — this is the literature basis for elevating it.)

### 5. EMA / self-distillation targets as stable anti-collapse anchors
- **Paper:** *DINO*, Caron et al., **ICCV 2021**, arXiv:2104.14294.
- **Finding (3-0):** the momentum (EMA) teacher is a *necessary* anti-collapse component (k-NN ablation: momentum 72.8% vs previous-iteration 0.1%); DINO balances **centering** (→uniform) against **sharpening** (→peaked) because each alone is a distinct collapse mode.
- **Maps to us:** we distill to a *frozen* (not EMA) teacher; this supports the anchoring principle and offers centering+sharpening as concrete stabilizers if a soft-assignment head is ever put on the overlaps. Our F-3 already explored an EMA anchor.

### 6. Add a variance-covariance / redundancy-reduction term as a cheap symmetric guardrail
- **Papers:** *VICReg* Bardes/Ponce/LeCun **ICLR 2022** arXiv:2105.04906; *Barlow Twins* Zbontar et al. **ICML 2021** arXiv:2103.03230.
- **Finding (3-0):** VICReg = per-dimension **variance hinge** + **covariance decorrelation** (kills informational collapse); Barlow Twins drives the cross-view cross-correlation matrix toward identity. Both need **no negatives, momentum, stop-grad, or big batches** → attach directly & symmetrically to a consistency loss.
- **Maps to us:** we already run VICReg var+cov; this confirms both terms counter the dimensional/informational collapse that strong E2P warp variation can induce, and that strengthening them (or adding a Barlow off-diagonal term on **overlap** features) is architecturally cheap. Canonical VICReg belongs on an expander, not raw backbone (matches our `losses.py` note).

### 7. Why E2P overlap-consistency *risks* erosion (the mechanism)
- **Paper:** *Understanding Dimensional Collapse in Contrastive SSL*, Jing/Vincent/LeCun/Tian, **ICLR 2022**, arXiv:2110.09348.
- **Finding (3-0):** dimensional collapse (embeddings spanning a low-rank subspace) is a subtler failure than constant-vector collapse; **along any direction where augmentation-induced variance exceeds data variance, the weight collapses** → strong augmentation ⇒ low-rank covariance.
- **Maps to us:** strong cross-view **E2P warp variation behaves like aggressive augmentation** → candidate driver of dimensional collapse → motivates the variance-covariance guardrails (#6). *Caveat:* corollary derived under contrastive InfoNCE + linear model; our pipeline is non-contrastive, so "E2P≈strong-aug→collapse" is a reasoned (hedged) inference.

### 8. Parameter-efficient adaptation preserves the prior — and hints a distortion lever
- **Papers:** *Surgical Fine-Tuning* Lee et al. **ICLR 2023** arXiv:2210.11466; *LoRA Learns Less and Forgets Less* Biderman et al. **TMLR 2024** arXiv:2405.09673.
- **Finding (3-0):** LoRA **forgets less** than full FT (keeps out-of-domain performance); surgical FT shows the best layer-subset depends on shift type, and for **INPUT-level shifts (image corruptions) tuning only the first (early) layers works best**.
- **Maps to us:** supports the ~0.59M LoRA as an erosion-resistant stay-near-init anchor, and motivates **concentrating LoRA capacity in EARLY layers** to absorb equirectangular distortion (currently q/v across all layers). *Caveats:* LoRA "forgets less" evidence is from LLMs (cross-modality extrapolation); surgical FT is vision/distribution-shift (close match).

> **REFUTED — do not use (1-2 vote):** the thesis that *hard identity/InfoNCE-style assignment targets disrupt semantic clusters while soft/balanced assignments are inherently "safer."* This did **not** survive verification — do not invoke it to explain M1 vs TC3. Rely on the equipartition/ME-MAX (#2) and partial-prototype-collapse (#3) arguments instead.

---

## AXIS 2 — Distortion-robust / geometry-aware panorama features (VERIFIED, follow-up pass)

Ranked by transfer-friendliness to a FROZEN planar backbone. 24 claims, 23 unanimous (3-0), 1 refuted.

### 1. Tangent images / E2P projection — validates the whole approach (frozen-compatible ✓)
- **Tangent Images**, Eder/Shvets/Lim/Frahm, **CVPR 2020**, arXiv:1912.09390. Render sphere → locally-planar gnomonic tiles on a subdivided icosahedron → an **UNMODIFIED frozen perspective-pretrained net runs directly, NO fine-tuning, ~7% acc loss** (92.6% acc / 93.1% IoU preserved). Our E2P = a coarser tangent-plane family → **the frozen-DINOv3 + E2P design is directly endorsed; move distortion handling into the projection, not the operator.** Caveat: shown for CNNs not ViTs; FoV/angular-resolution normalization is a precondition.

### 2. OOOPS / Open Panoramic Segmentation — closest design-pattern match (frozen-compatible ✓)
- Zheng et al., **ECCV 2024**, arXiv:2407.02685. **FROZEN CLIP + lightweight Deformable Adapter Network (DAN)**; train on pinhole, zero-shot on 360°. DAN is the credited component (+1.4% mIoU; "frozen CLIP necessary for zero-shot"). **GRAFT: a DAN-style deformable adapter (or fold into the LoRA) on frozen DINOv3 to absorb residual pano deformation on top of E2P.** The input-space RERP-augmentation-as-primary framing was REFUTED (0-3) — the backbone-side module is what matters.

### 3. Frozen-backbone adapter peers (LoRA-relevant; medium conf, 2-1)
- Survey arXiv:2606.27745 groups: **OmniSAM** (LoRA on frozen SAM2, <3MB, arXiv:2503.07098) — directly validates our LoRA; **Dense360 / ERP-RoPE** (arXiv:2506.14471) — an equirectangular-aware rotary positional encoding, **drop-in, no backbone change → the single most transfer-friendly graft**; **GoodSAM** (frozen SAM teacher + distortion-aware rectification, arXiv:2403.16370). "They engineer the interface between spherical inputs and planar models" — exactly our pattern. (Dense360 is an MLLM, not dense-seg — adapt the RoPE idea, not the model.)

### 4. Spherical kernel resampling of a pretrained model (planar-compatible; frozen NOT guaranteed)
- arXiv:2507.09216 (Jingguo Liu et al., **ICMEW 2025**) — identity confirmed. Re-projects a pretrained **ConvNeXt**'s conv-kernel sample points onto the sphere for de-distortion; pretrained weights = de-distortion basis + init. **Resamples weights as INITIALIZATION (implies fine-tuning) → not verified strictly frozen.** Concept maps to adjusting DINOv3 patch-embed sample locations per latitude.

### 5. KTN — frozen source model, learned operator transform (CNN not ViT; indirect)
- Su & Grauman, **CVPR 2019**, arXiv:1812.03115. Transfers conv kernels from a FROZEN perspective CNN to ERP via a function of polar angle; needs no annotated 360° data. Validates "learn a per-latitude transform of a frozen operator" but is conv-specific → indirect for a ViT.

### 6. Multi-projection fusion — NOT frozen, but graftable SEAM/OVERLAP-merge mechanisms
- **OmniFusion** (CVPR 2022 Oral, arXiv:2203.00838): tangent patches + **geometry-aware feature fusion** (3D geom + 2D img) to resolve inter-patch discrepancy — **the most relevant mechanism for merging our overlapping E2P tiles.** **UniFuse** (RA-L 2021, arXiv:2102.03550): unidirectional cubemap→ERP fusion at the **decoding stage only** → safe to append after a frozen encoder. **Elite360D/M** (arXiv:2403.16376 / 2408.09336): ERP + icosphere bi-projection fusion. All trained end-to-end → only their fusion modules graft.

### 7. Distortion-native redesigns — INCOMPATIBLE with a frozen ViT (exclude as backbones)
- **PanoFormer** (ECCV 2022, arXiv:2203.09283), **SphereUFormer** (CVPR 2025, arXiv:2412.06968), **Trans4PASS** (CVPR 2022, arXiv:2203.01452). Rebuild tokenization/attention, train from scratch ("none used pretrained weights"). Don't adopt as backbones — but PanoFormer's learnable token-flow / Trans4PASS's Deformable Patch Embedding are the conceptual seeds for the DAN-style adapter (#2).

**Framing (survey arXiv:2606.27745):** *no* surveyed method delivers BOTH strict spherical equivariance AND full reuse of perspective-pretrained weights — "geometric fidelity vs pretrained compatibility" is a structural trade-off. Our frozen-DINOv3 + E2P + LoRA sits, **by principled design, on the pretrained-compatibility side** (an accepted position, not an oversight).

---

## What to try next (cross-axis — both axes VERIFIED)

1. **Keep TC3's fixed frozen-teacher prototypes** (A1 #1,#2,#3): learned prototypes partial-collapse, projector spaces erode; fixed frozen codes can't. Don't revert to M1-style learned codes without Sinkhorn/ME-MAX.
2. **Graft a DAN-style deformable adapter or ERP-RoPE onto the frozen backbone** (A2 #2,#3): OOOPS's Deformable Adapter Network and Dense360's ERP-RoPE are the two most graftable distortion modules — compose with frozen DINOv3 + LoRA without touching backbone weights. **ERP-RoPE is the cheapest drop-in.**
3. **Use OmniFusion-style geometry-aware fusion for E2P overlap merge** (A2 #6): augment the current coverage-mean stitch with a geometry-aware discrepancy fusion; UniFuse's decoding-stage-only fusion is the safe append after a frozen encoder.
4. **Decorrelation guardrail on OVERLAP features** (A1 #6,#7): VICReg-cov / Barlow off-diagonal on the E2P overlap set as a cheap erosion floor.
5. **Elevate Gram anchoring** (A1 #4): A/B `losses.py:gram_anchor` vs current token+relational distill as the dense-feature erosion guard.
6. **Concentrate LoRA in early layers for distortion** (A1 #8): surgical-FT says input-level (distortion) shifts absorb best early — test early-only LoRA vs all-layer q/v.

---
*3-0 verified sources — AXIS 1: arXiv 2501.18452, 2006.09882, 2204.07141, 2410.14060, 2508.10104, 2104.14294, 2105.04906, 2103.03230, 2110.09348, 2210.11466, 2405.09673, 2304.07193. AXIS 2: 1912.09390, 2407.02685, 2606.27745, 2507.09216, 1812.03115, 2203.00838, 2102.03550, 2403.16376, 2408.09336, 2203.09283, 2412.06968, 2203.01452 (+ 2-1: 2503.07098, 2506.14471, 2403.16370).*
