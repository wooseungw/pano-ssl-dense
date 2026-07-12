# Sphere-completion SSL (P3: full-sphere context) — experiment log

**Session 2026-07-11.** Follows the "define pano-specialization first" framing: specialization =
measurable gain on one of five axes (P1 distortion, P2 ray-identity, P3 sphere-context, P4
pano-native priors, P5 fusion-friendliness) under head-free/laundering-proof measurement, with
no erosion (G1) and no per-tile regression (G2). P2/P5 are TC3's solved home ground; P4 is
mostly Q1-locked; this session tested **P3 (new)** and **P1 (never content-controlled before)**.

House discipline held: both bets were gated by frozen diagnostics BEFORE any training
(iron law #1), and the training carried one pre-registered guard per failure-analysis root cause.

---

## Gate A — frozen context predictability (`scripts/diag_context_headroom.py`) → HEADROOM

Is a hidden tile's frozen pooled feature already a linear readout of ZERO-OVERLAP (>85°) frozen
context? (S2D3D, scene-disjoint tr 60 / te 30 panos, 24 tiles/pano, D=768.)

| predictor (held-out centered-cos) | ccos |
|---|---|
| context-mean copy / nearest-allowed copy | +0.070 / +0.009 |
| **ridge from strict context** | **+0.496** (R² +0.205) |
| nearest-OVERLAP copy (leaky ceiling) | +0.561 |

Copy baselines ~zero, linear signal real but unsaturated → scene-closure structure exists that
frozen does not linearly expose → pano-MAE-class bet licensed (capability framing only).
Side finding: ceiling/floor rings complete far easier than the horizon ring (0.55/0.57 vs 0.36).

## Gate B — content-controlled obliquity headroom (`scripts/diag_obliquity_headroom.py`) → MARGINAL

Same 3D ray seen by two tiles at different off-axis angles; pair the frozen depth-probe errors at
the SAME overlap cell (N=472,441). The naive stratification shows ~20% (0.223→0.183 across
obliquity quartiles) — but it is mostly CONTENT CONFOUND: the paired same-cell delta is only

| |gap| quartile | mean_delta | rel | sign |
|---|---|---|---|---|
| top (0.105–0.195) | +0.0231 | **+11.5%** | 0.534 |
| lower three | +0.0003…+0.0098 | +0.1…+4.9% | 0.50–0.52 |

**Verdict: MARGINAL.** E2P tangent tiling already neutralizes most distortion at the representation
level; the true distortion cost is small with weak sign consistency. The spherical-RoPE / AdaLN
architecture bet (P1) is **low-EV** — only worth bundling with an independently justified change.

---

## Training — `scripts/train_ssl_completion.py` (15 ep, pinned pool 810, LoRA 0.59M, GPU-hours ≈ 1)

Design guards (one per root cause, pre-registered):
- **L (locus):** completion head FIXED to the gate's linear class `[ctx_mean, f_near, geom5]→D`
  (cannot absorb the task) + a **control head co-trained on frozen pooled features** with identical
  loss/budget → held-out `delta = student − control` IS the encoder contribution, measured online.
- **A (erosion):** dominant distill anchor (L_ANCHOR=1.0 ≫ L_COMP=0.25), token+Gram, random tile
  subset each step; online token-drift instrumentation.
- **B (M2):** capability objective (completion ccos), no per-tile accuracy claim anywhere.

### Result (seed 0; ckpts `runs/ckpt_ssl_completion/{best,last}`)

| val | ep1 | ep2 | ep3 | ep5 | ep6–15 (plateau) |
|---|---|---|---|---|---|
| student ccos | 0.570 | 0.643 | 0.659 | 0.714 | 0.72–0.74 |
| control ccos | 0.552 | 0.629 | 0.649 | 0.708 | 0.71–0.73 |
| **delta (encoder)** | +0.018 | +0.014 | +0.010 | +0.006 | **+0.006…+0.008, sign-stable 10 ep** |

Token drift 0.004–0.007 throughout (anchor held; no erosion signal at the online level).

### Honest read

1. **The encoder contribution is real-looking but SMALL: asymptotically ≈ +0.007 ccos (~1% rel),**
   never negative across 10 plateau epochs. The early +0.018 was head-training asymmetry, not
   encoder learning — deltas must be read at control-head convergence, never before.
2. **Most of Gate A's "headroom" was harvested by simply TRAINING the linear head longer:** the
   frozen-control lane alone reached 0.73, far above the one-shot ridge floor 0.496. Methodological
   lesson for future gates: a ridge floor UNDERESTIMATES the linear-class ceiling; kill thresholds
   should anticipate a trained-head control, or the gate over-promises.
3. Classification per the claim ladder (2,4_REPRESENTATION_QUALITY.md): this is at most a
   *capability/accessibility* delta, single-seed, small-n val (16 panos / 160 targets). It is
   **not** a TC3-scale capability win (ret@1 0.21→0.86) and licenses **no deployment claim**.

### De-risk (COMPLETE, 2026-07-12) & pre-registered call → **SMALL-POSITIVE (3/3)**

Seeds 1–2 launched (`tmux pano_comp_seeds`, logs `scratchpad/train_completion_s{1,2}.log`,
ckpts `runs/ckpt_ssl_completion_s{1,2}`). Pre-registered decision rule:
- plateau delta > 0 with the same sign-stability in **3/3 seeds** → record P3 as
  **small-positive** (first axis where a LoRA encoder beats a matched frozen-control beyond a
  fixed linear class) and only then consider a dense-target v2 + erosion/purity suite;
- any seed ~0 or negative → record as **null** (the linear-class ceiling was the story), close P3.

**Result:** plateau delta (ep6–15, held-out, n=160/val) positive and sign-stable in all three seeds
— never negative in any epoch of any seed:

| seed | plateau delta range | plateau mean |
|---|---|---|
| 0 | +0.0059…+0.0076 | ≈ +0.0069 |
| 1 | +0.0046…+0.0055 | ≈ +0.0051 |
| 2 | +0.0050…+0.0073 | ≈ +0.0066 |

**Recorded verdict: P3 = SMALL-POSITIVE, ≈ +0.006 ± 0.001 ccos (~0.9% rel).** First measured axis
where the LoRA encoder beats a matched frozen-control beyond a fixed linear head class, multi-seed
sign-consistent. Scope guards unchanged: capability-level only; ~30× smaller than a TC3-scale win;
no deployment claim without the dense-target v2 + erosion/purity suite. Ckpts:
`runs/ckpt_ssl_completion{,_s1,_s2}/{best,last}`.

Either way the P3 question exits this session ANSWERED, with the same honest-negative machinery
that closed the accuracy question.
