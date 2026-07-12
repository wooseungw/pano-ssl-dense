# Handoff brief — pano-ssl-dense state for external systems (2026-07-12)

Minimal context another model/agent needs before touching this project. Every claim below is
backed by a doc in `docs/`; do not re-litigate settled items without new external information.

## 1. Project identity and champion state

- Stack: **frozen DINOv3 ViT-B/16 + E2P tangent tiles (3×8 @65°) + LoRA(0.59M) + fusion + decoder.**
- Deployment champion: **TC3 adapter** (`ckpt_ssl_tc3`) — a *coherence* engine, not an accuracy one:
  head-free overlap retrieval 0.21→0.86, Hungarian 0.87, ghosting −22%, purity 0.838→0.862
  (no erosion). Seg fold-1 = 57.7 mIoU; fusion champion = **uniform masked-mean**
  (field 0.557 @64×128, blend 0.611). See `COHERENCE_DELIVERABLE.md`, `RESULTS.md`.

## 2. Settled invariants (do NOT re-propose; 9× confirmed)

1. **SSL-for-accuracy is a rigorous negative in this regime** (in-domain-strong frozen × low-data ×
   dense). Root causes A/B/L in `FAILURE_ANALYSIS.md`; gate logic Q0–Q3 in `SSL_ACCURACY_DESIGN.md`.
2. **Consistency ≠ accuracy** (iron law 1). Pure agreement objectives inject zero information.
3. **Anchor-strength thesis**: without a DOMINANT frozen-teacher distill term, semantics erode
   (purity ladder: TC3 0.862 > geo > frozen 0.838 > EMA > weak-distill 0.753 > none 0.728).
4. **Probe gains launder** (§3.9): only multi-seed real-decoder results count (F-2: +0.078 linear
   → −0.065, 0/5 UPerNet).
5. **Uniform mean beats learned fusion AND inverse-variance/Kalman fusion** (correlated-error
   mechanism; `INVVAR_FUSION_LOG.md` KILL).
6. Graveyard (provably dead / tested-negative): yaw-equivariance, seam loss, gravity-axis,
   closed-layout, ensemble→single distill (M2), learned set-fusion, IV fusion, EMA anchor at small
   scale, weak-distill VICReg (**latest kill: the `train_ssl_vicreg` run, stopped at ep130 —
   L_SEM=1 vs VICReg=25 is the §12 dead config; do NOT resume `runs/ckpt_ssl_vicreg`**).

## 3. New this session (2026-07-11~12) — `SPHERE_COMPLETION_LOG.md`

- **"Pano-specialization" is now DEFINED** as 5 measurable axes + 2 guards:
  P1 distortion-robustness, P2 ray-identity (TC3, solved), P3 sphere-context, P4 pano-native
  priors, P5 fusion-friendliness; guards G1 no-erosion (purity ≥ frozen), G2 no per-tile
  regression; measurement must be head-free or multi-seed real-decoder.
- **Gate A (context predictability): HEADROOM.** Hidden-tile frozen feature from zero-overlap
  (>85°) frozen context: ridge ccos 0.496 vs copy ~0.07 (`diag_context_headroom.py`).
- **Gate B (content-controlled obliquity): MARGINAL.** Same-3D-ray paired errors: true distortion
  cost only +11.5% at top gap quartile, sign 0.53; the naive ~20% stratification was content
  confound. **E2P tangent tiling already neutralizes most distortion** → P1 architecture bets
  (spherical RoPE / AdaLN) are low-EV standalone (`diag_obliquity_headroom.py`).
- **P3 sphere-completion: SMALL-POSITIVE, 3/3 seeds** (`train_ssl_completion.py`). Fixed linear
  head class + co-trained frozen-control lane → encoder contribution = held-out delta =
  **+0.006 ± 0.001 ccos (~0.9% rel)**, sign-stable, drift ≤0.007 (anchor held). First axis where
  the LoRA encoder beats a matched frozen-control beyond a fixed linear class. Capability-level
  only; ~1/30 of a TC3-scale win; NO deployment claim without dense-target v2 + purity suite.
  - **Methodological lesson for all future gates: a one-shot ridge floor UNDERESTIMATES the
    trained-linear ceiling** (control lane alone reached 0.73 ≫ 0.496). Gates must anticipate a
    trained-head control or they over-promise headroom.
- **Tiling Pareto** (`diag_tiling_pareto.py`, analytic): champion 3×8@65 covers only **97.5%**
  (pole holes). Candidate: 12-tile 2×5@90+poles = 100% coverage, multiplicity 2.0, 0.5× compute.
  Exact minimal covering (cubemap 6) has ZERO overlap → kills the SSL/fusion signal → the correct
  formalization is **constrained covering**: min N s.t. coverage=100%, distortion budget (measured
  weak by Gate B), overlap-graph connectivity/multiplicity.

## 4. Machine/run state

- GPU 0: **foreign project** (`hoiio` env, tmux `recon_ablation_s0`, 81GB) — do not touch.
- GPU 1: free. No pano experiments running.
- Checkpoints on disk (repo-ignored): `runs/ckpt_ssl_completion{,_s1,_s2}/{best,last}` (P3),
  `runs/ckpt_ssl_vicreg/` (dead config, keep as user artifact), `runs/ckpt_ssl_tc3` (champion).
- Code/doc state pushed: `main` @ `0f38cd1`.

## 5. Open levers, ranked

1. **Gate C — re-tiling equivalence** (eval-only): does the TC3 coherence deliverable survive at
   12/14 tiles? Prerequisite build: pole-tile pair/warp support in `build_geometry` (ring-only
   today). This is the load-bearing pillar of the paper direction ("provably-minimal tiling +
   coherence adapter at 0.5× compute"); if it fails, the direction honestly converts to an
   analysis result (multiplicity is necessary for coherence).
2. **P3 v2**: dense-target completion + full erosion/purity suite (precondition for any claim).
3. **Import-A**: MoGe-2 depth distill → DINOv3+LoRA (only evidence-backed accuracy path; geometry
   only; N3 starved-readout locus guard + gram/CKA instrumentation mandatory).
4. Parked: shape-from-shading (frozen probe gate first); inter-pano parallax (squeezed by P1
   floor-check — "the distillable gain does not need parallax, the parallax-needing gain is not
   distillable").

## 6. Rules for any new proposal

- Gate it Q0–Q3 (`SSL_ACCURACY_DESIGN.md`) BEFORE building; run a minutes-scale frozen diagnostic
  BEFORE any training (iron law #1); pre-register kill thresholds; include a trained-head control;
  one guard per root cause (A source / B sink / L locus); evaluate head-free or multi-seed
  real-decoder; label transductive results as such; a lower SSL loss is never evidence.
