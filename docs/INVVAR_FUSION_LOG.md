# Inverse-variance depth fusion (open lever #2) — experiment log

**Thesis.** `docs/CAN_SSL_RAISE_ACCURACY.md` §6-7 and `docs/PANO_ADAPT_RECIPE_GATE.md` name exactly two
unclosed levers; this pursues **#2 — inverse-variance / Kalman fusion for depth/pointmap**, with a
**label-free per-view uncertainty σ learned from cross-view disagreement**.

**Why it is gate-safe (does NOT re-enter the SSL-accuracy negative).**
- It is **not a representation-learning objective** — it is an inference-time *fusion* method. The encoder
  (incl. LoRA) is NOT in the optimizer; feature gradients are stop-grad'd. So it never engages iron law 1
  (consistency≠accuracy) or the M2 trap (single-view). What "consistency" teaches here is **error magnitude
  (σ), not the answer** — so iron-law-1 does not apply.
- Fusion is **closed-form** inverse-variance (Kalman), not a learned combiner → not the F-2 §3.9 laundering
  class. The head estimates σ only. Distilling the fused field back into the encoder = the M2 graveyard =
  **forbidden**.
- **Failure mode is benign:** if σ carries no signal it degrades to σ=const ⇒ uniform mean ⇒ the validated
  0.557/§3.6 baseline. Worst case is *no gain*, never erosion.

## Design

**Observation model.** At an overlap point p (E2P tiles share one optical center ⇒ zero parallax ⇒ both
tiles see the same ray ⇒ f* cancels): `f_i(p) = f*(p) + ε_i`, `ε_i ~ N(0, σ_i² I)`.
MAP/Kalman fusion: `f̂ = Σ Λ_i f_i / Σ Λ_i`, `Λ_i = 1/σ_i²`.

**Supervision (label-free) — pairwise-residual heteroscedastic NLL** (Kendall & Gal, pairwise form).
σ is a **depth**-uncertainty head (Gate 0 killed the feature residual: `1−cos` 0.074 partial vs `|Δlogd|`
0.261 — see Results). The depth residual `d = logd_i − logd_j = ε_i − ε_j` has `Var(d) = σ_i²+σ_j²`
(independence), so (scalar depth ⇒ D=1):
```
L = Σ_p Σ_(i,j)  |logd_i − logd_j|² / (σ_i²+σ_j²)  +  log(σ_i²+σ_j²)
σ_i = head(sg(f_i), obliquity_i),   log σ² parameterization
logd_fused = Σ Λ_i logd_i / Σ Λ_i,   Λ_i = 1/σ_i²
```
Balance point σ_i²+σ_j² = E|d|² ⇒ the NLL self-calibrates (σ→∞ blocked by log, σ→0 by the quadratic).
(The feature-space `‖f_i−f_j‖²` form would train σ to predict the wrong thing — do not use it.)

**Collapse modes & guards (pre-registered).**
1. *Representation collapse* (most fatal): if encoder grad flowed through the numerator it would collapse
   features to view-invariant mush (M1 erosion / iron law 1). Guard: **stop-grad the features; encoder not in
   optimizer.** "Better features" come from fusion (`scatter_mean → scatter_precision` / output-level Kalman),
   never from moving the encoder.
2. *σ→∞* blocked by `log`. 3. *σ→0* blocked by the quadratic. 4. *σ=const* → degrades to uniform (benign,
   the good property); detect via held-out σ-vs-residual Spearman/AUROC. 5. *Obliquity shortcut* — head ignores
   features, outputs a cos-lat weight; **control**: an obliquity-only head, the delta is the feature-conditional
   information. 6. *Identifiability* — NLL only fixes pairwise σ_i²+σ_j²; a connected overlap graph identifies
   individual σ_i (a triangle suffices). Note: the global σ scale is **gauge-free for the fusion** (cancels in
   Σ Λf/Σ Λ); a weak scale prior is needed only for a *calibrated* uncertainty output.

**Honest caveat (independence).** Same encoder + same optical center ⇒ ε_i,ε_j not independent (ρ>0). A
confidently-wrong-**and-agreeing** point gets a small residual ⇒ low σ ⇒ fusion over-trusts it. Therefore σ
trained on residual then validated on residual is **circular** — Gate 1 must validate σ against **held-out GT
error**. (ρ-correction `σ_i²+σ_j²−2ρσ_iσ_j` deferred to v2.)

## Gates (house style: diagnose before train)

- **Gate 0** (frozen, minutes, kill-cheap): does cross-view disagreement predict per-view depth error, BEYOND
  obliquity? `scripts/diag_disagreement_error.py`. Kill if no correlation; MODE-5 if all obliquity.
- **Gate 1** (after σ-head train): held-out **σ vs GT depth-error AUROC** (NOT vs residual — circular) + the
  obliquity-only-control delta. This is the per-view discrimination test (Gate 0 only tests the pairwise
  necessary condition).
- **Gate 2** (Q3, laundering-proof): **output-level** inverse-variance depth fusion vs **uniform** depth
  fusion (§3.6), real DPT decode, **multi-seed ≥3–5**, pre-registered min-Δ against ~±0.003 seed noise.
  Output-level (N-decode, fuse per-tile depth) removes the §3.9 decoder-laundering path; the single-decode
  feature-precision-field is the *efficiency* variant (v2) that reintroduces laundering. Report absolute
  depth err, |Δlogd|, δ<1.25. **Indoor-scoped** (S2D3D GT).

## Results

### Gate 0 — 2026-07-09 (frozen DINOv3, S2D3D, tr=60 / va=30 panos, N=472,441 overlap cells) → **PROCEED**

`scripts/diag_disagreement_error.py`.

| signal | Spearman vs per-view depth error |
|---|---|
| **resid_depth `|Δlogd|`** | **0.266** |
| resid_feat `1−cos` | 0.085 |
| obliquity `min(cos)` | −0.071 (weak baseline / mode-5 signal) |
| **resid_depth ( · \| obliquity)** | **0.261** (partial — orthogonal to obliquity) |
| resid_feat ( · \| obliquity) | 0.074 |
| obs-model `resid²~err²` | 0.266 |

Heteroscedasticity confirmed: mean |err| 0.284→0.265→0.253→0.238 across obliquity quartiles (least→most
weight), but disagreement carries **~3.7× more** error-signal than obliquity and is nearly independent of it.

**Findings.** (1) Existence condition met — a label-free σ has real signal (not mode-5). (2) **σ source =
task-space depth residual, not feature residual** (`|Δlogd|` 0.261 partial ≫ `1−cos` 0.074): build **depth-σ +
output-level Kalman fusion** (also the laundering-free path); the feature-precision-field variant is dead as a
signal source. (3) **Honest ceiling:** 0.266 is *moderate* — a perfect σ may still yield only a small gain over
uniform-mean (§9.5 "uniform captures ~all fusion headroom"); Gate 2's pre-registered min-Δ is load-bearing.
Gate 0 is a *necessary* condition (disagreement predicts pair error) — the *sufficient* test is whether IV
fusion beats uniform (Gate 0.5).

### Gate 0.5 — TRAIN-FREE inverse-variance fusion — 2026-07-09 → **KILL** (lever does not cash out)

`scripts/diag_invvar_fusion.py` (frozen DINOv3, S2D3D, tr=60/va=30, field 64×128, N=199,315 covered cells,
shrinkage τ=0.084 = median loo-σ). Raw σ_i = |logd_i − median_{j≠i} logd_j|; closed-form IV-fuse the frozen
linear-probe depth vs uniform / median / trimmed. Coverage: cov1=0.116, cov2=0.246, **cov≥3=0.638** (reach was
NOT the limiter). Mean per-pano-normalized log-depth error (↓):

| subset | n | uniform | median | trimmed | IV | IV−uniform |
|---|---|---|---|---|---|---|
| all covered | 199,315 | **0.1784** | 0.1804 | 0.1802 | 0.1796 | +0.0011 |
| cov==2 | 49,096 | 0.1763 | 0.1763 | 0.1763 | 0.1763 | −0.0000 |
| **cov≥3** | 127,153 | **0.1771** | 0.1802 | 0.1799 | 0.1789 | **+0.0018** |
| **hi-disagree (cov≥2, top-30%)** | 52,875 | **0.2043** | 0.2090 | 0.2085 | 0.2072 | **+0.0029** |

**Verdict: KILL.** (1) cov==2 → IV ≡ uniform exactly (−0.0000), as designed (equal loo-σ). (2) At cov≥3 **IV
LOSES to uniform (+0.0018), and loses MOST on hi-disagreement cells (+0.0029)** — exactly where reweighting
should help. (3) **Uniform mean beats every alternative** (median, trimmed, IV all worse) → this is the §9.5
"averaging is king, view-errors are noise-like" result now confirmed for **depth**. Mechanism = the flagged
correlated-error failure: shared encoder + shared optical center ⇒ a confidently-wrong-**and-agreeing** point
has low disagreement ⇒ IV over-trusts it; disagreement predicts *joint* pair error (Gate 0's 0.266) but not
*which view is worse*, so IV cannot reweight correctly. Shrinkage already pulls IV toward uniform (τ>0) and it
still loses — a less-shrunk IV would be worse.

**Does this also kill the learned σ-head?** Effectively yes, at this scale/probe. The head is trained on the
same pairwise-residual signal, so insofar as it learns disagreement it inherits the same harm; its only upside
is the *feature-conditional* signal beyond disagreement, which Gate 0 measured at a weak 0.074 partial — far too
small to reverse a result where uniform beats even median/trimmed. Not worth the build (advisor's decision tree:
"IV loses → strong kill", not "head is last hope" which was reserved for a TIE). A stronger DPT decoder would
*launder* fusion-method differences (§3.9), not reveal an IV win, so Gate 2 is not warranted.

**Lever status.** Open lever #2 (inverse-variance / Kalman depth fusion) is now **tested-negative** — the cheap,
principled form loses to uniform mean; **uniform masked-mean remains the fusion champion for depth as for seg.**
Of the gate doc's two genuinely-open levers, only **#1 inter-pano parallax** remains (the big build: MP3D not on
disk, ≈ mini-DUSt3R). Honest ledger (unchanged): this was never an SSL-accuracy bet — it was inference-time
fusion quality (lever #2), and it did not cash out.

**Caveat:** single-split, linear probe. But the multi-method direction (uniform > robust baselines > IV, worse
on hi-disagree) is a *structural* signal with a mechanistic cause, not a noise artifact — consistent with the
project's repeated finding. Not re-run multi-seed because a train-free LOSS (not a tie) is decisive.
