# MoGe-2 + inter-pano parallax self-calibration (open lever #1) — experiment log

**Direction (user-chosen 2026-07-09).** After [[pano-ssl-invvar-fusion-negative]] closed lever #2, pursue the
project's OTHER open lever — **#1 inter-pano parallax** (CAN_SSL §7: "the unique genuinely-new lever") — now
feasible because two blockers dissolved: **MP3D arrived on disk** and **MoGe-2 is a ready DUSt3R-class metric
geometry FM** (no from-scratch mini-DUSt3R). Chosen scope: **C, diagnose-first**.

**What C is (honest framing).** C = **A + an unverified inter-pano-parallax bet**. A = "import MoGe-2" (a
strong external geometry FM → better depth/normal, but *using a stronger model*, not an SSL ceiling-break). C's
only novel content over A is the parallax delta: multi-view geometry makes correct pseudo-labels that a single
view can't (LoRA3D's exact mechanism — the ONE Tier-1 precedent in our regime). Distill target = **MoGe-2 /
parallax pseudo-labels → DINOv3+LoRA** (stays pano-ssl-dense = the RICA external-oracle design). NOT
LoRA-on-MoGe-2 (that would be a different, MoGe-2 project).

**Two session-traps that must be threaded (or we repeat this session's negatives):**
- **Q0 / MG-SOG:** MP3D & S2D3D ship **GT depth + poses**. GT is a **yardstick only, never a training signal**
  (training on GT depth = supervised = Q0 fail). SSL-clean lever = MoGe-2 + multi-view **RGB** → pseudo-labels
  (poses-as-calibration-metadata is borderline; decide explicitly at build).
- **Gate 0.5 / correlated-error:** the diagnostic must be **GT-referenced** (single-view MoGe-2 error *vs GT*,
  and the fraction two-view geometry fixes), NOT "where views disagree" (disagreement re-inherits the
  correlated-error failure that just killed lever #2).

**Honest ledger (held).** Even a clean win here is **geometry (depth/normal/pointmap), not seg** — the 57.7
seg ceiling still does not move. But it would be the project's first genuine "SSL raises single-view accuracy"
result, on the one lever that carries new information.

---

## Setup (done, 2026-07-09)

- **MoGe-2 in the `pano` env:** `moge==2.0.0` + `utils3d` installed via **`pip install --no-deps`** (matching
  `hoiio`'s pins: microsoft/moge @`0744441`, utils3d @`3fab839`) — **cu130 torch 2.11.0 left intact** (verified).
  `MoGeModel.from_pretrained("Ruicheng/moge-2-vitl")` loads and runs: `infer(img, apply_mask=True, fov_x=65)` →
  metric `points`/`depth`(m)/`intrinsics`/`mask`. No `normal` (that's the `-normal` variants). **Binary mask,
  no continuous confidence** (so no free σ). API cribbed from `/home/wsw/3dhoi/.../moge_v2.py`.
- **Parallax substrate = Stanford2D3D (already wired), NOT MP3D-first.** area_1 = 190 panos, area_3 = 85, each
  with `global_xyz.exr` + `pose.json` (`camera_location`, K) + GT `depth` → multiple posed panos per building =
  real inter-pano parallax with GT correspondence for free. **area_5 still missing** (0 panos). This runs the
  diagnostic with **zero MP3D extraction cost**; MP3D (1.1 TB on disk, skyboxes NOT yet extracted via
  `extract_skybox.sh`) is the scale-up substrate only if C greenlights.
- **GT metric units:** use `depth.png / 512` meters (65535 = invalid) — validated (~1.66 m office median). The
  `global_xyz.exr` range came out ≈ `‖camera_location‖` (27 m) = a **different pose frame** (S2D3D's
  `camera_original_rotation` / `rotation_from_original_to_point` tangle); **parked** — depth.png is the direct
  per-ray GT, no frame needed.

---

## Diagnostic plan (GT-referenced, diagnose-before-build)

- **P0 — is single-view MoGe-2 already near-GT?** (`scripts/diag_moge_p0.py`) Per E2P tile, MoGe-2 metric range
  vs GT range; AbsRel + δ<1.25, raw AND median-scale-aligned (raw−aligned gap = monocular SCALE error =
  parallax-fixable; aligned = STRUCTURE error). **If near-GT → C collapses to A, stop.** Also answers framing-A
  viability. → *result below.*
- **P1 — the parallax delta.** For points seen in ≥2 area_1 panos (via `global_xyz` GT correspondence), measure
  the fraction of single-view MoGe-2 error that cross-view geometry corrects (GT-referenced). If parallax
  corrects little → weak lever; if a lot → real target → scope the LoRA3D-style build.

### P0 result — 2026-07-09 → **proceed to P1** (not near-GT; moderate, partly parallax-favorable headroom)

`scripts/diag_moge_p0.py`, 20 area_1 panos × 24 tiles = 480 valid tiles.

| metric | RAW (absolute) | median-aligned |
|---|---|---|
| AbsRel ↓ | 0.130 | **0.069** |
| δ<1.25 ↑ | 0.862 | 0.930 |

Scale check: median MoGe range 1.71 m vs GT 1.74 m (**ratio 1.016**) — MoGe-2 metric scale is essentially
correct (validates `depth.png/512`).

**Read.** (1) NOT near-GT (aligned AbsRel 0.069 > 0.05; raw 0.130) → **C does not collapse to A**; there is
headroom. (2) The **global** scale is right (1.016), yet **per-tile** median alignment cuts AbsRel 0.130→0.069
— i.e. **~47% of MoGe-2's error is per-tile SCALE DRIFT** (each view's overall scale wobbles though the average
is metric). Per-view scale is the canonical thing inter-pano parallax pins (LoRA3D's lever) → the headroom is
partly of the *right kind*. (3) Residual **structure** error after scale alignment = 6.9% (δ<1.25 = 0.93):
moderate; some parallax-fixable, some monocular-irreducible.

**Honest EV chain (sobering).** MoGe-2 is already good (δ<1.25 aligned 0.93, ≫ the linear probe's ~0.13–0.19
log-err), so framing **A/B (use MoGe-2 for geometry) is now clearly attractive and cheap**. For C the gain must
survive a chain that attenuates at every step: moderate headroom → fraction parallax actually corrects (P1) →
distill into DINOv3+LoRA → survive a real decoder (§3.9 laundering). **Confound to watch:** part of the per-tile
scale drift could be an E2P-tile-vs-pinhole projection mismatch (a tiling artifact), not monocular ambiguity
parallax can fix — P1 must be GT-referenced to separate them.

**P1 gate (decisive).** Does inter-pano parallax correct a LARGE fraction of MoGe-2's error (→ C worth the
multi-week build) or little (→ error is monocular-irreducible → take A/B)?

**Correspondence UNBLOCKED (frame resolved 2026-07-09).** `global_xyz.exr` is **world-frame, shared across
panos** (each pano's points cluster around its own `camera_location`), with an **x↔z axis-swap** vs the pose's
`camera_location` convention (pano1 centroid `[1.39,20.3,-17.84]` ↔ cam `[-17.84,20.3,1.4]`). The swap is
consistent across panos, so **cross-pano correspondence = matching equal `global_xyz` world coords** — no
convention untangling needed. GT depth stays `depth.png/512`; `norm(global_xyz)`=27 m was distance-from-origin,
never depth. → P1 is buildable: for points seen in ≥2 area_1 panos (different `camera_location` = real
baseline), compare single-view MoGe-2 depth error vs GT to what a two-view scale/geometry constraint recovers.

**⚠️ P1 DESIGN CORRECTION (failure-analysis critic, `docs/FAILURE_ANALYSIS.md` §7).** P1 as first specced
tests only **N1 (source headroom)** — but the decisive condition for C is **N2 (single-view LEARNABILITY)**,
and the most-likely death of C is that the ~47% per-tile scale drift is **aleatoric wobble a single view
cannot recover = the M2 trap in geometry clothes** (cause B in a costume). Headroom-alone → **false
greenlight**. P1 MUST therefore have TWO legs:
- **Leg 1 (N1, source):** fraction of MoGe-2 error a two-view constraint corrects, with the **E2P-tile-vs-
  pinhole projection artifact separated** from the genuinely parallax-fixable part (P0's flagged confound).
- **Leg 2 (N2, sink — the decisive one):** on **held-out** tiles, regress the *parallax-corrected target*
  from **single-view** features (frozen/MoGe) and measure whether one view **predicts** the correction.
  Systematic/predictable → N2 clears (distillable). Unpredictable → N2 fails → C = B-in-costume → **stop**.
- **Pre-register the N3 LOCUS guard for the build** (not P1): distill into the encoder with a
  **starved/linear/frozen read-out** (no fat decoder head between LoRA and target) + instrument
  **gram/CKA-vs-frozen** that the LoRA path actually moved — else it silently reproduces TermB's `gram 0.001`
  and launders under §3.9. Then N4 = real DPT multi-seed vs frozen-DINOv3-DPT.

### P1 floor-check (N2 upper bound) — 2026-07-09 → **FLOOR-PASS = import-A, NOT parallax** (C squeezed)

`scripts/diag_p1_floor.py`, 80 train / 40 test area_1 panos (pano-disjoint), 1920/960 tiles. Ridge held-out
R² predicting the per-tile MoGe→GT scale correction `log s = log·median(d_gt/d_sv)` (spread ≈ ±14%, `log s`
std 0.128):

| predictor | R² | beyond-geom |
|---|---|---|
| geometry (pitch, obliq) | **+0.085** | — |
| MoGe DINOv2 feat | +0.426 | **+0.342** |
| DINOv3 feat (deployment encoder) | +0.291 | **+0.206** |

**Read (advisor-framed as a KILL gate, not a greenlight).** (1) **Not a clean kill** — the MoGe error IS
single-view-learnable (both encoders predict it well beyond geometry) and **content-dependent, not a tiling
artifact** (geom R² 0.085 → not the WRONG-TOOL case). DINOv3 carries ~60% of MoGe's beyond-geom signal
(0.206/0.342). (2) **But it is import-A, not parallax** — a single view predicting the correction means the
error is **monocular-fixable without parallax**; `GT−MoGe` is dominated by MoGe's monocular suboptimality, not
the parallax delta. This licenses ONLY the deferred cross-pano parallax-delta test; it does NOT greenlight the
build. (3) **The squeeze (N2 realized):** MoGe's error = a single-view-predictable part (~⅓, import-A,
parallax-redundant) + a single-view-unpredictable part (~⅔, not distillable into a single-view encoder). Any
*unique* parallax contribution lives in the second part, which a single-view encoder structurally cannot carry.
**⇒ C collapses toward import-A: the distillable gain does not need parallax, and the parallax-needing gain is
not distillable.**

**Verdict.** The evidence-backed path is **framing A — distill MoGe-2 depth → DINOv3+LoRA** (a real, cheap
single-view geometry upgrade over the linear probe's 0.13–0.19 log-err toward MoGe's 0.069; gated by the N3
LOCUS guard so the encoder, not a head, carries it). C (inter-pano parallax) is **not killed but not justified**
— the multi-week cross-pano build's bar (a *strong, parallax-specific* signal) is unmet, and the floor-check
argues the parallax delta is marginal-to-non-distillable. Honest call per the failure analysis: **A over C.**
