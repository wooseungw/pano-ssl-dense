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

---

## Import-A build — MoGe-2 depth distill → DINOv3+LoRA (2026-07-17~18, IN PROGRESS)

**Scripts.** `scripts/cache_moge_targets.py` (targets: per-token log-range from MoGe-2 on the champion
E2P tiles, median-pooled to the 32×32 token grid; RGB-only = Q0-clean; 670/670 pool panos cached, 0
failed, 52 min) + `scripts/train_ssl_importa.py` (15 ep, pinned pool 810 via the path-translated
`configs/pool_pin_20260702.local.tsv`, LoRA 0.59M, ~80 min/seed).

**Guards (pre-registered, one per root cause).** N3/L: depth head = STARVED 1×1 conv (D→1) + a control
head co-trained on FROZEN dense features with identical loss/budget → held-out delta = encoder
contribution; token-drift instrumented online. A: dominant anchor L_ANCHOR=1.0 ≫ L_DEPTH=0.25
(token+Gram). B: target is a single-view function (MoGe on the same tile), no ensemble average.
GT depth (`depth.png/512`) is a validation YARDSTICK only, never a training signal.

**Pre-registered decision rule.** Held-out GT log-range RMSE, delta = control − student, read ONLY at
control-head convergence (plateau ep6–15): delta > 0 sign-stable in 3/3 seeds → proceed to N4
(`depth_s2d3d_bench.py` per-adapter vs the three frozen runs) + erosion/purity suite; any seed ≤ 0 →
NULL, stop. Drift guard ≤ 0.02. Deployment artifact = `last` (crash-safe, refreshed every val);
best-by-delta is deliberately NOT kept (delta shrinks as the control head converges → max-delta would
always pick the ep1 head-asymmetry artifact).

**Transductivity note.** The pinned pool contains **zero area_5 panos** (checked 2026-07-18), so the
fold-1 bench comparison is NOT transductive for these adapters; `bench_common`'s blanket
"SSL row is TRANSDUCTIVE" caveat is legacy from older pools and does not apply.

**Interim results.**
- **seed 1 — COMPLETE (15 ep):** plateau delta +0.066…+0.091, **10/10 plateau epochs positive**,
  control converged (gt_rmse 0.3447→0.3443 flat over ep13–15); final student 0.2756 vs control
  0.3443 ≈ **20% lower held-out GT log-RMSE**. Drift 0.008 ≤ 0.02 (anchor held), gram ~0.0001.
  Ckpt `runs/ckpt_ssl_importa_s1/last`.
- **seed 0 — CRASHED at ep13** (GPU1 fell off the bus, 2026-07-18 ~14:5x): plateau ep6–13 delta
  +0.065…+0.081, 8/8 positive before the crash; no final ckpt (pre-fix trainer only wrote `last` at
  exit). Rerun required.
- **seed 2 — not started** (launch chain was waiting on seed 0's GPU).

**Machine incident.** GPU1 hardware fault ("Unable to determine the device handle for GPU1") wedged
CUDA/NVML machine-wide (GPU0 also unusable until reboot). Recovery: reboot, then
`bash scratchpad/relaunch_importa.sh` (reruns s0 + s2 and chains the three N4 depth benches,
EPOCHS=12, vs the existing frozen runs `runs/071{2,3}_*_depth_s2d3d_frozen_f1`).

**Relaunch (2026-07-20).** Post-reboot the chain had NOT been restarted (both GPUs idle since
07-18; the 07-18 s1 bench attempt died at CUDA init in the pre-reboot wedged state). Smoke run
(SMOKE_STEPS=2) confirmed CUDA healthy, 810-pool intact, 0 missing targets. Launched
`scratchpad/relaunch_importa_gpu0.sh` (GPU0-only — GPU1 unproven post-fault): tmux
`pano_importa_gpu0` chains s0 → s2 → N4 depth benches (importa_s0/s1/s2, EPOCHS=12), with
`pano_gpumon` logging temp/power to `scratchpad/gpumon.log` for crash forensics.

**Implementation audit (2026-07-20, prompted by "SSL isn't learning" concern).** Trainer + losses
re-audited: no bug found. The near-zero `tok/gram/drift` in the train log is the INTENDED dominant
anchor (L_ANCHOR=1.0 erosion defense), not absent learning — the learning signal is the pre-registered
held-out delta, and seed 1's plateau delta +0.066 (student 0.2781 vs control 0.3443 GT log-RMSE,
10/10 positive) shows the encoder DOES carry depth beyond the frozen+linear class. Minor note:
logged `drift` duplicates `tok` (both mean 1−cos student-vs-teacher); redundant, harmless.

**SSL-only ablation (2026-07-20, user-requested "turn off everything but self-supervision").**
`L_ANCHOR=0` (anchor OFF, MoGe pseudo-label distill only), SEED=0, 6 ep on GPU1 — seed- and
val-set-matched against the anchored s0 rerun. Log `scratchpad/train_importa_noanchor_s0.log`,
ckpt `runs/ckpt_ssl_importa_noanchor/last`. Result: **the encoder learns features without the
anchor** — student 0.2828 vs its frozen-control 0.4057 GT log-RMSE (~30% lower), delta positive
every epoch (+0.145→+0.123), drift 0.19 (features genuinely moved off frozen DINOv3). Refutes
the "SSL implementation can't learn" hypothesis at the mechanism level. Anchor comparison:
anchored student is equal-or-slightly-better at matched epochs (ep6 0.2806 vs 0.2828) with 10x
smaller drift (0.02 vs 0.19) → the anchor preserves planar semantics at zero depth cost
(anchor-strength thesis confirmed; L_ANCHOR=1.0 stands). Side result: GPU1 survived 1 h at
90 °C under load and returned to healthy idle — post-reboot burn-in passed (gpumon logged).

**SSL-only vs frozen — downstream benches (2026-07-20, user-requested).** The no-anchor adapter
(`runs/ckpt_ssl_importa_noanchor/last`) run through the IDENTICAL fold-1 protocol as the three
frozen baselines (EPOCHS=12, same head/decoder, 782 tr / 373 va):
- **Depth** (`runs/0720_1920_depth_s2d3d_importa_noanchor_f1`): AbsRel 0.1303, RMSE 0.4274,
  d1 85.5, SI-d1 86.7 vs frozen (3 runs) AbsRel 0.1252–0.1428, RMSE 0.401–0.443, d1 82.5–85.0.
  Read: **parity-to-marginal-plus** — d1/SI-d1 sit just above the frozen spread (+0.5 pp over the
  best frozen), AbsRel/RMSE inside it. The +0.12 linear-probe delta largely LAUNDERS under a real
  2.36M decoder (§3.9), as the failure analysis predicted. Single seed; not conclusive alone.
- **Seg** (`runs/0720_1950_seg_s2d3d_importa_noanchor_f1`): mIoU **53.4** vs frozen 58.5/60.5/60.5
  (mean 59.8) → **−6.4 pp EROSION**. Drift 0.19 without the anchor destroyed planar semantics —
  exactly the failure mode L_ANCHOR=1.0 is designed to prevent (cause-A confirmation by ablation).
- Caveat hygiene: the auto-printed "SSL row is TRANSDUCTIVE" line is the legacy bench_common
  blanket; this pool contains zero area_5 panos (2026-07-18 check) → fold-1 rows are NOT
  transductive.
Net read: pure-SSL learning is real but its downstream depth value barely survives a decoder,
while its semantic cost is large. The anchored adapters (s0/s1/s2 → N4 benches on the GPU0
chain) are the configuration that can win depth WITHOUT paying the seg tax — that comparison
remains the decisive one. Meanwhile the anchored **seed 0 rerun COMPLETED 15 ep**: plateau
delta +0.065…+0.081, 10/10 positive, final student 0.2692 vs control 0.3369 → **2/3 seeds
positive**, s2 training.

**SSL-only LONG run — 200 ep scaling probe (2026-07-20, user-directed; IN PROGRESS).** Question:
does pure-SSL (no-anchor) keep improving with more epochs, and where do the three target tasks
(depth, seg, pointmap) go as drift accumulates? Setup: `L_ANCHOR=0 SEED=0 EPOCHS=200 SNAP_EVERY=20
PATIENCE=20 MIN_DELTA=5e-4 VAL_EVERY=5`, ckpt `runs/ckpt_ssl_importa_noanchor_long` (trainer gained
snapshot + train-loss early-stop; snapshot path verified end-to-end with a 1-ep run). Early-stop =
epoch-mean train dep_s not improving >5e-4 for 20 consecutive epochs (the user's "stop when loss
stops improving", at measurement cadence). GPU swap per user: long train on GPU0 (tmux
`pano_noanchor_long`); GPU1 (tmux `pano_gpu1_chain`, `scratchpad/relaunch_gpu1_chain.sh`) reruns
the three anchored N4 depth benches (s0 bench was killed mid-run by the swap, restarted clean),
runs the frozen vitb16 pointmap baseline, then auto-benches every ep20/40/... snapshot on
depth+seg+pointmap (fold-1, EPOCHS=12, identical protocol; `pointmap_bench.py` gained
ENC_ADAPTER/TAG support, smoke-verified). Anchored 3-seed status: **s2 completed (delta +0.059,
plateau positive) → 3/3 seeds positive**; N4 benches queued on GPU1.

**N4 VERDICT — POSITIVE (2026-07-20, pre-registered rule met).** All three anchored adapters beat
the ENTIRE frozen 3-run spread on every depth metric (fold-1, identical protocol): AbsRel
0.1192/0.1193/0.1209 vs frozen 0.1252–0.1428; d1 86.8/87.0/87.1 vs 82.5–85.0; SI-d1 88.1–88.2 vs
83.3–86.3. Runs `runs/0720_2055/2120/2145_depth_s2d3d_importa_s{0,1,2}_f1`. Combined with 3/3
positive training deltas → the import-A distilled encoder carries a real depth gain THROUGH a
2.36M decoder (no §3.9 laundering): the project's first SSL-beats-frozen downstream result.
Remaining for the erosion/purity suite: anchored seg bench (expect ~frozen, unlike no-anchor's
−6.4 pp) — queue on GPU1 after the snapshot watcher's ep60 gap.

**Long-run interim (ep50/200).** Train dep_s still falling (0.0327→0.0180→0.0150 @ep20/40/50; no
early-stop). Snapshots: depth d1 86.4 (ep20) → 87.2 (ep40, edges past the anchored trio); seg
mIoU 54.6 @ep20 (erosion persists, −5.2 pp vs frozen mean); pointmap @ep20 beats frozen on all
three axes (logDepthErr 0.164 vs 0.195, xtile 0.127 vs 0.146, pointGap 0.169 vs 0.192). Val
student GT-RMSE 0.2411 @ep50 (best yet). Shape so far: geometry keeps improving with epochs,
semantic erosion does not heal.

**Long-run FINAL (2026-07-21) — early-stopped ep175; downstream saturates at ~ep40.** Train dep_s
fell 6x past ep20 (0.0327→0.0055) but downstream flat-lined far earlier — full trajectory
(snapshots ep20…160, fold-1, identical protocol):
| ep | dep_s | depth d1 | seg mIoU | pointmap err/xtile/gap |
|---|---|---|---|---|
| 20 | .0327 | 86.4 | 54.6 | .164/.127/.169 |
| 40 | .0180 | **87.2** | 55.7 | .163/.125/.166 |
| 80 | .0100 | 86.5 | 55.1 | .161/.122/.162 |
| 120 | .0067 | 86.4 | 54.5 | .158/.121/.162 |
| 160 | .0056 | 86.8 | 54.3 | .158/.120/.162 |
Depth peaks ep40 then oscillates ±0.4 (noise); pointmap saturates ~ep100; seg stays −4…−5.5 pp
below frozen (never heals, slowly worsens); val student GT-RMSE flat ~0.246 from ep50. Read:
**post-ep40 SSL-loss reduction is MoGe-target overfitting that buys zero downstream** — the
distillable signal is exhausted in ~40 ep. 175-ep ckpt: `runs/ckpt_ssl_importa_noanchor_long/last`
(unbenched; ep160≈ep175 by flatness).

**Anchored seg — erosion NOT eliminated (2026-07-21).** importa_s1 seg mIoU **55.87**
(`runs/0721_0107_seg_s2d3d_importa_s1_f1`) vs frozen 58.5/60.5/60.5 → **−3.9 pp vs frozen mean**,
below the frozen spread. The anchor MITIGATES erosion (no-anchor −6.4/−5.5 pp) but does not
prevent it — the "anchor preserves semantics at zero cost" read from the 6-ep matched pair was
too optimistic at 15 ep. s0/s2 seg benches launched (seed spread check). Deployment note: LoRA
is detachable, so the seg tax only binds a SINGLE-encoder-all-tasks deployment; per-task adapter
switching (depth: adapter ON, seg: OFF) sidesteps it entirely.

**Anchored seg 3-seed (2026-07-21) — erosion is seed-stable.** mIoU 56.98/55.87/55.22 (s0/s1/s2,
mean 56.0) vs frozen 58.5/60.5/60.5 → **−3.8 pp**, all three below the frozen spread. Confirms
the anchor mitigates (no-anchor −5.5…−6.4) but does not eliminate the seg tax.

**REVIVAL arms — failed ideas on the working scaffold (2026-07-22, user-directed; IN PROGRESS).**
Hypothesis: the consistency-family ideas died by cause A (self-referential targets = no new
information); with MoGe distill as an EXTERNAL information source, they may now act as useful
shapers. Training form + validation UNCHANGED (anchored import-A, delta rule, GT yardstick);
terms added as opt-in losses in `train_ssl_importa.py` (L_WARP/L_CODE, byte-identical when 0):
- **+WARP** (`L_WARP=0.25`, GPU0, ckpt `runs/ckpt_ssl_importa_warp`): patch-level E2P
  warp-equivariance on overlap pairs (the original geo-SSL core, `warp_equivariance_loss`).
- **+CODE** (`L_CODE=0.25`, K=512, GPU1, ckpt `runs/ckpt_ssl_importa_code`): M1 swapped-code
  prediction on overlaps vs sinkhorn-balanced FROZEN-teacher codes (`code_swap_loss`) — semantic
  identity pressure; the candidate seg-erosion mitigator.
Pair sampling reuses `TS.build_geometry` on the champion tiling (spec order verified identical to
the MoGe target cache). Both smoke-verified (warp=0.307, code=6.246=ln512 at init). Protocol:
15 ep SEED=0 → depth/seg/pointmap fold-1 benches (identical to all prior rows). Baselines to
beat: anchored depth-only (d1 86.8–87.1, seg 55.2–57.0, pm .158–.164/.120–.127/.162–.169 domain);
frozen (d1 82.5–85.0, seg 58.5–60.5, pm .195/.146/.192).

**REVIVAL arms — first results (2026-07-22, single-seed).** Anchored depth-only pointmap baseline
filled in (importa_s0/s1/s2: .180–.181/.137–.138/.182–.184). Results vs that trio:
- **+WARP**: depth d1 86.7 (inside trio spread), seg 56.6 (inside), pointmap **.179/.132/.177** —
  xtile −.005…−.006 and pointGap −.005…−.007 beyond the trio's ±.001 spread = a real cross-tile
  coherence gain, mechanism-matched (patch-level equivariance).
- **+CODE**: depth d1 **87.5** (+0.4 pp over trio max — hint, needs seeds), seg 56.2 (inside; the
  hoped-for erosion rescue did NOT materialize — frozen-teacher codes keep the cause-A ceiling on
  seg), pointmap .178/.131/.177 (same coherence gain as WARP).
Read: on the working scaffold the revived terms are no longer erosive/flat — they buy a small,
mechanism-consistent pointmap-coherence improvement; SSL-only remains the pointmap champion
(.158/.120/.162) via wholesale geometric drift. CONFOUND: both arms also switched tile sampling
to overlap PAIRS; attribution requires the sampling control. → launched: **pairctrl** (GPU0,
PAIR_SAMPLE=1, zero extra loss, `runs/ckpt_ssl_importa_pairctrl`) and **CODE seed 1** (GPU1,
`runs/ckpt_ssl_importa_code_s1`), each with the full depth/seg/pointmap chain.

**Stitch-fusion exploration (2026-07-22, user-directed "smoother overlaps than the linear mean").**
Added `STITCH_W=feather` to `bench_common.stitch_field` (opt-in; default `uniform` stays
bit-identical to every historical row): Hann tile-center blending — each tile's contribution
fades to ~0 exactly where its footprint ends, so the coverage-count step at footprint boundaries
vanishes. Purely GEOMETRIC weighting, deliberately NOT error-dependent confidence (the
inverse-variance lane is a documented negative, INVVAR_FUSION_LOG). Evaluated with the trained
importa_code depth head, full fold-1 (373 panos), no retraining (`stitch_head_explore.py`,
`stitch_edge_metric.py`, figure `runs/stitch_explore_importa_code/stitch_compare.png`):
- Board: AbsRel 0.1197→0.1202, RMSE 0.3840→0.3838, d1 87.45→87.28 — statistically unchanged.
- Seam step (coarse-grid adjacent pairs |dlog d| at coverage-count transitions vs equal-count
  reference): uniform **1.05x** its interior ref (= the seam) → feather **0.94x** (seam excess
  GONE; equal-count pairs identical 0.0199 both → interior untouched).
- Visual: tile-lattice ghosting (ceiling arcs, rectangular patches) eliminated; |log diff|
  concentrates exactly on the tile-footprint lattice.
Verdict: feather = free smoothness, zero accuracy cost, zero training. Use `STITCH_W=feather`
for deployment/viz outputs; benches keep `uniform` default for board comparability. Next rung
(only if residual smoothness is ever needed): a trained post-stitch ERP refiner head — parked,
no evidence it is worth the protocol change.

**REVIVAL — attribution & replication RESOLVED (2026-07-22).**
- **pairctrl** (pair sampling, zero extra loss): d1 86.9, seg 56.4, pointmap **.181/.137/.182** —
  every number inside the depth-only anchored trio. The pair-sampling confound contributes
  NOTHING; the pointmap-coherence gain belongs to the LOSS TERMS.
- **CODE seed 1**: pointmap **.179/.132/.176** — replicates seed 0's coherence gain (2/2 seeds,
  vs trio .137–.138 xtile / .182–.184 gap ⇒ xtile −4…5%, pointGap −3…4%). Depth d1 86.8 —
  seed 0's 87.5 did NOT replicate (seed noise; CODE spread 86.8–87.5 overlaps trio). Seg 55.9 —
  no rescue, as expected under cause A.
**Final revival verdict.** On the working import-A scaffold, the previously-failed patch-level
consistency terms produce a REAL, seed-replicated, control-attributed, mechanism-matched gain in
cross-tile pointmap coherence — and nothing else (depth, seg unchanged). Modest but genuine:
the first instance in this project of a dead idea adding measurable downstream value once an
external information source anchors the objective. Config choice: +CODE (or +WARP) at 0.25 is
free coherence for pointmap-facing deployments; depth-only remains the simplest for depth-only use.

**M1-FAITHFUL code-swap (2026-07-22, user intent: "run the SSL term the ORIGINAL way").** The
first revival used a SHORTCUT code-swap (teacher's frozen codes as target, one-directional warp,
8-tile subsample) — convenient but NOT how M1 (SEMANTIC_IDENTITY_SSL, `train_ssl_m1.py`) defined
it. Added the faithful path to `train_ssl_importa.py` (`CODE_TGT=online`, `PAIR_ALL=1`):
- **online codes**: target = the STUDENT's OWN prototype scores, sinkhorn-balanced JOINTLY over
  the whole tile batch, stop-grad (`code_targets`) — proper swapped-prediction, not a teacher
  distill. Collapse monitor `perp` (student softmax usage perplexity /512) now logged: sinkhorn
  targets are balanced by construction so only the STUDENT can collapse.
- **PAIR_ALL**: full tile set + ALL bidirectional overlap pairs (`build_geometry_bidir`), matching
  M1's coverage instead of the 4-pair subsample.
Two arms, SEED=0, 15 ep + full depth/seg/pointmap chain:
- **codeon** (online codes only, GPU0, `runs/ckpt_ssl_importa_codeon`).
- **m1full** (online + bidir pairs + warp 0.25, GPU1, `runs/ckpt_ssl_importa_m1full`) = the
  closest reproduction of the original M1 objective, now anchored by MoGe distill.
Smoke-healthy: code 6.2→5.6 (ln512 start, dropping), perp ~480/512 (no collapse), warp 0.30,
delta +0.145. Question: does the FAITHFUL swap (self-referential target) beat the shortcut
(teacher target) on the scaffold, or does its self-referentiality re-expose cause A? Baselines:
shortcut CODE .131/.176 (xtile/gap), depth-only trio .137/.183, seg frozen 59.8.

**TERMINOLOGY CORRECTION (2026-07-22, user-insisted — logged so it is not repeated).** Everything
above labelled "SSL" that uses a MoGe-2 target OR a frozen-teacher token/Gram anchor is
**DISTILLATION**, NOT self-supervised learning. The information source is an external
supervised/pretrained model, so the data-processing bound applies (FAILURE_ANALYSIS cause A). Only
losses whose target is the data's own geometry (warp-equivariance on E2P overlaps) or a
target-free anti-collapse (VICReg var/cov) are PURE SSL. Correct labels going forward:
- "import-A / distill lane" = MoGe-2 depth distillation (label-free w.r.t. OUR data, but supervised
  in origin). NOT SSL.
- "anchor" (tok+Gram to frozen DINOv3) = self-distillation. NOT SSL.
- "PURE SSL" = warp-equivariance (overlap feature learning) + VICReg, LoRA encoder, NO distill.

**PURE-SSL run — the actual thing (2026-07-22, user protocol).** Objective on `train_ssl.py` with
`W_DISTILL=0` (distill fully OFF), `W_REG=1.0`, `GAMMA=1.0` (VICReg is now the SOLE anti-collapse,
canonical target, since the distill anchor that previously did that job is gone):
    L = w_warp · warp_equivariance(overlap)  +  (25·var + cov)      # pure SSL, no teacher target
Only the LoRA-adapted DINOv3 encoder is trained (0.59 M); NO task head is trained during SSL
(structure-preserving, contamination-free). Teacher forward skipped when W_DISTILL=0 (1.87→1.37
s/it). 200 ep, snapshot every 5 ep (`SNAP_EVERY`, `runs/ckpt_ssl_pure/epN`), same pinned pool.
Health @step50: erank 61.9 (NO collapse without the anchor — VICReg holds), warp 0.35 active.
**Evaluation (user protocol): identical LINEAR-PROBE head for frozen vs every SSL snapshot** —
`DECODER=linear` = bare 1×1 conv (D→C, ~769 params) added to depth/seg benches (pointmap ridge is
already linear). A linear probe cannot launder depth out of frozen features (unlike the 2.36 M
decoder), so it isolates what the ENCODER learned. Watcher `scratchpad/linprobe_watch.sh` (GPU1
after m1full): frozen linear baseline once, then depth/seg/pointmap on each epN snapshot. This is
the project's first properly-labelled pure-SSL-vs-frozen curve on all three tasks.

**PURE-SSL raw-VICReg → COLLAPSE (2026-07-22).** The raw-feature run (`ckpt_ssl_pure`, var/cov on
the backbone directly) collapsed: erank 61.9 → **~17 stable** as w_warp ramped in. Diagnosis (user's
own insight, correct): VICReg var/cov are anti-collapse REGULARIZERS, not the SSL signal; the SSL
learning is warp-equivariance (the invariance/positive-pair term). On RAW backbone features var/cov
are too weak to balance the warp-invariance pull → dimensional collapse (only ~17/768 dims survive).
ep5/ep10 snapshots kept + linprobe-benched as the downstream collapse evidence (frozen linear
baseline depth d1 79.6 for reference — note linear-probe frozen is far below the 2.36M-decoder
frozen's 82–85, as expected: the probe can't launder).

**OPTION 1 — canonical VICReg on an EXPANDER (2026-07-22, user "1번으로 빡세게").** `train_ssl_vicreg.py`
already implements it: overlap-invariance + var/cov computed on a DISCARDED expander (local 1024 +
global + sub-token, 4.44M aux), canonical L_INV=25/L_VAR=25/L_COV=1, gamma=1.0, `SEM=none` (NO
anchor — the clean test of "does proper var+cov alone hold rank?"). Smoke verdict: **erank
65.9→81.0 rising, NOT collapsing** — moving var/cov to the expander fixes the raw-feature collapse
at the mechanism level (backbone rank held at ~75 with zero distill). Launched 200 ep on GPU0
(BATCH=1, expandable_segments; BATCH=2 OOMs the 3-scale graph at 24GB), SNAP_EVERY=5, built-in
early-stop on val-EMA plateau. Eval: `scratchpad/vicreg_lpwatch.sh` (GPU1) — same DECODER=linear
frozen-vs-snapshot 3-task probe. This is the properly-labelled test of whether PURE SSL (no distill
of any kind) can move a linear-probe above frozen.

**"CVPR-level SSL loss" — honest reframing + strongest untested shot (2026-07-22, user request).**
The user asked to raise the SSL loss to CVPR level. Honest staff read (grounded in THIS project's
own evidence, not hand-waving): a better loss does NOT break the seg wall — that is structurally
proven, not a loss-engineering gap:
- SSL_SUCCESS_CASES_LIT (67 papers): only NeCo/DIP/LoRA3D beat a strong frozen FM on dense
  accuracy; NONE are MIM; ZERO in our regime (frozen + low-data + dense).
- E1 position pretext = NULL (frozen already encodes pitch/FOV, Q1-null, killed pre-train).
- E2/E3 Term B cross-view completion (CroCo mechanism) = NULL on every head (depth/normal/seg
  linear all tie-or-worse vs frozen, 3-seed). Structural ceiling: predictor absorbs completion →
  encoder frozen; frozen target → frozen ceiling; intra-pano no-parallax → completion ≡ consistency.
- Iron law: consistency ≠ accuracy; strong decoders launder linear-probe gains (§3.9).
→ CVPR-level here is either (A) the rigorous NEGATIVE + 3-cause taxonomy + collapse/laundering
mechanisms (analysis-track, ~complete), or (B) the ONE genuine-SSL design ranked #1 by the gate
but NEVER benched: **PANO-iBOT** (`train_pano_ibot.py`). Unlike Term B, its target is an EMA
momentum teacher (evolving, not the frozen ceiling) — structurally able to leave cause B.
L = ibot_loss(student[mask], EMA_teacher[mask]) + L_GRAM·gram_anchor(student, frozen) +
L_KOLEO·koleo. Smoke healthy: ibot 5.7→0.7, gram 0.14→0.02 (anti-erosion holds), no collapse.

**Pivot (2026-07-22).** GPU0 moved from the slow/weak expander-VICReg (erank held ~35–65 at
warm 0.27 — expander DOES fix the raw-collapse, mechanism logged; ep5 snapshot kept + linprobe-
benched as the VICReg data point) to **PANO-iBOT 200 ep, SNAP_EVERY=5** (`runs/ckpt_ssl_ibot`).
Unified watcher `scratchpad/unified_lpwatch.sh` (GPU1) linprobe-benches BOTH vicreg/epN and
ibot/epN on depth/seg/pointmap vs the shared frozen-linear baseline. Framing is GEOMETRY-first
(depth/pointmap), per the lit — seg is the acknowledged hard wall. Decision rule: iBOT linear-probe
> frozen-linear on depth/pointmap, sign-stable across snapshots → the genuine-SSL positive
(method contribution); else → the negative is confirmed by the strongest design → path A.
