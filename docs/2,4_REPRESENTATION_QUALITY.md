# Sections 2 and 4 — Representation Quality

**Status:** literature-backed evaluation specification  
**Last updated:** 2026-07-11  
**Scope:** frozen DINOv3 teacher vs panorama-adapted student/LoRA representations

## Scope and claim levels

This document defines what must be measured before claiming that a panorama-adapted
student has a representation that is *better* than its frozen teacher. No single
intrinsic metric establishes that claim. The evidence must cover three distinct levels:

1. **Representation health** — the student has not collapsed, become excessively
   redundant, or discarded most feature directions.
2. **Panorama-specific capability** — matched E2P views, seams, and dense
   correspondences are represented more consistently and accurately.
3. **Frozen usefulness and downstream transfer** — the fixed student features support
   better k-NN/probe/decoder performance under the same protocol.

The claim language must match the strongest level actually demonstrated:

| Evidence obtained | Allowed conclusion |
| --- | --- |
| Better rank, variance, covariance, or uniformity only | “The representation is less collapsed / more distributed / less redundant.” |
| Better overlap matching, retrieval, CKA, or seam metrics | “Panorama view consistency or correspondence improved.” |
| Better frozen probes but no real-decoder result | “Linear or neighborhood accessibility improved.” |
| Significant frozen downstream and correspondence gains, with no important regressions | “The student is more useful than the teacher for the evaluated panorama tasks.” |
| Multi-task, multi-dataset, multi-seed gains with no material regression | “The student has a broadly better panorama representation than the teacher under this evaluation scope.” |

“Better representation” must never be inferred from a lower SSL training loss alone.

## 2. Related Work

### 2.1 Alignment and uniformity

Wang and Isola separate contrastive representation geometry into:

- **Alignment**: normalized features of positive pairs should be close.
- **Uniformity**: the full normalized feature distribution should spread across the
  hypersphere rather than concentrate in one region.

For normalized features `f(x)` and positive pairs `(x, y)`, report the paper-style
quantities with the exact exponent and temperature recorded in the result:

```text
alignment = E[ ||f(x) - f(y)||_2^alpha ]
uniformity = log E[ exp(-t ||f(x) - f(y)||_2^2) ]
```

Pano-SSL must split alignment into two cases:

1. appearance-augmentation alignment for the same rendered tile;
2. geometry-conditioned alignment for the same spherical location observed by
   overlapping E2P tiles.

Also report a **random-pair baseline** and `lift = matched similarity - random
similarity`. Alignment alone admits a constant representation; uniformity alone admits
well-distributed noise. Neither is evidence of semantic usefulness in isolation.

Primary source: [Wang & Isola, ICML 2020](https://proceedings.mlr.press/v119/wang20k.html).

### 2.2 Global spectral health

Let `X` be the centered `N x D` backbone feature matrix, `s_i` its singular values, and
`lambda_i` the eigenvalues of its covariance matrix. Report:

- the full normalized singular-value/eigenvalue curve;
- **RankMe**: entropy-based effective rank using normalized singular values;
- **covariance effective rank**: entropy-based effective rank using normalized
  covariance eigenvalues;
- **participation ratio**: `(sum lambda)^2 / sum(lambda^2)`;
- **stable rank**: `sum(s_i^2) / max(s_i^2)`;
- top-1, top-3, and top-10 explained-energy ratios;
- spectral tail energy and the fraction of near-zero directions.

Important repository-specific distinction: `scripts/train_ssl.py::erank` currently uses
the entropy of **covariance eigenvalues**. Since covariance eigenvalues are proportional
to squared singular values, this is not numerically identical to RankMe’s
singular-value entropy. The report must name these separately rather than labeling the
current value “RankMe.”

A larger or flatter spectrum is only an anti-collapse indicator. Noise can increase
rank, and a useful task may naturally occupy a lower-dimensional manifold.

Primary sources:

- [RankMe, Garrido et al., ICLR 2023](https://openreview.net/forum?id=uGEBxC8dnEh)
- [Understanding Dimensional Collapse, Jing et al., ICLR 2022](https://openreview.net/forum?id=YevsQ05DEN7)
- [Matrix Information Theory for SSL, ICML 2024](https://proceedings.mlr.press/v235/zhang24bi.html)

### 2.3 Local dimensionality and neighborhood health

Global rank can look healthy while local neighborhoods occupy degenerate subspaces.
Therefore report, on a fixed token sample:

- local intrinsic dimensionality or a local participation-ratio estimate;
- its median, lower quantile, and panorama/domain distribution;
- teacher-to-student top-k neighbor overlap for several `k` values;
- rank correlation between teacher and student neighbor orderings;
- GT-defined neighborhood precision when semantic/surface labels are available.

Teacher-neighbor preservation measures *fidelity to the teacher*, not superiority.
Improvement over the teacher requires a GT-defined neighborhood target, correspondence
target, or downstream metric. Local dimensionality is sensitive to sample density and
neighborhood size, so it belongs in the post-training extended report rather than the
fast training loop.

Primary sources:

- [LDReg, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/496d8e7c79c39e284d3b461d3fed13d7-Abstract-Conference.html)
- [NeCo: Patch Neighbor Consistency, ICLR 2025](https://arxiv.org/abs/2408.11054)

### 2.4 Variance, covariance, and redundancy

For both the **backbone output** and the **SSL projector output**, report:

- per-dimension standard-deviation mean, minimum, q05, q25, and median;
- the fraction of dimensions below a fixed standard-deviation threshold;
- mean squared off-diagonal covariance;
- mean squared off-diagonal **correlation**;
- covariance Frobenius off-diagonal/diagonal ratio;
- largest-eigenvalue concentration and condition statistics.

Raw covariance is feature-scale dependent; standardized correlation must be reported
beside it. These second-order statistics detect dead dimensions and redundancy but do
not prove statistical independence or semantic usefulness.

Relevant primary papers provide complementary interpretations:

- [VICReg](https://arxiv.org/abs/2105.04906): per-dimension variance floor plus
  off-diagonal covariance reduction.
- [Barlow Twins](https://arxiv.org/abs/2103.03230): cross-correlation toward the identity.
- [W-MSE](https://proceedings.mlr.press/v139/ermolov21a.html): whitening as an explicit
  scattering/anti-degeneracy mechanism.
- [CorInfoMax, NeurIPS 2022](https://papers.nips.cc/paper_files/paper/2022/hash/e4cd50120b6d7e8daff1749d6bbaa889-Abstract-Conference.html): log-determinant covariance barrier against dimensional collapse.
- [Matrix-SSL, ICML 2024](https://proceedings.mlr.press/v235/zhang24bi.html): matrix
  uniformity and covariance alignment.
- [MCR2, NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/6ad4174eba19ecb5fed17411a34ff5e6-Abstract.html): coding-rate expansion and discriminative reduction.

Whitening or rank maximization can flatten a structured teacher spectrum while harming
semantic purity. These metrics are guardrails, not monotonic optimization targets.

### 2.5 Representational similarity and teacher preservation

Report the following on identical samples:

- linear CKA between teacher and student backbone tokens;
- cross-view CKA between matched overlap tokens;
- optional layer-by-layer CKA heatmap;
- token cosine drift and relational Gram drift;
- teacher/student neighborhood overlap at multiple `k`.

CKA measures similarity of representational geometry and is invariant to orthogonal
transformations and isotropic scaling. High teacher-student CKA means preservation, not
improvement. Low CKA means change, not necessarily damage. A useful interpretation is:

- high CKA plus equal downstream performance: mostly teacher preservation;
- high CKA plus better panorama correspondence: a localized panorama adaptation;
- lower CKA plus better downstream performance: useful structural change;
- lower CKA plus worse downstream performance: likely semantic erosion.

Primary sources:

- [CKA, Kornblith et al., ICML 2019](https://proceedings.mlr.press/v97/kornblith19a.html)
- [DINOv3 Gram anchoring](https://arxiv.org/abs/2508.10104)

### 2.6 Dense correspondence and panorama geometry

Cosine similarity is insufficient because it does not test whether the correct target
is retrieved among hard alternatives. On held-out overlap regions report:

- matched cosine, random cosine, and lift;
- Recall@1 and Recall@5;
- mean reciprocal rank or mAP where the candidate set supports it;
- Hungarian one-to-one matching accuracy;
- median and p90 spherical angular correspondence error;
- PCK at predeclared angular/cell thresholds;
- bidirectional and cycle-consistency error;
- seam-band feature/prediction disagreement;
- results stratified by FOV, pitch, obliquity, domain, and overlap size.

Use hard negatives from the same room/panorama or same semantic category. Easy random
negatives can inflate retrieval scores. Geometry-defined overlap positives demonstrate
view consistency, not necessarily semantic improvement.

Primary sources:

- [VICRegL, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/39cee562b91611c16ac0b100f0bc1ea1-Abstract-Conference.html)
- [NeCo, ICLR 2025](https://arxiv.org/abs/2408.11054)
- [Geometry-aware semantic correspondence, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Telling_Left_from_Right_Identifying_Geometry-Aware_Semantic_Correspondence_CVPR_2024_paper.html)

### 2.7 Semantic organization without a heavy decoder

Use identical frozen features and identical evaluation budgets for teacher and student:

- image and pixel k-NN;
- linear classification/segmentation probes;
- 1%, 10%, and 100% label-efficiency curves;
- k-means purity, NMI, and ARI on held-out labeled cells;
- class-balanced and per-class scores.

k-NN tests local semantic organization, while a linear probe tests global linear
separability. Both are sensitive to normalization, optimizer, learning rate, sample
budget, and class balance. The hyperparameter-search budget must be identical and probe
results must not replace real-decoder evaluation.

Primary evaluation example: [DINO, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper).

### 2.8 Paper index and what each adds

| Paper | Evaluation insight relevant here | Important limitation |
| --- | --- | --- |
| Wang & Isola 2020 | Alignment and uniformity as complementary geometry measures | Developed for normalized contrastive representations; neither proves semantics alone |
| RankMe 2023 | Label-free singular-spectrum score correlated with downstream selection | High rank can also be noise; use with transfer metrics |
| Jing et al. 2022 | Dimensional collapse can occur without complete constant collapse | Theory is contrastive/linearized; transfer carefully to dense non-contrastive SSL |
| VICReg 2022 | Separate invariance, variance floor, and covariance redundancy | Projector statistics need not equal backbone quality |
| Barlow Twins 2021 | Identity cross-correlation combines alignment and decorrelation | Whitening pressure can distort a useful teacher spectrum |
| W-MSE 2021 | Explicit whitening/scattering prevents degenerate batches | Batch and sample-estimation sensitive |
| CorInfoMax 2022 | Log-det covariance as a barrier to dimensional collapse | Second-order information does not ensure semantic content |
| MCR2 2020 | Coding-rate expansion/reduction connects diversity and discrimination | Class/partition structure is not directly available in pure SSL validation |
| Matrix-SSL 2024 | Matrix entropy/uniformity and covariance alignment | Global/from-scratch evidence may not transfer to dense LoRA adaptation |
| LDReg 2024 | Global rank can hide local dimensional collapse | Local ID estimators are neighborhood/sample sensitive |
| CKA 2019 | Robust comparison of representational geometry | Similarity, not superiority |
| VICRegL 2022 | Local correspondence and global representation must be evaluated together | Local positives and thresholds define the result |
| NeCo 2025 | Patch-neighbor ordering is a useful dense post-training signal | Teacher-neighbor preservation may only reproduce teacher biases |
| DINO 2021 | k-NN and linear evaluation for SSL representations | Probe choices affect conclusions |
| DINOv3 2025 | Gram anchoring protects dense feature maps during long training | Preservation does not itself add new information |

## 4. Evaluation

### 4.1 Fast validation metrics

Compute on a fixed, domain-balanced diagnostic subset every validation interval. Cache
frozen-teacher features where possible.

| Group | Required fields |
| --- | --- |
| Objective | train total/components, validation total, validation EMA |
| Spectral health | covariance erank, RankMe, stable rank, participation ratio, top-3 energy |
| Variance/redundancy | std q05/median, dead-dim fraction, off-diagonal correlation |
| Distribution | augmentation alignment, overlap alignment, uniformity |
| Correspondence | matched/random/lift, R@1, R@5, cross-view CKA, angular median/p90 |
| Teacher relation | token drift, teacher-student CKA, Gram drift, neighbor overlap@10/@50 |

Every scalar row must contain:

```text
metric_name
teacher
student
delta = student - teacher
better_direction = up | down | neutral
n_samples
domain_breakdown
bootstrap_ci_low
bootstrap_ci_high
```

`better_direction=neutral` is required for similarity/preservation metrics such as
teacher-student CKA, for which neither larger nor smaller universally means better.

### 4.2 Extended post-training metrics

Run once for `best`, `last`, and the frozen teacher:

- full singular-value curves and local dimensionality distributions;
- layerwise CKA heatmaps;
- Hungarian matching, cycle consistency, and viewpoint-bin breakdowns;
- purity/NMI/ARI and multi-budget k-NN/linear probes;
- segmentation/depth/normal/pointmap benchmarks;
- multi-seed paired deltas and confidence intervals.

The report should keep **intrinsic health**, **teacher preservation**, **panorama
capability**, and **downstream usefulness** in separate tables. Combining them into one
unjustified scalar “representation score” would hide important trade-offs.

### 4.3 Decisive downstream transfer

The final teacher-vs-student decision uses the existing frozen-encoder benchmarks:

- segmentation: mIoU and pixel accuracy;
- metric depth: AbsRel, SqRel, RMSE, RMSE-log, delta1/2/3;
- surface normal: mean/median angular error and threshold accuracy;
- pointmap/dense geometry: 3D point gap, correspondence, and fusion consistency;
- real decoder results in addition to linear probes.

Use the same preprocessing, split, decoder, training budget, and evaluation resolution.
Run at least three seeds and all applicable official folds. Report paired deltas,
confidence intervals, and sign consistency. A checkpoint must be selected using an SSL
validation split, never the downstream test score.

Any adapter whose SSL pool contains the downstream test images must be labeled
**transductive**. Such a result can show domain adaptation, but it is not clean evidence
of unseen-data generalization.

### 4.4 Repository mapping

#### 4.4.1 Already available

| Capability | Existing location |
| --- | --- |
| Covariance effective rank | `scripts/train_ssl.py::erank` |
| Validation overlap, teacher/student erank, drift, PCA3 ratio | `scripts/train_ssl_vicreg.py::emit_validation_viz` |
| Matched/random/lift, retrieval@1, Hungarian@1, linear CKA | `scripts/diag_consistency_metrics.py` |
| Multi-seed purity, ARI, frozen probes, paired deltas | `scripts/derisk_tc3.py` |
| VICReg variance/covariance and KoLeo | `losses.py` |
| Segmentation/depth/normal frozen benchmarks | `scripts/*_s2d3d_bench.py` |

#### 4.4.2 Missing from the consolidated report

- exact RankMe distinct from covariance erank;
- stable rank, participation ratio, spectral curves, and tail energy;
- alignment/uniformity pair;
- standardized off-diagonal correlation and dead-dimension statistics;
- R@5, angular error/PCK, cycle consistency, and stratified correspondence;
- teacher-student CKA and neighborhood-order preservation;
- local dimensionality distribution;
- one machine-readable teacher/student/delta/CI schema;
- one final report combining intrinsic, correspondence, probe, and downstream evidence.

### 4.5 Final decision rule

A student passes the “better than teacher” gate only when all of the following hold:

1. no material collapse or redundancy regression on backbone features;
2. panorama correspondence improves with hard-negative and angular-error evidence;
3. at least one predeclared primary downstream family improves significantly;
4. no predeclared major task regresses beyond its equivalence margin;
5. the result is stable across seeds/folds and uses a non-transductive test where a
   generalization claim is made.

If only conditions 1–2 pass, report a **better panorama consistency representation**, not
a generally better representation. If condition 3 passes only for geometry, report a
**better geometry representation** rather than a semantic or universal improvement.
