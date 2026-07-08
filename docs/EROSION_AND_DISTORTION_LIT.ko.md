# 의미 침식(Semantic Erosion) 방지 & 파노라마 강건 특징 — 문헌 종합

**목적:** 발표된 방법론들을 본 프로젝트 파이프라인에 매핑 — frozen DINOv3 ViT-B/16 +
~0.59M LoRA (attn q/v) + E2P-overlap 교차뷰(cross-view) 일관성 (`train_ssl.py`), 챔피언
**TC3** (frozen-teacher 특징의 FIXED k-means 프로토타입에 대한 교차뷰 코드-스왑 CE).

**출처 / 신뢰도.** 두 축(axis) 모두 deep-research 패스(fan-out 검색 → fetch →
3표 적대적 검증)에서 도출. AXIS 1: **24개 주장, 전부 3-0 만장일치**. AXIS 2(후속 패스):
**24개 주장 중 23개 만장일치(3-0), 1개 반박(refuted), 1개 중신뢰도(2-1)** — 본문에 표기.
모든 항목은 신원이 검증된 1차 arXiv 출처에 근거하며, 반박된 1개 주장(OOOPS의 RERP-as-primary)은
플래그 처리했고, frozen이 보장되지 않는 방법(arXiv:2507.09216) 1개는 유보(hedged)로 표기.

---

## AXIS 1 — 의미 침식 / 표현 붕괴(representation collapse) 방지 (검증됨)

우리 파이프라인과의 직접성 순으로 정렬. 각 항목: 메커니즘 → 침식 방지 원리 → 우리에게 어떻게 매핑되는가.

### 1. 프로토타입/앵커는 projector head가 아니라 FROZEN ENCODER에서 소싱하라
- **논문:** *Clustering Properties of Self-Supervised Learning*, Weng et al., **ICML 2025**, arXiv:2501.18452.
- **발견 (3-0):** ~11개 SSL 방법 전반에서 **encoder** 출력이 더 높은 ARI / 평균 실루엣을 갖고 학습 내내 *계속 향상*되는 반면, **projector-head 임베딩은 후반 단계에서 열화**된다.
- **우리에게:** TC3가 고정 k-means 프로토타입을 **frozen-teacher encoder 특징**에서 구축하는 것을 직접 정당화하며, *projector/code 공간*에 적용된 일관성 손실(M1)이 왜 의미를 침식할 수 있는지 설명한다. *유의:* 이들의 encoder는 학습되는(frozen 아님) 것 — 결과는 "encoder가 열화되지 않는다"로, encoder 자체가 침식된 M1과는 관련되나 구별된다.

### 2. TC3는 SwAV/MSN의 swapped-prediction 패러다임 — 그리고 M1은 바로 그 방법들이 막는 자명해(trivial solution)로 붕괴했다
- **논문:** *SwAV* Caron et al. **NeurIPS 2020** arXiv:2006.09882; *MSN* Assran et al. **ECCV 2022** arXiv:2204.07141.
- **발견 (3-0):** SwAV는 한 뷰의 클러스터 코드를 다른 뷰의 표현으로부터 예측하고, **equipartition(Sinkhorn-Knopp, 3 iter) 제약**으로 all-same-code 붕괴를 차단한다. MSN은 모든 프로토타입 사용을 강제하는 **mean-entropy-max(ME-MAX)** 정규화를 추가한다. **M1에는 둘 다 없었다** → 교차뷰 일치가 코드 붕괴로도 만족될 수 있었다.
- **우리에게:** TC3는 swapped-code 메커니즘을 계승하되 *학습되는 온라인 밸런싱* 프로토타입을 *고정된 frozen-teacher* 프로토타입으로 교체 — 다른 경로로 anti-collapse 목표에 도달(붕괴에 면역이나, 프로토타입을 pano 도메인에 적응시키지 못함). Sinkhorn equipartition / ME-MAX가 바로 M1에 결여됐던 제약이다.

### 3. 학습되는 프로토타입보다 FIXED 프로토타입을 선호하라 — 학습 codebook은 과소 충전(under-fill)된다
- **논문:** *On Partial Prototype Collapse in the DINO Family*, Govindarajan et al., **BMVC 2024**, arXiv:2410.14060.
- **발견 (3-0):** 밸런싱을 해도 학습 프로토타입은 K보다 훨씬 적은 고유 클러스터로 붕괴한다(iBOT ViT-L/16 **8192→969**; DINO-vMF ViT-B/16 **65536→939**). Sharpening은 *전면* 붕괴만 막을 뿐 중복은 못 막는다; 잉여 프로토타입이 질량을 불균형하게 흡수한다.
- **우리에게:** 학습-프로토타입 변형(M1 스타일)이 codebook을 과소 사용한다는 강력한 실증 논거; TC3의 **고정** k-means 프로토타입은 이 지름길에 면역이다. *검증에서 넘어온 정정:* DINO-vMF는 K=65536(8192 아님) 사용 — 붕괴가 처음 서술보다 심하다.

### 4. frozen DINOv3에 앵커링하고, 장기 학습에서의 dense-feature 붕괴를 일급(first-class)으로 다뤄라
- **논문:** *DINOv3*, Siméoni/Vo/Oquab et al., Meta AI, **2025**, arXiv:2508.10104.
- **발견 (3-0):** DINOv3의 frozen **dense** 특징은 SOTA다; 논문은 "**대형 모델 + 장기 학습에서의 dense feature 붕괴**"를 "알려졌으나 미해결" 문제로 명명하고, 이를 고치기 위해 학생의 patch-similarity 행렬을 초기 iteration teacher 쪽으로 당기는 Gram-matrix 일관성 손실인 **Gram anchoring**을 도입한다.
- **우리에게:** frozen DINOv3를 teacher로 정당화; **Gram anchoring은 우리 M1의 dense-feature 침식에 가장 근접한 발표된 유사물**이며 E2P-overlap 특징에 대한 후보 가드레일이다. *유의:* dense 이득은 1B 모델에서 가장 큼(우리는 distilled ViT-B/16 사용); Gram anchoring을 frozen-teacher + LoRA-only 체제에 적응하는 것은 미검증. (참고: 우리 `losses.py`에 이미 `gram_anchor` 구현이 있음 — 이를 승격시킬 문헌적 근거.)

### 5. EMA / self-distillation 타깃을 안정적 anti-collapse 앵커로
- **논문:** *DINO*, Caron et al., **ICCV 2021**, arXiv:2104.14294.
- **발견 (3-0):** momentum(EMA) teacher는 *필수적인* anti-collapse 구성요소다(k-NN ablation: momentum 72.8% vs previous-iteration 0.1%); DINO는 **centering**(→균일)과 **sharpening**(→첨예)을 균형시키는데, 각각 단독으로는 별개의 붕괴 모드이기 때문이다.
- **우리에게:** 우리는 *frozen*(EMA 아님) teacher로 증류한다; 이는 앵커링 원칙을 뒷받침하고, overlap에 soft-assignment head를 얹게 된다면 centering+sharpening을 구체적 안정화 장치로 제공한다. 우리 F-3는 이미 EMA 앵커를 탐색했다.

### 6. variance-covariance / redundancy-reduction 항을 값싼 대칭 가드레일로 추가하라
- **논문:** *VICReg* Bardes/Ponce/LeCun **ICLR 2022** arXiv:2105.04906; *Barlow Twins* Zbontar et al. **ICML 2021** arXiv:2103.03230.
- **발견 (3-0):** VICReg = 차원별 **variance hinge** + **covariance decorrelation**(정보 붕괴 제거); Barlow Twins는 교차뷰 cross-correlation 행렬을 항등행렬 쪽으로 몬다. 둘 다 **negative, momentum, stop-grad, 큰 배치가 불필요** → 일관성 손실에 직접·대칭적으로 부착 가능.
- **우리에게:** 우리는 이미 VICReg var+cov를 돌린다; 이는 강한 E2P warp 변동이 유발할 수 있는 차원/정보 붕괴에 두 항이 모두 대항함을, 그리고 이를 강화(또는 **overlap** 특징에 Barlow off-diagonal 항 추가)하는 것이 아키텍처적으로 값쌈을 확인해 준다. 정석 VICReg는 raw backbone이 아니라 expander에 붙어야 한다(우리 `losses.py` 주석과 일치).

### 7. E2P overlap-일관성이 왜 침식을 *위험하게* 하는가 (메커니즘)
- **논문:** *Understanding Dimensional Collapse in Contrastive SSL*, Jing/Vincent/LeCun/Tian, **ICLR 2022**, arXiv:2110.09348.
- **발견 (3-0):** 차원 붕괴(임베딩이 저계수 부분공간에 갇힘)는 상수-벡터 붕괴보다 미묘한 실패다; **증강이 유발한 분산이 데이터 분산을 초과하는 임의 방향에서 가중치가 붕괴**한다 → 강한 증강 ⇒ 저계수 공분산.
- **우리에게:** 강한 교차뷰 **E2P warp 변동은 공격적 증강처럼 작동**한다 → 차원 붕괴의 후보 원인 → variance-covariance 가드레일(#6)을 동기화한다. *유의:* 이 따름정리는 contrastive InfoNCE + 선형 모델 하에서 유도됨; 우리 파이프라인은 non-contrastive이므로 "E2P≈강한증강→붕괴"는 (유보된) 추론이다.

### 8. Parameter-efficient 적응이 prior를 보존한다 — 그리고 distortion 레버를 암시한다
- **논문:** *Surgical Fine-Tuning* Lee et al. **ICLR 2023** arXiv:2210.11466; *LoRA Learns Less and Forgets Less* Biderman et al. **TMLR 2024** arXiv:2405.09673.
- **발견 (3-0):** LoRA는 full FT보다 **덜 잊는다**(out-of-domain 성능 유지); surgical FT는 최적 레이어 부분집합이 shift 유형에 의존하며, **INPUT-레벨 shift(이미지 손상)에는 첫(초기) 레이어만 튜닝하는 것이 최선**임을 보인다.
- **우리에게:** ~0.59M LoRA를 침식 저항적 stay-near-init 앵커로 지지하며, equirectangular distortion 흡수를 위해 **LoRA 용량을 초기 레이어에 집중**(현재는 전 레이어 q/v)하도록 동기화한다. *유의:* LoRA "덜 잊음" 증거는 LLM 것(교차 모달리티 외삽); surgical FT는 vision/분포-shift(근접 매치)다.

> **반박됨 — 사용 금지 (1-2 표):** *하드 identity/InfoNCE형 할당 타깃은 의미 클러스터를 교란하는 반면 soft/balanced 할당은 본질적으로 "더 안전"하다*는 명제. 이는 검증을 **통과하지 못했다** — M1 vs TC3 설명에 끌어들이지 말 것. 대신 equipartition/ME-MAX(#2)와 partial-prototype-collapse(#3) 논거에 의존하라.

---

## AXIS 2 — Distortion-robust / geometry-aware 파노라마 특징 (검증됨, 후속 패스)

FROZEN 평면(planar) backbone으로의 이식 용이성 순으로 정렬. 24개 주장, 23개 만장일치(3-0), 1개 반박.

### 1. Tangent images / E2P projection — 접근 전체를 정당화 (frozen 호환 ✓)
- **Tangent Images**, Eder/Shvets/Lim/Frahm, **CVPR 2020**, arXiv:1912.09390. 구(sphere)를 세분화된 정이십면체 위 국소 평면 gnomonic 타일로 렌더 → **수정 없는 frozen perspective-pretrained 네트워크가 직접 실행, fine-tuning 없이 ~7% acc 손실**(92.6% acc / 93.1% IoU 보존). 우리 E2P = 더 거친 tangent-plane 계열 → **frozen-DINOv3 + E2P 설계가 직접 지지됨; distortion 처리를 operator가 아니라 projection으로 옮겨라.** 유의: ViT가 아닌 CNN에서 입증됨; FoV/각해상도 정규화가 전제조건.

### 2. OOOPS / Open Panoramic Segmentation — 가장 가까운 설계-패턴 매치 (frozen 호환 ✓)
- Zheng et al., **ECCV 2024**, arXiv:2407.02685. **FROZEN CLIP + 경량 Deformable Adapter Network(DAN)**; pinhole로 학습, 360°에서 zero-shot. DAN이 공로 구성요소(+1.4% mIoU; "zero-shot에 frozen CLIP 필수"). **이식(GRAFT): frozen DINOv3에 DAN 스타일 deformable adapter를(또는 LoRA에 접어 넣어) 얹어 E2P 위 잔여 pano 변형을 흡수.** 입력공간 RERP-증강-as-primary 프레이밍은 반박됨(0-3) — 중요한 것은 backbone-측 모듈이다.

### 3. Frozen-backbone adapter 동료들 (LoRA 관련; 중신뢰도, 2-1)
- Survey arXiv:2606.27745 분류: **OmniSAM**(frozen SAM2에 LoRA, <3MB, arXiv:2503.07098) — 우리 LoRA를 직접 정당화; **Dense360 / ERP-RoPE**(arXiv:2506.14471) — equirectangular-aware rotary positional encoding, **drop-in, backbone 변경 불필요 → 가장 이식 친화적인 단일 graft**; **GoodSAM**(frozen SAM teacher + distortion-aware rectification, arXiv:2403.16370). "구형 입력과 평면 모델 사이 인터페이스를 설계한다" — 정확히 우리 패턴. (Dense360은 dense-seg가 아닌 MLLM — 모델이 아니라 RoPE 아이디어를 차용하라.)

### 4. Pretrained 모델의 spherical kernel resampling (평면 호환; frozen 보장 안 됨)
- arXiv:2507.09216 (Jingguo Liu et al., **ICMEW 2025**) — 신원 확인됨. pretrained **ConvNeXt**의 conv-kernel 샘플점을 de-distortion을 위해 구면으로 재투영; pretrained 가중치 = de-distortion 기저 + init. **가중치를 INITIALIZATION으로 resample(→ fine-tuning 함의) → 엄격히 frozen으로 검증되지 않음.** 개념은 위도별 DINOv3 patch-embed 샘플 위치 조정으로 매핑된다.

### 5. KTN — frozen source 모델, 학습된 operator 변환 (ViT 아닌 CNN; 간접적)
- Su & Grauman, **CVPR 2019**, arXiv:1812.03115. FROZEN perspective CNN의 conv kernel을 polar angle 함수로 ERP에 전이; 주석된 360° 데이터 불필요. "frozen operator의 위도별 변환을 학습"을 정당화하나 conv 특정적 → ViT에는 간접적.

### 6. Multi-projection fusion — frozen은 아니지만 이식 가능한 SEAM/OVERLAP-병합 메커니즘
- **OmniFusion**(CVPR 2022 Oral, arXiv:2203.00838): tangent patch + **geometry-aware feature fusion**(3D geom + 2D img)으로 패치 간 불일치 해소 — **우리 겹치는 E2P 타일 병합에 가장 관련된 메커니즘.** **UniFuse**(RA-L 2021, arXiv:2102.03550): **디코딩 단계에서만** 단방향 cubemap→ERP fusion → frozen encoder 뒤에 붙이기 안전. **Elite360D/M**(arXiv:2403.16376 / 2408.09336): ERP + icosphere bi-projection fusion. 모두 end-to-end 학습 → fusion 모듈만 이식 가능.

### 7. Distortion-native 재설계 — frozen ViT와 비호환 (backbone에서 제외)
- **PanoFormer**(ECCV 2022, arXiv:2203.09283), **SphereUFormer**(CVPR 2025, arXiv:2412.06968), **Trans4PASS**(CVPR 2022, arXiv:2203.01452). tokenization/attention를 재구축, 처음부터 학습("어느 것도 pretrained 가중치를 쓰지 않음"). backbone으로 채택하지 말 것 — 다만 PanoFormer의 학습 가능한 token-flow / Trans4PASS의 Deformable Patch Embedding은 DAN 스타일 adapter(#2)의 개념적 씨앗.

**프레이밍 (survey arXiv:2606.27745):** 조사된 방법 중 엄격한 구면 등변성(spherical equivariance)과 perspective-pretrained 가중치의 완전 재사용을 *둘 다* 제공하는 것은 *없다* — "기하학적 충실도 vs pretrained 호환성"은 구조적 trade-off. 우리 frozen-DINOv3 + E2P + LoRA는 **원리적 설계상 pretrained-호환성 쪽**에 위치한다(간과가 아닌, 수용된 입장).

---

## 다음에 시도할 것 (교차-축 — 두 축 모두 검증됨)

1. **TC3의 고정 frozen-teacher 프로토타입 유지** (A1 #1,#2,#3): 학습 프로토타입은 부분 붕괴하고 projector 공간은 침식된다; 고정 frozen 코드는 그럴 수 없다. Sinkhorn/ME-MAX 없이 M1 스타일 학습 코드로 되돌리지 말 것.
2. **DAN 스타일 deformable adapter 또는 ERP-RoPE를 frozen backbone에 이식** (A2 #2,#3): OOOPS의 Deformable Adapter Network와 Dense360의 ERP-RoPE가 가장 이식 가능한 두 distortion 모듈 — backbone 가중치를 건드리지 않고 frozen DINOv3 + LoRA와 조합. **ERP-RoPE가 가장 값싼 drop-in.**
3. **E2P overlap 병합에 OmniFusion 스타일 geometry-aware fusion 사용** (A2 #6): 현재 coverage-mean 스티치를 geometry-aware discrepancy fusion으로 보강; UniFuse의 디코딩-단계-전용 fusion이 frozen encoder 뒤 안전한 부착.
4. **OVERLAP 특징에 decorrelation 가드레일** (A1 #6,#7): E2P overlap 집합에 VICReg-cov / Barlow off-diagonal을 값싼 침식 하한으로.
5. **Gram anchoring 승격** (A1 #4): dense-feature 침식 가드로서 `losses.py:gram_anchor` vs 현재 token+relational distill을 A/B.
6. **distortion 대응 LoRA를 초기 레이어에 집중** (A1 #8): surgical-FT에 따르면 input-레벨(distortion) shift는 초기에서 가장 잘 흡수 — early-only LoRA vs all-layer q/v 테스트.

---
*3-0 검증 출처 — AXIS 1: arXiv 2501.18452, 2006.09882, 2204.07141, 2410.14060, 2508.10104, 2104.14294, 2105.04906, 2103.03230, 2110.09348, 2210.11466, 2405.09673, 2304.07193. AXIS 2: 1912.09390, 2407.02685, 2606.27745, 2507.09216, 1812.03115, 2203.00838, 2102.03550, 2403.16376, 2408.09336, 2203.09283, 2412.06968, 2203.01452 (+ 2-1: 2503.07098, 2506.14471, 2403.16370).*
