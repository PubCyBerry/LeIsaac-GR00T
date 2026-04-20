# VLA 모델 Zero-Shot 로봇 제어 가능성 조사

> 작성일: 2026-04-08  
> 대상 하드웨어: SO-ARM101 6축 로봇 팔  
> 프레임워크: Hugging Face LeRobot

---

## Executive Summary

**현재 공개된 어떤 VLA 모델도 SO-ARM101에서 Fine-Tuning 없이(Zero-Shot) 신뢰성 있게 동작하지 않는다.**

- 대부분의 VLA 모델은 학습에 사용된 로봇 플랫폼 내에서만 일반화를 보여주며, 새로운 embodiment로의 Zero-Shot 전이는 실패하거나 극히 낮은 성공률을 기록한다.
- SmolVLA가 SO-100/101 데이터로 학습되어 가장 유망하지만, normalization 통계값 불일치로 인해 Zero-Shot 추론이 **기술적으로 불가능**하다.
- 가장 진보된 Zero-Shot cross-embodiment 결과를 보여주는 RDT2도 단순 태스크에서 25~52% 성공률에 그친다.
- 실용적 접근: **50회 이상의 teleoperation 데모를 수집하여 Fine-Tuning하는 것이 유일하게 신뢰할 수 있는 방법**이다.

---

## 모델별 상세 분석

### 1. SmolVLA (450M) - Hugging Face

| 항목 | 내용 |
|------|------|
| **HF Repo** | [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) (31.9K+ 다운로드) |
| **파라미터** | 450M |
| **아키텍처** | SmolVLM 백본 + PixelShuffle 시각 토큰 압축(64개/프레임) + 교차/자기 어텐션 액션 전문가 |
| **학습 데이터** | LeRobot 커뮤니티 데이터셋 (SO-100/SO-101 포함) |
| **라이선스** | Apache 2.0 |
| **LeRobot 호환** | 네이티브 |

**Zero-Shot 결과:**
- 공식 문서: "SmolVLA is a base model, so fine-tuning on your own data is required."
- 사전학습 + Fine-Tuning 시 SO-100에서 78.3% 성공률 달성 (Zero-Shot 아님)
- Fine-Tuning 없이 배포 시: **로봇이 거의 움직이지 않음** (중립 위치 근처에서 미세 떨림)

**기술적 문제:**
- **GitHub Issue [#2374](https://github.com/huggingface/lerobot/issues/2374)**: 사전학습 모델의 액션 출력 범위가 [-5, 5]이나 로봇은 [-100, 100]을 기대 → normalization 통계값 불일치
- **GitHub Issue [#1239](https://github.com/huggingface/lerobot/issues/1239)**: "smolvla model not working properly"
- **GitHub Issue [#1270](https://github.com/huggingface/lerobot/issues/1270)**: "smolvla not work!"
- LeRobot이 사전학습 SmolVLA 모델에 대한 Zero-Shot 추론을 네이티브로 지원하지 않음 (학습 시 사용된 normalization 통계에 접근 필요)

**SO-ARM101 적합성:** 모든 모델 중 **가장 높음**. SO-100/101 데이터로 학습됨. 단, 반드시 Fine-Tuning 필요 (50회+ 데모 권장).

---

### 2. Pi0 / Pi0-FAST / Pi0.5 - Physical Intelligence

| 항목 | 내용 |
|------|------|
| **HF Repo** | [`lerobot/pi0_base`](https://huggingface.co/lerobot/pi0_base), [`lerobot/pi0fast-base`](https://huggingface.co/lerobot/pi0fast-base), [`lerobot/pi05_base`](https://huggingface.co/lerobot/pi05_base) |
| **파라미터** | ~3B |
| **아키텍처** | PaliGemma VLM 백본 + Flow Matching 기반 액션 전문가. Pi0-FAST는 DCT 압축으로 이산 토큰 사용 |
| **학습 데이터** | 7개 로봇 플랫폼, 68개 고유 태스크 (ALOHA, DROID/Franka 등) |
| **라이선스** | Gemma (모델), Apache 2.0 (코드) |
| **LeRobot 호환** | 네이티브 |

**Zero-Shot 결과 (Penn PAL Lab 독립 평가):**
- 평균 태스크 진행률: ~42.3% (성공률이 아닌 진행률)
- "캔을 보라색 상자에 넣기": 16.7% 성공률
- 단순 태스크: 20~50% 성공률
- 프롬프트 표현에 민감, 세밀한 조작과 다단계 태스크에서 실패
- 메모리가 없어 상태 추적 불가

**기술적 문제:**
- **GitHub Issue [#694](https://github.com/huggingface/lerobot/issues/694), [#699](https://github.com/huggingface/lerobot/issues/699)**: 사전학습 Pi0 정책에 normalization 통계값이 누락되어 Zero-Shot 추론이 기술적으로 불가
- ALOHA 체크포인트는 ALOHA 플랫폼에서만 동작하며 로봇 셋업에 매우 민감

**SO-ARM101 적합성:** 낮음~중간. 주로 듀얼암(ALOHA)과 Franka에서 학습됨. SO-ARM101은 학습 분포에 포함되지 않음.

---

### 3. X-VLA (0.9B) - ICLR 2026

| 항목 | 내용 |
|------|------|
| **HF Repo** | [`lerobot/xvla-base`](https://huggingface.co/lerobot/xvla-base) (9.3K 다운로드) |
| **파라미터** | 0.9B |
| **아키텍처** | Florence-2-Large VLM + "소프트 프롬프트" 메커니즘 (embodiment별 학습 가능 임베딩) |
| **학습 데이터** | 290K 에피소드 (DROID, Robomind, Agibot) - 7개 플랫폼, 5종 로봇 팔 |
| **라이선스** | Apache 2.0 |
| **LeRobot 호환** | 네이티브 |

**Zero-Shot 결과:**
- AgiBot World Challenge 1위 (IROS 2025)
- 6개 시뮬레이션 벤치마크 + 3개 실제 로봇에서 SOTA
- 단, 소프트 프롬프트 접근법은 여전히 대상 도메인 데이터로 embodiment별 프롬프트 학습 필요 → 엄밀한 Zero-Shot이 아님

**기술적 문제:**
- **GitHub Issue [#2741](https://github.com/huggingface/lerobot/issues/2741)**: xvla-base가 Phase II 도메인 적응을 거쳤는지 불분명
- **GitHub Issue [#2942](https://github.com/huggingface/lerobot/issues/2942)**: "stuttering motion (불연속 액션 출력)" 보고

**SO-ARM101 적합성:** 중간. Cross-embodiment 전이를 위해 설계되었으나 SO-ARM101 데이터로 Phase II 프롬프트 학습 필요.

---

### 4. OpenVLA (7B) - Stanford/Berkeley

| 항목 | 내용 |
|------|------|
| **HF Repo** | [`openvla/openvla-7b`](https://huggingface.co/openvla/openvla-7b) (1.18M+ 다운로드) |
| **파라미터** | 7B |
| **아키텍처** | Prismatic VLM (DINOv2 + SigLIP) + Llama-2 LLM. 이산화된 액션을 언어 토큰으로 출력 |
| **학습 데이터** | 970K 궤적 (Open X-Embodiment) |
| **라이선스** | MIT |
| **LeRobot 호환** | 아니오 |

**Zero-Shot 결과:**
- **모델 카드에서 명시적 선언: "OpenVLA models do NOT zero-shot generalize to new (unseen) robot embodiments."**
- 학습 분포 내 로봇(BridgeV2/WidowX)에서는 RT-2-X 대비 16.5% 향상
- 학습 분포 내 WidowX에서도 실제 Zero-Shot 성공률은 ~10%에 불과 (장난감 가지 집기 태스크, GitHub Issue [#312](https://github.com/openvla/openvla/issues/312))
- Fine-Tuning 후에도 성능이 오히려 하락하는 사례 보고

**SO-ARM101 적합성:** 매우 낮음. 7-DoF 엔드이펙터 델타를 출력하므로 관절 위치 제어를 사용하는 SO-ARM101과 호환 불가. 16GB+ VRAM 필요.

---

### 5. Octo (27M~93M) - UC Berkeley

| 항목 | 내용 |
|------|------|
| **HF Repo** | [`rail-berkeley/octo-base-1.5`](https://huggingface.co/rail-berkeley/octo-base-1.5) (93M), [`rail-berkeley/octo-small-1.5`](https://huggingface.co/rail-berkeley/octo-small-1.5) (27M) |
| **아키텍처** | Transformer 기반 Diffusion Policy |
| **학습 데이터** | 800K 궤적 (Open X-Embodiment 25개 데이터셋) |
| **라이선스** | MIT |
| **LeRobot 호환** | 아니오 (JAX 기반) |

**Zero-Shot 결과:**
- 학습 분포 내 태스크에서 RT-1-X 대비 29% 향상
- 새로운 장면에서 "약간 저하", 새로운 행동(뒤집기, 정밀 삽입)에서 "높은 저하"
- 공식적으로 "효과적인 Fine-Tuning을 위한 다재다능한 정책 초기화"로 설계됨
- **GitHub Issue [#29](https://github.com/octo-models/octo/issues/29)**: "Fine-Tuning 후 랜덤 행동" 보고

**SO-ARM101 적합성:** 낮음. SO-ARM101이 학습 데이터에 포함되지 않음. JAX 의존성으로 LeRobot과 직접 통합 불가. 현재 신규 모델들에 의해 대체됨.

---

### 6. RDT-1B / RDT2 - Tsinghua University

| 항목 | 내용 |
|------|------|
| **HF Repo** | [`robotics-diffusion-transformer/rdt-1b`](https://huggingface.co/robotics-diffusion-transformer/rdt-1b) |
| **파라미터** | 1.2B |
| **아키텍처** | Diffusion Transformer. 통합 액션 공간 (엔드이펙터, 관절, 위치, 속도, 차륜 이동) |
| **학습 데이터** | RDT-1B: 46개 데이터셋, 1M+ 에피소드. RDT2: UMI 데이터 확장 |
| **라이선스** | MIT |
| **LeRobot 호환** | 아니오 |

**Zero-Shot 결과:**
- **RDT2가 현재 가장 유망한 Zero-Shot cross-embodiment 결과를 보유**
- 4U 프로토콜(Unseen embodiment/scene/object/instruction)에서:
  - 단순 태스크(집기, 놓기, 누르기, 닦기): **25~52% 성공률**
  - 처음 보는 물체에서 베이스라인 대비 4배 향상
- 논문 저자 인정: "성공률이 높지는 않지만 이 결과의 의의는 심대하다"

**SO-ARM101 적합성:** 낮음~중간. 가장 유망한 Zero-Shot 결과이나 실용적 수준에 미달. LeRobot 미통합. 멀티 GPU Fine-Tuning 필요.

---

### 7. NVIDIA GR00T N1.5 (3B)

| 항목 | 내용 |
|------|------|
| **HF Repo** | [`nvidia/GR00T-N1.5-3B`](https://huggingface.co/nvidia/GR00T-N1.5-3B) |
| **파라미터** | 3B |
| **아키텍처** | 이중 시스템 (System 2: VLM 추론/계획 + System 1: 빠른 액션 모델). EmbodimentTag로 플랫폼 적응 |
| **학습 데이터** | 15TB, 320K+ 궤적 (주로 휴머노이드 + 매니퓰레이션) |
| **라이선스** | NVIDIA 커스텀 |
| **LeRobot 호환** | 네이티브 (N1.5부터) |

**Zero-Shot 결과:**
- 전체 학습 데이터 기준 Pick-and-Place 82% 성공률
- 10% 데이터만 사용 시 42.6%
- SO-ARM101에서의 Zero-Shot 결과는 보고되지 않음

**SO-ARM101 적합성:** 높음 (Fine-Tuning 전제). [SO-101 Fine-Tuning 공식 튜토리얼](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning) 존재. Fine-Tuning에 ~25GB VRAM 필요.

---

### 8. Google RT-2 / RT-X (비공개)

| 항목 | 내용 |
|------|------|
| **HF Repo** | **비공개** - 모델 가중치 미공개 |
| **파라미터** | 55B (RT-2) |
| **학습 데이터** | Google 내부 로봇 + 인터넷 이미지 데이터 |

**Zero-Shot 결과:**
- 동일 로봇 플랫폼 내에서 RT-1 대비 32% → 62%로 일반화 향상
- **핵심 한계**: "로봇이 웹 사전학습으로부터 새로운 동작 능력을 획득하지는 않음" — 물리적 스킬은 로봇 학습 데이터에서 본 동작으로 제한

**SO-ARM101 적합성:** 없음. 비공개 모델, 외부 하드웨어 배포 불가.

---

### 9. 신흥 접근법

| 모델 | 핵심 아이디어 | 결과 |
|------|-------------|------|
| **DreamZero** (ICLR 2026 Workshop) | 비디오 확산으로 물리 역학 학습 | SOTA VLA 대비 일반화 2배 향상. 30분 플레이 데이터로 embodiment 적응 |
| **VLA-Pilot** (2025.11) | 추론 시 스티어링으로 Fine-Tuning 회피 | 사전학습 VLA 성공률 평균 31% 향상. Fine-Tuning과 유사한 성능 |
| **Sim2Real-VLA** | 합성 데이터만으로 학습, 실세계 Zero-Shot 전이 | 초기 단계이나 실데이터 수집 불필요 가능성 |

---

## 종합 비교 테이블

| 모델 | 파라미터 | LeRobot 통합 | SO-ARM101 Zero-Shot 가능? | 예상 성공률 | Fine-Tuning 후 전망 | 추론 VRAM |
|------|---------|-------------|--------------------------|-----------|-------------------|----------|
| **SmolVLA** | 450M | 네이티브 | 기술적 불가 (normalization) | 0% | 높음 (78.3% 보고) | 8GB |
| **Pi0-FAST** | ~3B | 네이티브 | 통계값 누락, 다른 embodiment | 0~5% | 높음 | 8GB |
| **X-VLA** | 0.9B | 네이티브 | Phase II 필요 | 미확인 | 중간~높음 | 8~16GB |
| **GR00T N1.5** | 3B | 네이티브 | 미확인 | 미확인 | 높음 (튜토리얼 존재) | ~16GB |
| **OpenVLA** | 7B | 미지원 | 명시적 불가 선언 | ~0% | 중간 | 16GB |
| **RDT2** | 1.2B | 미지원 | 제한적 가능 | 25~52% (단순 태스크) | 중간 | ~8GB |
| **Octo** | 27~93M | 미지원 (JAX) | Fine-Tuning용 설계 | 0~5% | 낮음~중간 | <4GB |
| **RT-2** | 55B | 미지원 | 비공개 | N/A | N/A | N/A |

---

## 커뮤니티 실패 사례 모음

### SmolVLA
- [huggingface/lerobot#2374](https://github.com/huggingface/lerobot/issues/2374) — SO-101에서 로봇 동결, 중립 위치 근처 미세 떨림. 원인: 액션 범위 [-5,5] vs 로봇 기대 [-100,100]
- [huggingface/lerobot#1239](https://github.com/huggingface/lerobot/issues/1239) — "smolvla model not working properly"
- [huggingface/lerobot#1270](https://github.com/huggingface/lerobot/issues/1270) — "smolvla not work!"
- [huggingface/lerobot#2915](https://github.com/huggingface/lerobot/issues/2915) — Fine-Tuning에서도 어려움 보고

### Pi0
- [huggingface/lerobot#694](https://github.com/huggingface/lerobot/issues/694) — 사전학습 Pi0에서 normalization 통계값 누락
- [huggingface/lerobot#699](https://github.com/huggingface/lerobot/issues/699) — 동일 문제 중복 보고

### OpenVLA
- [openvla/openvla#312](https://github.com/openvla/openvla/issues/312) — 학습 분포 내 로봇(WidowX)에서도 ~10% 성공률. Fine-Tuning 후 오히려 성능 하락

### X-VLA
- [huggingface/lerobot#2741](https://github.com/huggingface/lerobot/issues/2741) — Phase I/II 구분 혼란
- [huggingface/lerobot#2942](https://github.com/huggingface/lerobot/issues/2942) — 불연속 액션 출력 (stuttering motion)

### Octo
- [octo-models/octo#29](https://github.com/octo-models/octo/issues/29) — Fine-Tuning 후 랜덤 행동

---

## VLA는 정말 일반화하는가? — 비판적 시각

최근 연구들이 현재 VLA 모델의 일반화 능력에 근본적인 의문을 제기하고 있다:

1. **지시문 무시**: 언어 입력이 "충분히 활용되지 않거나 완전히 무시됨". 지시와 무관하게 물체를 향해 손을 뻗고 잡으려는 행동 반복 ([Chen, 2025](https://medium.com/@yananchen1116/does-vla-model-really-have-generalisation-capabilities-or-is-it-just-a-overfit-270ffd7a04db))
2. **과적합 의심**: 학습 데이터의 지배적 행동(pick-and-place)을 암기하여 일반화처럼 보이는 것일 가능성
3. **취약한 일반화**: 카메라 시점 변경, 로봇 초기 자세 변화에서 성능 급격 저하 ([LIBERO-Plus](https://arxiv.org/abs/2510.13626))
4. **분포 이탈 취약성**: 학습 분포에서 약간만 벗어나도 "brittle generalization" 현상

---

## SO-ARM101 프로젝트 권장 사항

### 결론: Fine-Tuning 없는 배포는 현재 불가능

어떤 사전학습 VLA 모델도 SO-ARM101에서 Zero-Shot으로 신뢰성 있게 동작하지 않는다. 이는 이론적 한계가 아니라 **실증적으로 확인된 사실**이다.

### 권장 접근법 (우선순위순)

#### 1순위: SmolVLA Fine-Tuning (권장)
```bash
# 50회+ 텔레오퍼레이션 데모 수집 후
uv run lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=taehunkim/SoArm_pick_and_place \
  --batch_size=64 \
  --steps=20_000 \
  --output_dir=outputs/train/smolvla/pick_and_place \
  --policy.device=cuda \
  --wandb.enable=true
```
- SO-100/101 데이터로 사전학습되어 가장 적합
- 450M 파라미터로 소비자 GPU에서 학습 가능
- 사전학습 효과: 동일 Fine-Tuning 대비 26.6%p 성공률 향상 (51.7% → 78.3%)

#### 2순위: ACT 정책 처음부터 학습 (안정적 대안)
```bash
uv run lerobot-train \
  --dataset.repo_id=taehunkim/SoArm_pick_and_place \
  --policy.type=act \
  --policy.device=cuda \
  --output_dir=outputs/train/act_so101/pick_and_place \
  --steps=50_000
```
- VLA 특유의 normalization 이슈 없음
- 더 예측 가능하고 디버깅 용이
- A4000에서 ~45분 학습

#### 3순위: GR00T N1.5 Fine-Tuning (실험적)
- [공식 SO-101 튜토리얼](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning) 참고
- 25GB VRAM 필요 (A5000 이상 권장)

### 주시할 기술
- **X-VLA**: 소프트 프롬프트 기반 효율적 cross-embodiment 적응
- **RDT2**: 현재 가장 유망한 Zero-Shot cross-embodiment 결과
- **VLA-Pilot**: Fine-Tuning 없이 추론 시 스티어링으로 성능 향상
- **DreamZero**: 비디오 확산 기반 물리 역학 학습 — 30분 플레이 데이터로 적응

---

## 참고 자료

### 논문
- [RT-2: Vision-Language-Action Models](https://robotics-transformer2.github.io/) (Google DeepMind, 2023)
- [Octo: An Open-Source Generalist Robot Policy](https://arxiv.org/abs/2405.12213) (UC Berkeley, 2024)
- [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246) (Stanford/Berkeley, 2024)
- [Pi0: A Vision-Language-Action Flow Model](https://www.pi.website/blog/pi0) (Physical Intelligence, 2024)
- [SmolVLA](https://huggingface.co/blog/smolvla) (Hugging Face, 2025)
- [X-VLA: Cross-Embodiment VLA](https://arxiv.org/abs/2510.10274) (ICLR 2026)
- [RDT-1B / RDT2](https://rdt-robotics.github.io/rdt2/) (Tsinghua University, 2025-2026)
- [GR00T N1](https://arxiv.org/html/2503.14734v1) (NVIDIA, 2025)
- [OpenVLA-OFT](https://openvla-oft.github.io/) (2025)
- [VLA-Pilot](https://arxiv.org/abs/2511.14178) (2025)

### 블로그 및 튜토리얼
- [Hugging Face Pi0 Blog](https://huggingface.co/blog/pi0)
- [GR00T N1.5 SO-101 Fine-Tuning Tutorial](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning)
- [SmolVLA Fine-Tuning Guide (Medium)](https://medium.com/correll-lab/fine-tuning-smolvla-for-new-environments-code-included-af266c56d632)
- [Penn PAL Lab Pi0 독립 평가](https://penn-pal-lab.github.io/Pi0-Experiment-in-the-Wild/)

### 비판적 분석
- [Does VLA Really Have Generalisation Capabilities?](https://medium.com/@yananchen1116/does-vla-model-really-have-generalisation-capabilities-or-is-it-just-a-overfit-270ffd7a04db)
- [LIBERO-Plus: Robustness Analysis](https://arxiv.org/abs/2510.13626)
