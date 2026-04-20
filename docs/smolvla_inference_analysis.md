# SmolVLA 추론 속도 분석 보고서

> **작성일**: 2026-04-08
> **환경**: Windows 11, RTX A4000 (16GB VRAM), RAM 32GB, Python 3.14, lerobot 0.5.0
> **대상 모델**: `lerobot/smolvla_base` (SmolVLM2-500M-Video-Instruct 기반)
> **측정 결과**: 추론 속도 ~1.7Hz (목표 30Hz 대비 약 1/18 수준)

---

## 1. 개요

SO-ARM101 Follower Arm에 SmolVLA base 모델을 얹어 `lerobot-record`로 실행했을 때, 추론 속도가 1.7Hz로 측정되어 실시간 로봇 제어가 불가능했다. 본 보고서는 이 성능 저하의 근본 원인을 코드 레벨에서 분석하고, 실용적인 해결책을 제시한다.

### 테스트에 사용된 명령어

```powershell
lerobot-record `
  --robot.type=so101_follower `
  --robot.port=COM6 `
  --robot.id=my_follower_arm_1 `
  --robot.cameras="{ camera1: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, backend: 700}, camera2: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30, backend: 700}, camera3: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, backend: 700}}" `
  --dataset.single_task="Pick up the cube and place it in the box" `
  --dataset.repo_id=taehunkim/eval_smolvla_base_test `
  --dataset.episode_time_s=50 `
  --dataset.num_episodes=1 `
  --policy.path=lerobot/smolvla_base `
  --policy.use_amp=true `
  --policy.compile_model=true `
  --dataset.fps=30
```

---

## 2. SmolVLA 모델 아키텍처

### 2.1 전체 구조

```
SmolVLAPolicy
├── VLAFlowMatching (모델 본체)
│   ├── SmolVLMWithExpertModel
│   │   ├── VLM (SmolVLM2-500M-Video-Instruct)
│   │   │   ├── Vision Encoder (SigLIP) ← frozen
│   │   │   ├── Connector (modality projection + resampling)
│   │   │   └── Text Model (16 layers, hidden_size=1024)
│   │   └── Action Expert (16 layers, hidden_size=768)
│   │       └── Cross-Attention + Self-Attention 교차 배치
│   ├── state_proj: Linear(32 → 1024)
│   ├── action_in_proj: Linear(32 → 768)
│   ├── action_out_proj: Linear(768 → 32)
│   └── action_time_mlp: 768 → 768 (2-layer MLP)
└── Action Queue (maxlen=50)
```

### 2.2 핵심 설정값 (configuration_smolvla.py)

| 설정 | 값 | 설명 |
|------|-----|------|
| `vlm_model_name` | SmolVLM2-500M-Video-Instruct | VLM 백본 (약 500M 파라미터) |
| `num_vlm_layers` | 16 | VLM 텍스트 모델 레이어 수 |
| `expert_width_multiplier` | 0.75 | Expert hidden_size = VLM의 75% |
| `num_steps` | **10** | Flow Matching 디노이징 반복 횟수 |
| `chunk_size` | 50 | 한 번 추론에 생성하는 액션 시퀀스 길이 |
| `n_action_steps` | 50 | 큐에서 사용할 액션 스텝 수 |
| `resize_imgs_with_padding` | (512, 512) | 이미지 전처리 목표 해상도 |
| `tokenizer_max_length` | 48 | 언어 토큰 최대 길이 |
| `use_cache` | true | KV 캐시 사용 여부 |
| `compile_model` | false (기본) | torch.compile 사용 여부 |

### 2.3 추정 파라미터 수 및 메모리

| 컴포넌트 | 파라미터 수 | bfloat16 메모리 |
|----------|------------|----------------|
| Vision Encoder (SigLIP) | ~150M | ~300MB |
| Text Model (16 layers) | ~350M | ~700MB |
| Action Expert (16 layers, 0.75x) | ~80M | ~160MB |
| 투영 레이어 | ~10M | ~20MB |
| **합계** | **~590M** | **~1.2GB** |

런타임 추가 메모리 (KV 캐시, activation 등): ~500MB-1GB
**총 추정 VRAM 사용**: ~2-3GB (16GB RTX A4000의 15-19%)

---

## 3. 추론 파이프라인 상세 흐름

### 3.1 전체 실행 경로

```
lerobot-record 메인 루프 (lerobot_record.py:357)
│
├─ robot.get_observation()                    # 카메라 3대 + 모터 상태 읽기
│  ├─ bus.sync_read("Present_Position")       # 시리얼 통신 ~2-5ms
│  └─ cam.read_latest() × 3                  # 백그라운드 스레드에서 최신 프레임 ~3-9ms
│
├─ predict_action() (control_utils.py:68)     # 동기 블로킹 호출
│  ├─ torch.inference_mode() + torch.autocast(cuda)
│  ├─ prepare_observation_for_inference()     # numpy→tensor, 배치 추가
│  ├─ preprocessor(observation)               # 정규화 파이프라인
│  └─ policy.select_action(observation)       # ★ 핵심 추론
│     │
│     ├─ [큐가 비어있으면] _get_action_chunk()  # ★ 무거운 추론 (588ms)
│     │  ├─ prepare_images()                  # 640×480→512×512 리사이즈+패딩
│     │  ├─ prepare_state()                   # 관절값→32차원 패딩
│     │  └─ model.sample_actions()            # VLA 순전파
│     │     ├─ embed_prefix()                 # SigLIP + 언어 + 상태 임베딩
│     │     ├─ KV 캐시 초기 계산              # 1회 전체 VLM forward
│     │     └─ Denoising Loop × 10           # ★★★ 핵심 병목
│     │        └─ denoise_step() × 10
│     │           ├─ embed_suffix()
│     │           └─ vlm_with_expert.forward()
│     │              └─ 16 layers × eager_attention (float32)
│     │
│     └─ [큐에 액션 있으면] queue.popleft()    # 즉시 반환 (<0.1ms)
│
├─ robot.send_action()                        # 시리얼 통신 ~3-5ms
└─ precise_sleep()                            # FPS 맞추기 위한 대기
```

### 3.2 Denoising Loop 상세 (핵심 병목)

```python
# modeling_smolvla.py:833-870
num_steps = 10              # 설정값
dt = -1.0 / num_steps      # = -0.1

x_t = noise                 # 랜덤 노이즈에서 시작

for step in range(num_steps):           # ★ 10회 반복
    time = 1.0 + step * dt              # [1.0, 0.9, 0.8, ..., 0.1]
    v_t = denoise_step(x_t, timestep)   # Expert 모델 전체 forward pass
    x_t = x_t + dt * v_t               # Euler 적분으로 노이즈 제거
```

각 `denoise_step()`은:
1. 액션+타임스텝 임베딩 생성 (`embed_suffix`)
2. Attention 마스크 구성 (prefix_pad × suffix_len 2D 마스크)
3. **VLM+Expert 전체 forward pass** (16 layers, KV 캐시 활용)
4. 액션 출력 투영

### 3.3 Eager Attention 구현 (성능 저하 핵심)

```python
# smolvlm_with_expert.py:504-549
def eager_attention_forward(self, ...):
    # 1. Group Query Attention: key/value를 expand로 복제 (메모리 낭비)
    key_states = key_states[:, :, :, None, :].expand(...)
    value_states = value_states[:, :, :, None, :].expand(...)

    # 2. float32로 강제 업캐스팅 (Tensor Core 활용 불가)
    query_states = query_states.to(dtype=torch.float32)
    key_states = key_states.to(dtype=torch.float32)

    # 3. 수동 matmul (최적화된 CUDA 커널 미사용)
    att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
    att_weights *= head_dim**-0.5

    # 4. 마스킹 + softmax
    masked_att_weights = torch.where(attention_mask, att_weights, big_neg)
    probs = nn.functional.softmax(masked_att_weights, dim=-1)

    att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))
```

**문제점**:
- `torch.nn.functional.scaled_dot_product_attention` (Flash Attention) 미사용
- bfloat16 → float32 업캐스팅으로 **Tensor Core 비활성화**
- `.expand()` + `.reshape()`로 불필요한 메모리 할당
- 모든 연산이 개별 CUDA 커널로 실행 (fusion 없음)

---

## 4. 원인 분석 (우선순위 순)

### 4.1 원인별 영향도

| 순위 | 원인 | 시간 기여도 | 파일 위치 |
|------|------|-----------|----------|
| **1** | Flow Matching 10회 반복 | **~85%** (~500ms) | `modeling_smolvla.py:833` |
| **2** | Eager Attention float32 | ~5-10% (~40ms) | `smolvlm_with_expert.py:504` |
| **3** | 3대 카메라 SigLIP 인코딩 | ~5-7% (~40ms) | `smolvlm_with_expert.py:179` |
| **4** | chunk_size=50 긴 시퀀스 | ~3% (~15ms) | `configuration_smolvla.py:32` |
| **5** | RTX A4000 연산 한계 | 전체 영향 | (하드웨어) |
| **6** | torch.compile 초기 오버헤드 | 첫 호출만 | `modeling_smolvla.py:598` |

### 4.2 시간 분해 추정 (RTX A4000, 단일 추론 호출)

```
총 추론 시간: ~588ms (= 1/1.7Hz)
│
├── [전처리] 이미지 리사이즈+정규화                   ~10ms   (1.7%)
├── [비전] SigLIP 인코딩 × 3장                       ~40ms   (6.8%)
├── [언어] 토큰 임베딩                                ~5ms    (0.9%)
├── [상태] 관절값 투영                                ~2ms    (0.3%)
├── [KV캐시] Prefix KV 캐시 초기 계산                 ~30ms   (5.1%)
│
├── [디노이징] Flow Matching Loop × 10               ~498ms  (84.7%)
│   └── 각 회당 ~50ms:
│       ├── embed_suffix (액션+시간 임베딩)            ~5ms
│       ├── Attention 마스크 구성                      ~3ms
│       └── Expert 16-layer forward                  ~42ms
│           ├── Cross-Attention (float32 matmul)     ~25ms   ← Tensor Core 미활용
│           └── Feed-Forward MLP                     ~17ms
│
└── [후처리] 출력 투영 + CPU 전송                      ~3ms    (0.5%)
```

### 4.3 액션 청킹에 의한 완화와 한계

SmolVLA는 한 번의 추론으로 **50개 액션**을 생성하고 큐에 저장한다:

```
시간축 →
[추론 588ms][큐 소비 49프레임 × 33ms = 1633ms][추론 588ms][큐 소비...]
     ↑ 로봇 정지                                      ↑ 로봇 정지
```

- 30fps에서 50프레임 = ~1.67초 주기로 추론 실행
- **추론 중 588ms 동안 메인 루프 블로킹** → 로봇 정지
- 588ms / 33.3ms = **약 17.6프레임 분량의 끊김**
- 사용자 체감: 매 ~1.7초마다 약 0.6초간 로봇이 멈추는 현상

---

## 5. 해결책

### 5.1 즉시 적용 가능 (CLI 인자 변경만으로)

#### A. num_steps 감소 (10 → 4)

**가장 효과적인 단일 변경.** 디노이징 반복을 10→4로 줄이면 핵심 병목(85%)이 직접 감소.

```
예상 시간: 90ms(비디노이징) + 4 × 50ms = 290ms → ~3.4Hz
개선률: ~2배
```

```powershell
--policy.num_steps=4
```

> 품질 트레이드오프: 디노이징 스텝이 줄면 생성된 액션의 정밀도가 다소 감소할 수 있으나, 실시간 제어에서는 반응 속도가 더 중요.

#### B. 카메라 수 감소 (3대 → 1대)

3대 카메라 인코딩 시간 ~40ms를 ~15ms로 축소.

```powershell
--robot.cameras="{ camera1: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, backend: 700}}"
```

> 주의: SmolVLA base 모델의 학습 시 카메라 구성과 일치해야 최적 성능. 모델 카드의 `info.json`에서 기대하는 카메라 키를 확인할 것.

#### C. FPS 감소 (30 → 10)

루프 주기를 100ms로 늘려 추론 시간 여유 확보.

```powershell
--dataset.fps=10
```

> 액션 청킹(n_action_steps=50)과 결합하면 10fps × 50스텝 = 5초간 추론 불필요.

#### D. compile_mode 변경

`max-autotune`은 최적 커널을 탐색하느라 첫 컴파일이 매우 느림.

```powershell
--policy.compile_mode=reduce-overhead
```

> 또는 Windows에서 torch.compile 안정성이 낮으므로 아예 비활성화:
> `--policy.compile_model=false`

### 5.2 복합 최적화 명령어 (권장)

```powershell
lerobot-record `
  --robot.type=so101_follower `
  --robot.port=COM6 `
  --robot.id=my_follower_arm_1 `
  --robot.cameras="{ camera1: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, backend: 700}}" `
  --dataset.single_task="Pick up the cube and place it in the box" `
  --dataset.repo_id=taehunkim/eval_smolvla_optimized_test `
  --dataset.episode_time_s=50 `
  --dataset.num_episodes=1 `
  --policy.path=lerobot/smolvla_base `
  --policy.use_amp=true `
  --policy.compile_model=false `
  --policy.num_steps=4 `
  --dataset.fps=10
```

**예상 개선**:
- num_steps 10→4: ~2배
- 카메라 3→1대: ~15-20%
- fps 30→10: 루프 여유 확보
- compile 비활성화: 초기 오버헤드 제거
- **종합 예상: 1.7Hz → 4-5Hz (추론), 10fps 루프에서 안정 동작 가능**

### 5.3 코드 수정이 필요한 최적화

#### A. SDPA (Scaled Dot-Product Attention) 적용

`smolvlm_with_expert.py:504-549`의 `eager_attention_forward`를 PyTorch 내장 SDPA로 교체.

```python
# 변경 전 (현재)
query_states = query_states.to(dtype=torch.float32)
key_states = key_states.to(dtype=torch.float32)
att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
...

# 변경 후
att_output = torch.nn.functional.scaled_dot_product_attention(
    query_states.transpose(1, 2),
    key_states.transpose(1, 2),
    value_states.transpose(1, 2),
    attn_mask=attention_mask,
    enable_gqa=True,
)
```

**효과**: Flash Attention 커널 자동 적용, float32 업캐스트 불필요, ~20-30% 향상

#### B. INT8 양자화

```python
from torch.ao.quantization import quantize_dynamic
model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```

**효과**: 메모리 절반 + 추론 ~30-50% 향상 (정밀도 약간 감소)

#### C. 비동기 추론 파이프라인

현재 단일 스레드 동기 실행을 별도 스레드로 분리:

```python
# 개념 코드
inference_thread = Thread(target=inference_loop, args=(policy, obs_queue, action_queue))

# 메인 루프: 카메라+모터만 처리 (추론 대기 없음)
while True:
    obs = robot.get_observation()
    obs_queue.put(obs)
    action = action_queue.get_nowait()  # 비블로킹
    robot.send_action(action)
```

**효과**: 추론 블로킹에 의한 끊김 완전 제거. 단, 관찰-액션 간 1프레임 지연 발생.

### 5.4 근본적 대안 비교

| 방법 | 추론 속도 | 학습 필요 | 적합성 |
|------|----------|----------|--------|
| **SmolVLA base (현재)** | 1.7Hz | 없음 | 제로샷이지만 속도 부족 |
| **SmolVLA 최적화 (5.2)** | ~4-5Hz | 없음 | 10fps 기준 사용 가능 |
| **SmolVLA 파인튜닝** | ~4-5Hz | 20k steps (~2.5h) | 최적 성능 기대 |
| **ACT 정책** | **30Hz+** | 50k steps (~45min) | 경량, 실시간 적합 |
| **Diffusion Policy** | ~10Hz | 30k steps | 중간 복잡도 |

---

## 6. 별도 학습 없이 VLA 구동 가능성

### 6.1 SmolVLA base의 제로샷 전이

- `lerobot/smolvla_base`는 특정 로봇 구성(주로 Aloha-style)으로 학습됨
- SO-ARM101 6축과 학습 로봇의 액션 공간 불일치 가능성
- 언어 조건부 제어("Pick up the cube")는 구조적으로 작동
- 공식 예제(`examples/tutorial/smolvla/using_smolvla_example.py`)에서 SO100Follower 직접 사용을 지원

### 6.2 제로샷 테스트를 위한 권장 절차

1. **경량 설정으로 먼저 테스트** (카메라 1대, 10fps, num_steps=4)
2. 로봇이 의미 있는 동작을 하는지 관찰
3. 동작이 불규칙하면 → 파인튜닝 필요
4. 동작이 안정적이면 → num_steps/fps를 점진적으로 올려 품질 개선

### 6.3 추천 로드맵

```
Phase 1: 추론 최적화 (즉시)
  └─ num_steps=4, 카메라 1대, fps=10으로 테스트

Phase 2: 제로샷 평가 (1일)
  └─ 최적화 설정으로 SmolVLA base 동작 품질 평가
  └─ 태스크별 성공률 측정

Phase 3a: [동작 품질 양호] 점진적 설정 개선
  └─ fps 증가, 카메라 추가, num_steps 조정

Phase 3b: [동작 품질 불량] 파인튜닝
  └─ 텔레오퍼레이션으로 50 에피소드 수집
  └─ SmolVLA 파인튜닝 (20k steps, ~2.5시간)
  └─ 또는 ACT로 전환 (50k steps, ~45분)
```

---

## 7. 부록: 핵심 소스 코드 위치

| 파일 경로 | 핵심 함수/클래스 | 역할 |
|----------|----------------|------|
| `lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py` | `SmolVLAConfig` | num_steps, chunk_size 등 설정 |
| `lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py:833` | `sample_actions()` | 디노이징 루프 (핵심 병목) |
| `lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py:872` | `denoise_step()` | 단일 디노이징 스텝 |
| `lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py:322` | `select_action()` | 액션 큐 관리 |
| `lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py:504` | `eager_attention_forward()` | float32 수동 attention |
| `lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py:179` | `embed_image()` | SigLIP 비전 인코딩 |
| `lerobot/src/lerobot/scripts/lerobot_record.py:357` | `record_loop()` | 실시간 제어 메인 루프 |
| `lerobot/src/lerobot/utils/control_utils.py:68` | `predict_action()` | 추론 호출 래퍼 |

---

## 8. 결론

SmolVLA의 1.7Hz 추론 속도는 **Flow Matching 디노이징 10회 반복**이 전체 시간의 85%를 차지하는 것이 핵심 원인이다. 이 반복마다 16-layer Expert 모델의 전체 forward pass가 실행되며, float32로 업캐스팅된 eager attention이 GPU Tensor Core 활용을 차단한다.

**즉시 적용 가능한 가장 효과적인 해결책**은 `--policy.num_steps=4`로 디노이징 스텝을 줄이는 것이며, 카메라 축소와 fps 감소를 병행하면 10fps 기준으로 안정적인 실시간 구동이 가능할 것으로 예상된다.

장기적으로는 ACT 정책(경량, 30Hz+ 추론)이나 SmolVLA 파인튜닝이 SO-ARM101 환경에 더 적합한 선택이다.
