# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

SO-ARM 101 6축 로봇 팔에 대한 VLA(Vision-Language-Action) 기반 제어 시스템 개발 워크스페이스.  
Hugging Face LeRobot 프레임워크로 ACT, SmolVLA 를 학습하며, NVIDIA GR00T N1.6 정책은 Isaac-GR00T 서버를 통해 시뮬레이션(LeIsaac) 및 실기 추론에 활용한다.

- **기본 태스크**: `pick_and_place`
- **프로젝트 관리 폴더**: `D:/01_Work/01_Projects/11_VLA_기반_6축_Manipulator_제어/`

## 패키지 관리 (uv)

이 프로젝트는 `uv`를 사용한다. conda나 pip를 직접 사용하지 말 것.

```bash
# 프로젝트 초기화 (최초 1회)
uv init --python 3.11

# 의존성 설치
uv sync

# 스크립트 실행
uv run python <script.py>
uv run lerobot-train <args>
```

## 아키텍처 및 워크플로우

```
[SO-ARM101 리더 암] → 텔레오퍼레이션 → [LeRobot 데이터셋] → HF Hub
                                                ↓
                                          lerobot-train
                                         (ACT / SmolVLA)
                                                ↓
                                       outputs/train/*/
                                       checkpoints/last/
                                                ↓
                                     huggingface-cli upload
                                                ↓
                                      [HF Hub 모델 저장소]
                                                ↓
                                       lerobot-eval (추론)
                                                ↓
                                    [SO-ARM101 팔로워 암 실행]
```

### 정책 비교

| 정책 | 특징 | 권장 steps | A4000 예상 시간 |
|------|------|-----------|----------------|
| **ACT** | 경량, 빠른 학습 | 50,000 | ~45분 |
| **SmolVLA** | VLA, 언어 조건부 | 20,000 | ~2.5시간 |
| **GR00T N1.6** | NVIDIA 사전학습 VLA (3B, bf16), ZMQ 서버로 추론 | — | 추론 전용 (Windows 는 Docker 컨테이너 필수) |

## 출력 디렉토리 구조

```
outputs/train/<job_name>/
├── checkpoints/
│   ├── <step>/
│   └── last/
│       └── pretrained_model/   ← Hub 업로드 대상
└── logs/
```

## 추론 (Inference)

### GR00T N1.6 (Docker 기반)

Windows 호스트에서는 `flash-attn`, `deepspeed`, `torchcodec` 휠 빌드가 불가능하므로 추론 서버만 Linux 컨테이너에 격리한다. 클라이언트(Isaac Sim / LeIsaac) 는 호스트 Windows 에서 그대로 실행되며 ZMQ REQ-REP over TCP 5555 로 컨테이너에 접속한다.

- **이미지 빌드 (최초 1회, Git Bash)**: `bash Isaac-GR00T/docker/build.sh` → `gr00t-dev`
- **서버 기동 (PowerShell)**: `pwsh scripts/serve_gr00t.ps1` 또는 `docker compose up gr00t-server`
- **클라이언트 실행 (호스트)**: `uv run scripts/evaluation/policy_inference.py --policy_type=gr00tn1.6 --policy_host=localhost --policy_port=5555 --policy_timeout_ms=15000 ...`
- **상세 절차 · 검증 · 트러블슈팅**: `docs/test-inference-in-simulation.md`

서버 엔트리포인트는 `Isaac-GR00T/gr00t/eval/run_gr00t_server.py` 이며, 컨테이너 외부에서 접속하려면 **`--host 0.0.0.0`** 바인드가 필수다(기본값 `127.0.0.1` 은 컨테이너 내부 루프백이라 호스트에서 접속 불가).

## 하드웨어 제약

### 카메라 (Microdia Integrated_Webcam_HD, USB 2.0 전용)

지원 해상도 프로파일: `1280×720`, `640×480` 두 가지만 존재한다.

카메라를 **USB 허브 경유로 연결하면 동시에 1대만 스트리밍된다.** MJPEG로 전송량을 줄여도 동일하게 실패한다.
허브의 하드웨어 한계로 포맷/해상도 변경으로는 해결되지 않는다.

**카메라마다 PC USB 포트에 직접 연결할 것.** ThinkStation 기준 USB 3.2 포트가 8개 있으므로 허브 없이 연결 가능하다.

```python
# 카메라 인덱스는 PC 포트에 직접 연결 기준
OpenCVCameraConfig(index_or_path=0)
```

## 노트북

`notebooks/` 에 참고자료로 Google Colab 기반 참고 노트북이 있다.

| 파일 | 내용 |
|------|------|
| `training_act.ipynb` | ACT 학습 + 데이터셋 병합 워크플로우 |
| `training-smolvla.ipynb` | SmolVLA 학습 워크플로우 |
