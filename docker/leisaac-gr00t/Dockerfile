# syntax=docker/dockerfile:1.7
# Unified LeIsaac × GR00T image — single image, two virtualenvs.
#
#   /workspace/gr00t/.venv     Python 3.10  GR00T 학습 / 추론 (flash-attn, deepspeed, pytorch3d, torchcodec)
#   /workspace/leisaac/.venv   Python 3.11  IsaacSim 5.1 + LeIsaac + isaaclab + lerobot
#
# 두 venv 는 같은 컨테이너에서 GPU·HF 캐시·데이터셋 마운트를 공유하고 추론은 ZMQ (5555) 로 통신한다.
FROM nvcr.io/nvidia/pytorch:25.04-py3

ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# 1) System deps — IsaacSim Linux runtime + GR00T build tools
# NGC PyTorch 25.04 = Ubuntu 24.04 — t64 ABI 전환으로 일부 패키지명이 바뀌었다 (libasound2 → libasound2t64).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential yasm cmake libtool git pkg-config curl ca-certificates \
        libass-dev libfreetype6-dev libvorbis-dev autoconf automake texinfo tmux ffmpeg \
        libegl1 libgl1 libglu1-mesa libxcb1 libxrandr2 libxss1 libxcursor1 libxcomposite1 \
        libasound2t64 libxi6 libxtst6 libxkbcommon0 libxkbcommon-x11-0 libxshmfence1 \
 && rm -rf /var/lib/apt/lists/*

# 2) NVIDIA EGL/Vulkan ICD JSON — IsaacSim 렌더러용
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    cat >/usr/share/glvnd/egl_vendor.d/10_nvidia.json <<'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD" : { "library_path" : "libEGL_nvidia.so.0" }
}
EOF
RUN mkdir -p /usr/share/vulkan/icd.d && \
    cat >/usr/share/vulkan/icd.d/nvidia_icd.json <<'EOF'
{
    "file_format_version": "1.0.0",
    "ICD": { "library_path": "libGLX_nvidia.so.0", "api_version": "1.2.140" }
}
EOF

# 3) uv 설치 + Python 두 버전 prefetch
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
RUN uv python install 3.10 3.11

# 4) GR00T venv (Python 3.10) — 학습 + 추론
#    Isaac-GR00T 트리를 그대로 COPY 하고 그 안에서 .venv 생성
COPY Isaac-GR00T /workspace/gr00t
WORKDIR /workspace/gr00t

# NGC 컨테이너의 pip constraint 는 dill/scipy 버전을 강제하는데 GR00T 가 요구하는 버전과 충돌
RUN sed -i 's/dill==0.3.9/dill==0.3.8/g' /etc/pip/constraint.txt && \
    sed -i 's/scipy==1.15.2/scipy==1.15.3/g' /etc/pip/constraint.txt

# venv 생성 + 빌드 도구
RUN uv venv --python 3.10 .venv && \
    uv pip install --python .venv/bin/python --upgrade pip setuptools wheel

# CUDA 12.8 호환 torch 를 먼저 venv 에 설치 — 이후 pytorch3d / transformers 가 이 torch 를 그대로 본다.
# 호스트 pyproject.toml 과 같은 cu128 인덱스 사용.
RUN uv pip install --python .venv/bin/python \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.7.0 torchvision==0.22.0

# 무거운 ML/데이터 의존성 + pytorch3d 빌드/런타임 헬퍼 (ninja, fvcore, iopath)
RUN uv pip install --python .venv/bin/python \
        imageio h5py boto3 'transformers[torch]' deepspeed timm peft diffusers wandb \
        tianshou dm_tree openai albumentations==1.4.18 decord==0.6.0 torchcodec==0.4.0 \
        ninja fvcore iopath

# pytorch3d 는 PyPI 휠이 없어 source 빌드. setup.py 가 torch 를 import 하므로
# pip 의 build-isolation 을 끄고 .venv 의 torch 를 그대로 사용한다.
RUN cd /opt && git clone --depth 1 https://github.com/facebookresearch/pytorch3d.git && \
    cd pytorch3d && /workspace/gr00t/.venv/bin/pip install --no-build-isolation .

# GR00T 의존성 동기화 + editable 설치 (pyproject.toml 의 flash-attn 등이 이 단계에서 빌드됨)
RUN cd /workspace/gr00t && \
    uv sync --python 3.10 && \
    uv pip install --python .venv/bin/python -e .

# 5) LeIsaac venv (Python 3.11) — IsaacSim + leisaac + lerobot
WORKDIR /workspace/leisaac
COPY pyproject.toml uv.lock ./
# scripts/ 는 굽지 않고 runtime bind mount — 코드 편집이 빨리 반영되도록
RUN uv sync --frozen --python 3.11

# 두 venv 경로를 환경변수로 노출 (compose command 에서 참조)
ENV LEISAAC_VENV=/workspace/leisaac/.venv
ENV GR00T_VENV=/workspace/gr00t/.venv

# IsaacSim 이 EGL 헤드리스 렌더링을 위해 필요한 환경변수
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

# 6) 노출 포트
#    5555 GR00T policy server (ZMQ REQ-REP)
#    5556 leader-state ZMQ PUB (호스트→컨테이너, 양식만 노출)
#    8211 IsaacSim WebRTC livestream — 클라이언트 진입점
#    8011 IsaacSim WebRTC HTTP
#    49100 IsaacSim WebRTC signaling
#    47995-48005/udp WebRTC media
EXPOSE 5555 5556 8211 8011 49100
EXPOSE 47995-48005/udp

WORKDIR /workspace
CMD ["bash"]
