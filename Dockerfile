FROM nvcr.io/nvidia/pytorch:25.04-py3

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute \
    DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMNI_KIT_ACCEPT_EULA=YES

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglu1-mesa \
        libegl1 \
        libx11-6 \
        libxext6 \
        libxrender1 \
        libxrandr2 \
        libxinerama1 \
        libxi6 \
        libxcursor1 \
        libxxf86vm1 \
        libxt6 \
        libxkbcommon0 \
        libsm6 \
        libice6 \
        libfontconfig1 \
        libfreetype6 \
        libxml2 \
        libwayland-client0 \
        libwayland-cursor0 \
        libwayland-egl1 \
        libvulkan1 \
        zenity \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/repo

COPY pyproject.toml uv.lock .python-version ./

RUN uv sync --frozen --no-install-project

CMD ["bash"]
