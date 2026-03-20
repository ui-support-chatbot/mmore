# =============================================================================
# MMORE Dockerfile — Docker 20.10.8 compatible
# =============================================================================
# Workarounds applied (see docs/private/DOCKER_DEPLOYMENT.md):
#   1. python:3.12-slim base (nvidia/cuda APT hooks crash under old seccomp)
#   2. pip instead of uv (uv's Tokio runtime panics under old seccomp)
#   3. PIP_PROGRESS_BAR=off (pip's rich progress bar spawns threads, blocked)
#   4. Build with: DOCKER_BUILDKIT=0 docker build --no-cache -t mmore-rag .
#   5. Run  with: --security-opt seccomp=unconfined --pids-limit -1
# =============================================================================

FROM python:3.12-slim
WORKDIR /app

# ── Disable pip's threaded progress bar (crashes under Docker 20 seccomp) ──
ENV PIP_PROGRESS_BAR=off
ENV PIP_NO_COLOR=1

# ── Which extras to install: "rag" (lighter) or "all" (includes process) ──
ARG INSTALL_EXTRAS="rag"

# ── Install build tools (gcc/g++ needed by pandas, numpy, etc. when building from source) ──
# Disable APT post-invoke scripts (blocked by Docker 20's seccomp profile)
RUN rm -f /etc/apt/apt.conf.d/docker-clean && \
    echo 'APT::Update::Post-Invoke-Success {};' > /etc/apt/apt.conf.d/99no-post-invoke && \
    apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# ── Copy only what's needed for install ──
COPY pyproject.toml /app/
COPY src /app/src

# ── 1. Install GPU PyTorch first (wheel bundles CUDA runtime, no nvidia/cuda base needed) ──
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu126

# ── 2. Install MMORE + missing runtime deps + pinned transformers ──
RUN pip install --no-cache-dir \
    -e ".[$INSTALL_EXTRAS]" \
    nltk tiktoken uvicorn fastapi "transformers==4.48.0"

# ── Default entrypoint ──
ENTRYPOINT ["python", "-m", "mmore"]
