#!/bin/bash

# Ensure we are executing from the root of the MMORE repository
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "🚀 Starting MMORE Fully Local RAGAS Evaluation Container..."

# Spins up a temporary Python container bypassing server restrictions, 
# mounts the codebase, installs requirements, and executes the evaluation.
docker run -it --name mmore-eval --rm \
  --network host \
  --pids-limit -1 \
  --security-opt seccomp=unconfined \
  --entrypoint /bin/bash \
  -w /app \
  -v "$(pwd)":/app \
  -v ~/mmore_data:/app/mmore_data \
  -v ~/.cache/pip:/root/.cache/pip \
  mmore-rag -c "
    export PIP_PROGRESS_BAR=off && \
    export PIP_NO_COLOR=1 && \
    echo '[1/2] Loading Ragas Dependencies (uses cache after first run)...' && \
    pip install --quiet --disable-pip-version-check ragas datasets langchain-ollama langchain-huggingface pandas && \
    echo '[2/2] Running RAGAS Evaluation Script...' && \
    python evaluation/evaluate_models.py
  "

echo "✅ Evaluation Container Successfully Closed."
echo "📄 Check evaluation/evaluation_results_full_local.csv for your metrics!"
