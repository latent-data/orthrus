#!/usr/bin/env bash
set -euo pipefail

IMAGE=orthrus-benchmark
NGC_IMAGE=nvcr.io/nvidia/pytorch:25.12-py3
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NO_BUILD=0

# Consume --no-build; forward everything else to benchmark.py
BENCH_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--no-build" ]]; then
        NO_BUILD=1
    else
        BENCH_ARGS+=("$arg")
    fi
done

# --- Prerequisites ---
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Install the NVIDIA driver and try again." >&2
    exit 1
fi

if ! docker info 2>/dev/null | grep -q "nvidia"; then
    echo "ERROR: NVIDIA Container Toolkit not detected in Docker runtime." >&2
    echo "  Install it from: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/" >&2
    exit 1
fi

mkdir -p "${SCRIPT_DIR}/results"

if [[ "$NO_BUILD" -eq 1 ]]; then
    echo "==> Skipping build; using ${NGC_IMAGE} with mounted source ..."
    docker run --rm \
        --gpus all \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
        -v "${SCRIPT_DIR}:/workspace" \
        -w /workspace \
        "${NGC_IMAGE}" \
        bash -c 'pip install --no-deps -r requirements.txt && python -u benchmark.py "$@"' -- "${BENCH_ARGS[@]}"
else
    echo "==> Building image ${IMAGE} ..."
    docker build -t "${IMAGE}" "${SCRIPT_DIR}"

    echo "==> Running benchmark ..."
    docker run --rm \
        --gpus all \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
        -v "${SCRIPT_DIR}/results:/workspace/results" \
        "${IMAGE}" \
        python -u benchmark.py "${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"}"
fi
