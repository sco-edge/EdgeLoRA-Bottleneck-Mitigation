#!/usr/bin/env bash
# Build the pinned paper runtime on a Jetson Orin using its existing CUDA toolkit.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
runtime="${EDGELORA_RUNTIME:-${repo_root}/vendor/EdgeLoRA/edgelora}"
jobs="${BUILD_JOBS:-1}"
arch="${CUDA_ARCH:-sm_87}"

if [[ ! -f "${runtime}/Makefile" ]]; then
    printf 'Runtime missing. Run python3 scripts/setup_runtime.py first.\n' >&2
    exit 1
fi
command -v nvcc >/dev/null || { printf 'nvcc is required; activate the Jetson CUDA toolkit.\n' >&2; exit 1; }
command -v make >/dev/null || { printf 'GNU make is required.\n' >&2; exit 1; }
if [[ ! "$jobs" =~ ^[1-9][0-9]*$ ]]; then
    printf 'BUILD_JOBS must be a positive integer.\n' >&2
    exit 1
fi

# The original artifact uses GNU make. Keep compilation serial by default on 8 GB Jetson.
make -C "$runtime" -j"$jobs" GGML_CUDA=1 CUDA_DOCKER_ARCH="$arch" llama-server llama-quantize
printf '\nBuilt server: %s/llama-server\n' "$runtime"
