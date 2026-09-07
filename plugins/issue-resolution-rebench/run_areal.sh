#!/usr/bin/env bash
set -euo pipefail

# Run from this plugin regardless of the caller's working directory.
rebench_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$rebench_dir"

# Use the environment prepared with the README's one-time setup steps.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/tmp/adityabs-swerebench/areal-venv}"
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

# Configure Modal/W&B credentials and noninteractive SSH access before launch.
export HF_HOME="${HF_HOME:-/tmp/adityabs-swerebench/hf-cache}"
export OPENHANDS_SUPPRESS_BANNER=1
export ARROW_DEFAULT_MEMORY_POOL=system
export SGLANG_FORWARD_UNKNOWN_TOOLS=true
export MODAL_SANDBOX_V2=1
mkdir -p "$HF_HOME"

rebench_commit="$(git rev-parse --short=8 HEAD)"
rebench_timestamp="$(date -u +%Y%m%d-%H%M%SZ)"
rebench_trial="qwen3-4b-instruct-cispo-fsdp-${rebench_commit}-${rebench_timestamp}"

# Clear build files that may refer to an older virtual environment.
python -m flashinfer clear-cache

python -m platoon.issue_resolution.train_areal \
  --config platoon/issue_resolution/train_issue_resolution_fft_areal_fsdp.yaml \
  trial_name="$rebench_trial" \
  "$@"
