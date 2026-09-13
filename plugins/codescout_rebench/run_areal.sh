#!/usr/bin/env bash
set -euo pipefail

# Run from this plugin regardless of the caller's working directory.
codescout_rebench_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$codescout_rebench_dir"

# Use the environment prepared with the README's one-time setup steps.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/tmp/adityabs-codescout-rebench/areal-venv}"
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

# Configure Modal/W&B credentials and noninteractive SSH access before launch.
export HF_HOME="${HF_HOME:-/tmp/adityabs-codescout-rebench/hf-cache}"
export OPENHANDS_SUPPRESS_BANNER=1
export ARROW_DEFAULT_MEMORY_POOL=system
export SGLANG_FORWARD_UNKNOWN_TOOLS=true
export MODAL_SANDBOX_V2=1
mkdir -p "$HF_HOME"

codescout_rebench_commit="$(git rev-parse --short=8 HEAD)"
codescout_rebench_timestamp="$(date -u +%Y%m%d-%H%M%SZ)"
codescout_rebench_trial="qwen3-4b-instruct-cispo-fsdp-${codescout_rebench_commit}-${codescout_rebench_timestamp}"

# Clear build files that may refer to an older virtual environment.
python -m flashinfer clear-cache

python -m platoon.codescout_rebench.train_areal \
  --config platoon/codescout_rebench/train_codescout_rebench_areal.yaml \
  trial_name="$codescout_rebench_trial" \
  "$@"
