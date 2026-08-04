# Issue-resolution training with AReaL

The current configuration runs full-parameter CISPO training with the AReaL
FSDP backend on one 8-GPU node. Four GPUs are assigned to the FSDP actor and
four GPUs are assigned to SGLang rollout inference. The total context window is
65,000 tokens, including the prompt and completion.

## Install

```bash
cd /project/flame/adityabs/hyperion/plugins/issue-resolution-swesmith

uv sync --extra areal
```

The `areal` and `tinker` extras are mutually exclusive. Do not pass both extras
to the same `uv sync` or `uv run` command.

## Optional checks

Confirm that the GPUs, Apptainer, and the AReaL Python environment are visible:

```bash
nvidia-smi
apptainer --version

uv run --extra areal python -c \
  "import areal; from platoon.train.areal import PlatoonArealRLTrainer; print('AReaL ready')"
```

If Hugging Face or Weights & Biases require authentication in the environment,
log in before starting training:

```bash
uv run --extra areal huggingface-cli login
uv run --extra areal wandb login
```

Skip either login when credentials are already configured or the service is not
needed. The YAML currently enables online WandB logging.

## Train

Copy and run this block:

```bash
cd /project/flame/adityabs/hyperion/plugins/issue-resolution-swesmith
source .venv/bin/activate

export PATH="/home/adityabs/.local/apptainer/bin:/home/adityabs/.local/bin:$PATH"

export APPTAINER_CACHEDIR="/tmp/adityabs-apptainer/cache"
export APPTAINER_TMPDIR="/tmp/adityabs-apptainer/tmp"
export OPENHANDS_APPTAINER_BUILD_ROOT="/tmp/adityabs-apptainer/swesmith-build"

export APPTAINER_EXECUTABLE="/home/adityabs/.local/apptainer/bin/apptainer"
export LOCALIZATION_HOST_PATCH="/home/adityabs/.local/opt/gnu-patch/usr/bin/patch"

export HF_HOME="/tmp/adityabs-apptainer/hf-cache"
export OPENHANDS_SUPPRESS_BANNER=1

mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$HF_HOME"
chmod 700 "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

# Set a newly rotated WANDB_API_KEY securely before launching, unless
# authentication was already configured with `wandb login`.
export trial="areal-flame-swesmith-fft-cispo-$(date +%Y%m%d-%H%M%S)"
export WANDB_API_KEY="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"
export ARROW_DEFAULT_MEMORY_POOL=system
export OPENHANDS_APPTAINER_EXTRA_BINDS="/run/systemd/resolve/resolv.conf:/etc/resolv.conf"
export SGLANG_FORWARD_UNKNOWN_TOOLS=true
uv run --extra areal python -m platoon.issue_resolution.train_areal \
  --config platoon/issue_resolution/train_issue_resolution_fft_areal_fsdp.yaml \
  trial_name="$trial"
```

`APPTAINER_CACHEDIR` is Apptainer's internal cache. The prebuilt SWE-smith SIF
files are kept separately under `OPENHANDS_APPTAINER_BUILD_ROOT`.

Training artifacts, checkpoints, AReaL logs, and rollout event files are written
under the configured `cluster.fileroot`, currently:

```text
/tmp/platoon_skyrl_areal/experiments
```

Change `cluster.fileroot` and `cluster.name_resolve.nfs_record_root` in the YAML
before launching if the artifacts must survive a reboot or be shared across
multiple nodes.

## Useful command-line overrides

AReaL configuration overrides use `key=value` syntax:

```bash
uv run --extra areal python -m platoon.issue_resolution.train_areal \
  --config platoon/issue_resolution/train_issue_resolution_fft_areal_fsdp.yaml \
  trial_name="debug-run" \
  total_train_epochs=1 \
  train_dataset.batch_size=8 \
  workflow_config.group_size=2 \
  rollout.max_concurrent_rollouts=4 \
  stats_logger.wandb.mode=disabled
```

The smaller values above are useful for a smoke test, but they change the
effective training batch and should not be used as a replacement for the main
experiment configuration.

## Stop a run

Use `Ctrl-C` in the training terminal and allow AReaL to shut down its workers.
If a worker remains after shutdown, inspect its process before terminating it:

```bash
ps -fu "$USER" | grep -E 'train_areal|areal|sglang' | grep -v grep
```
