# Issue-resolution training with AReaL

The current configuration runs full-parameter CISPO training with the AReaL
FSDP backend on one 8-GPU node. Four GPUs are assigned to the FSDP actor and
four GPUs are assigned to SGLang rollout inference. The total context window is
32,768 tokens, including up to 30,720 prompt tokens and 2,048 completion tokens.

This Modal recipe uses the exact 257-instance easy subset from Platoon commit
`dd43ec0`, a binary resolved/not-resolved reward, and the easy-run rollout
policy: completed rollouts are not discarded based on finish, error, or budget
status. `filter_errors=False` likewise keeps their agent tokens. The existing
trainer-side zero-variance group filtering remains active; groups retained for
rollout accounting with zero advantages are removed before model work.

## Install

```bash
cd /project/flame/adityabs/hyperion_modal/plugins/issue-resolution-legacy

uv sync --extra areal
```

The `areal` and `tinker` extras are mutually exclusive. Do not pass both extras
to the same `uv sync` or `uv run` command.

## Optional checks

Confirm that the GPUs, Modal credentials, and the AReaL Python environment are visible:

```bash
nvidia-smi
uv run modal profile current

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
cd /project/flame/adityabs/hyperion_modal/plugins/issue-resolution-legacy
source .venv/bin/activate

export HF_HOME="/tmp/adityabs-swesmith/hf-cache"
export OPENHANDS_SUPPRESS_BANNER=1

mkdir -p "$HF_HOME"

# Set WANDB_API_KEY securely before launching, unless authentication was already
# configured with `wandb login`. Do not commit a real key to this file.
export trial="areal-flame-swesmith-fft-cispo-$(date +%Y%m%d-%H%M%S)"
export WANDB_API_KEY="<your-wandb-api-key>"
export ARROW_DEFAULT_MEMORY_POOL=system
export SGLANG_FORWARD_UNKNOWN_TOOLS=true
export MODAL_SANDBOX_V2=1
uv run --extra areal python -m platoon.issue_resolution.train_areal \
  --config platoon/issue_resolution/train_issue_resolution_fft_areal_fsdp.yaml \
  trial_name="$trial"
```

Each rollout creates a Modal Sandbox from the pre-published named image for its
SWE-Smith base image. The host must have working Modal credentials. It must also
be able to authenticate non-interactively to `adityabs@ogma.lti.cs.cmu.edu`;
training opens four reverse SSH forwards from Ogma to the four current AReaL
proxy workers. The number of forwards intentionally matches the YAML's
`sglang:d4p1t1` rollout topology.

Modal resource defaults can be overridden with `MODAL_CPU`, `MODAL_MEMORY`,
`MODAL_SANDBOX_TIMEOUT`, `MODAL_IDLE_TIMEOUT`, `MODAL_STARTUP_TIMEOUT`,
`MODAL_CLOUD`, `MODAL_REGION`, and `MODAL_ENVIRONMENT`.
Set `MODAL_SANDBOX_V2=1` in the trainer process to use Modal Sandbox V2, as in
the launch block above.

The named-image lookup mirrors the publisher in the `swe_smith_modal`
benchmarks branch, including component shortening and hash suffixes, and uses
the images built from software-agent-sdk commit `5acdf05`.

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
