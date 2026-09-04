# SWE-rebench issue-resolution training

This plugin trains OpenHands on the `filtered` split of `nebius/SWE-rebench`
with AReaL and one `ModalWorkspace` per rollout. It uses the pre-published Modal
named images produced by the `swerebench_modal` branch of
`adityasoni9998/benchmarks`.

The implementation intentionally keeps the legacy training behavior:

- observations are masked and only agent completion tokens receive loss;
- `filter_errors=False` keeps emitted agent tokens from completed rollouts;
- completed rollouts are not rejected based on finish, error, or budget status;
- the only environment reward is `1.0` when the SWE-rebench harness reports
  `resolved`, otherwise `0.0`.

The SDK dependencies are pinned to `5acdf05`, which is the revision embedded in
the named images. The SWE-rebench harness is pinned to the fork commit
`1e5839b` (package version 4.0.3), and Modal is pinned to 1.5.5. Do not update
the SDK pin without rebuilding and republishing the named images.

## Install

```bash
cd /project/flame/adityabs/hyperion_modal/plugins/issue-resolution-rebench
uv sync --extra areal
```

## Deploy the binary reward evaluator

The training process calls the evaluator adapted from Platoon commit
`7cb00ef`. It starts a clean Sandbox from the same pre-published named agent
image used by the rollout, applies the model patch, runs the SWE-rebench test
script, and grades the output with the pinned harness fork. The evaluator
removes the redundant dataset install command because the
`swerebench-testbed-reinstall-v2` image layer has already run it from
`/testbed` before publishing the named image. Deploy the evaluator once in the
same Modal environment used for training:

```bash
uv run modal deploy \
  platoon/issue_resolution/test_execution_reward/modal_test_execution.py
```

If using a non-default Modal environment, pass that environment to
`modal deploy` and set `MODAL_ENVIRONMENT` to the same value for training. The deployed
app name is `swerebenchv1-evaluation`.

## Preflight checks

```bash
uv run modal profile current

SWEREBENCH_MAX_INSTANCES=1 uv run python -c \
  "from platoon.issue_resolution.tasks import load_data, named_agent_server_image_for_instance; train, _ = load_data(); task = next(iter(train.values())); print(task.id, named_agent_server_image_for_instance(task.misc))"
```

`SWEREBENCH_MAX_INSTANCES` is only a local smoke-test aid. Unset it before a
real training launch.

Each named image contains a ready `/testbed`. Workspace startup resets the
checkout to the image's existing `HEAD` and removes untracked files before each
rollout.

## Train

```bash
cd /project/flame/adityabs/hyperion_modal/plugins/issue-resolution-rebench
source .venv/bin/activate

export HF_HOME="/tmp/adityabs-swerebench/hf-cache"
export OPENHANDS_SUPPRESS_BANNER=1
export ARROW_DEFAULT_MEMORY_POOL=system
export SGLANG_FORWARD_UNKNOWN_TOOLS=true
export MODAL_SANDBOX_V2=1
mkdir -p "$HF_HOME"

export trial="areal-flame-swerebench-fft-cispo-$(date +%Y%m%d-%H%M%S)"

uv run --extra areal python -m platoon.issue_resolution.train_areal \
  --config platoon/issue_resolution/train_issue_resolution_fft_areal_fsdp.yaml \
  trial_name="$trial"
```

The host must authenticate non-interactively to
`adityabs@ogma.lti.cs.cmu.edu`. As in the legacy plugin, the launcher opens
reverse SSH forwards so Modal Sandboxes can reach the four AReaL proxy workers.
Set `PLATOON_OGMA_SSH_TARGET` and `PLATOON_OGMA_PUBLIC_HOST` to override those
defaults.

Both agent-server and test-execution Sandboxes force Modal Sandbox V2 in code;
the exported `MODAL_SANDBOX_V2=1` above is intentionally redundant for
visibility. Rollout resources can be adjusted with `MODAL_CPU`, `MODAL_MEMORY`,
`MODAL_SANDBOX_TIMEOUT`, `MODAL_IDLE_TIMEOUT`, `MODAL_STARTUP_TIMEOUT`,
`MODAL_CLOUD`, and `MODAL_REGION`. The restored evaluator keeps the historical
five-minute timeout and Sandbox resource settings from `7cb00ef`.

For a small training smoke test, add overrides such as:

```bash
SWEREBENCH_MAX_INSTANCES=8 uv run --extra areal python -m \
  platoon.issue_resolution.train_areal \
  --config platoon/issue_resolution/train_issue_resolution_fft_areal_fsdp.yaml \
  trial_name="debug-run" \
  total_train_epochs=1 \
  train_dataset.batch_size=8 \
  workflow_config.group_size=2 \
  rollout.max_concurrent_rollouts=4 \
  stats_logger.wandb.mode=disabled
```
