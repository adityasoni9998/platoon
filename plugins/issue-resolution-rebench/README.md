# SWE-rebench issue-resolution training

This plugin trains OpenHands on the `filtered` split of `nebius/SWE-rebench`
with AReaL and one `ModalWorkspace` per rollout. It uses the pre-published Modal
named images produced by the `swerebench_modal` branch of
`adityasoni9998/benchmarks`.

The AReaL configuration uses a 65,000-token prompt-plus-completion context
window, reserves 2,048 tokens for each completion (62,952 input tokens), and
limits the agent to 40 iterations per rollout.

The episode and OpenHands conversation use a 30-minute timeout. Each agent
action/environment step still has a 10-minute timeout. The agent Sandbox
defaults to the episode timeout plus 5 minutes for setup and cleanup
(35 minutes total); `MODAL_SANDBOX_TIMEOUT` overrides that Sandbox lifetime.

The training behavior is:

- observations are masked and only agent completion tokens receive loss;
- `filter_errors=False` keeps emitted agent tokens from completed rollouts;
- completed rollouts are not rejected based on finish, error, or budget status;
- the binary reward is `1.0` when the SWE-rebench harness reports `resolved`,
  otherwise `0.0`.

Length reward is disabled for the initial run, so the total reward is just the
binary reward. To enable it later, set
`workflow_config.rollout_config.extra.enable_length_penalty: true` in YAML.
When enabled, the total reward is `binary_reward + (threshold - turns) / (max_turns - threshold)`,
with the length coefficient fixed at `1.0`. `turns` counts unique nonempty LLM
response IDs on agent action/message events, so multiple tool calls from one
response count as one turn. Set
`workflow_config.rollout_config.extra.length_penalty_threshold` in YAML
(default `10`); it must be an integer from zero up to, but excluding,
`workflow_config.rollout_config.max_steps`. The formula is not clamped: with
40 maximum turns, 1/10/25/40 turns yield length rewards of +0.3/0/−0.5/−1.
The length term is computed from the OpenHands event stream alongside the
binary reward in terminal `evaluate()` calls. The returned reward info records
both components and the turn count. Partial trajectories that time out before
terminal evaluation receive no additional length reward during cleanup.

This does not guarantee that every rollout contributes loss. Groups with
identical rewards are marked non-trainable and removed before model work,
even with `filter_zero_variance_groups: false`. Steps without recorded
completion IDs or trainable completion tokens are skipped, and rollouts that
fail to return trajectory data cannot contribute training samples.

The SDK dependencies are pinned to `5acdf05`, which is the revision embedded in
the named images. The SWE-rebench harness is pinned to the fork commit
`1e5839b` (package version 4.0.3), and Modal is pinned to 1.5.5. Do not update
the SDK pin without rebuilding and republishing the named images.

## One-time environment setup

```bash
cd /project/flame/adityabs/hyperion_modal/plugins/issue-resolution-rebench
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/tmp/adityabs-swerebench/areal-venv}"
uv sync --no-active --extra areal
```

This installs the training environment in `/tmp/adityabs-swerebench/areal-venv`.
Repeat setup if that directory is removed (for example, after `/tmp` cleanup),
or dependencies change. To use another location, export `UV_PROJECT_ENVIRONMENT`
before setup and in each shell used for deployment or training. An activated
repository `.venv` does not change which environment these commands use.

Configure Modal/W&B authentication and noninteractive SSH access before
training.

## One-time deployment of the binary reward evaluator

The training process calls the evaluator adapted from Platoon commit
`7cb00ef`. It starts a clean Sandbox from the same pre-published named agent
image used by the rollout, applies the model patch, runs the SWE-rebench test
script, and grades the output with the pinned harness fork. The evaluator
removes the redundant dataset install command because the
`swerebench-testbed-reinstall-v2` image layer has already run it from
`/testbed` before publishing the named image. Deploy it once after environment
setup, and redeploy when the evaluator code changes:

```bash
uv run --no-active --no-sync --extra areal modal deploy \
  platoon/issue_resolution/test_execution_reward/modal_test_execution.py
```

If using a non-default Modal environment, pass that environment to
`modal deploy` and set `MODAL_ENVIRONMENT` to the same value for training. The deployed
app name is `swerebenchv1-evaluation`.

## Preflight checks

```bash
uv run --no-active --no-sync --extra areal modal profile current

SWEREBENCH_MAX_INSTANCES=1 uv run --no-active --no-sync --extra areal python -c \
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
./run_areal.sh
```

The script activates the prepared virtual environment, sets runtime environment
variables, clears the FlashInfer cache before each launch, and starts training
with a trial name ending in the short Git commit and UTC
timestamp. It works from any directory and accepts additional `key=value`
config overrides. It preserves existing HF cache, Sandbox timeout, and Ogma
endpoint settings. Complete the setup and deployment steps above before launching.
Set `MODAL_ENVIRONMENT` to the environment where the evaluator was deployed.
Unset `SWEREBENCH_MAX_INSTANCES` for a full dataset run.
Activation applies to the script and its child processes; your calling shell's
environment stays unchanged.

The host must authenticate non-interactively to
`adityabs@ogma.lti.cs.cmu.edu`. As in the legacy plugin, the launcher opens
reverse SSH forwards so Modal Sandboxes can reach the four AReaL proxy workers.
Set `PLATOON_OGMA_SSH_TARGET` and `PLATOON_OGMA_PUBLIC_HOST` to override those
defaults.

Both agent-server and test-execution Sandboxes force Modal Sandbox V2 in code;
the launcher's exported `MODAL_SANDBOX_V2=1` is intentionally redundant for
visibility. Rollout resources can be adjusted with `MODAL_CPU`, `MODAL_MEMORY`,
`MODAL_SANDBOX_TIMEOUT`, `MODAL_IDLE_TIMEOUT`, `MODAL_STARTUP_TIMEOUT`,
`MODAL_CLOUD`, and `MODAL_REGION`. The restored evaluator keeps the historical
five-minute timeout and Sandbox resource settings from `7cb00ef`.

For a small training smoke test, add overrides such as:

```bash
SWEREBENCH_MAX_INSTANCES=8 ./run_areal.sh \
  total_train_epochs=1 \
  train_dataset.batch_size=8 \
  workflow_config.group_size=2 \
  rollout.max_concurrent_rollouts=4 \
  stats_logger.wandb.mode=disabled
```
