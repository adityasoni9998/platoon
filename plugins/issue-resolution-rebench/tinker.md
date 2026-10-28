# SWE-rebench training with Tinker

Requires `uv`, `ssh`, `ssh-keygen`, configured Modal credentials, and exported
`TINKER_API_KEY`, `TINKER_BASE_URL`, and `WANDB_API_KEY`. The Tinker service must
support the model and training mode selected in the YAML configuration.

From the repository root, create a separate environment for this plugin:

```bash
cd plugins/issue-resolution-rebench
export UV_PROJECT_ENVIRONMENT=/tmp/issue-resolution-rebench-tinker-venv
uv sync --locked --extra tinker
export OPENHANDS_SUPPRESS_BANNER=1
export MODAL_SANDBOX_V2=1
export ARROW_DEFAULT_MEMORY_POOL=system
```

Use the Modal profile/environment containing the pre-published SWE-rebench
named images. Deploy the existing evaluator once (and after evaluator changes):

```bash
uv run --locked --extra tinker modal deploy \
  platoon/issue_resolution/test_execution_reward/modal_test_execution.py
```

For a non-default Modal environment, pass `--env NAME` to deployment and export
`MODAL_ENVIRONMENT=NAME` for training. The app is `swerebenchv1-evaluation`.
See the [README](README.md) for the named-image and reward setup.

Launch training:

```bash
: "${TINKER_BASE_URL:?Export your Tinker service URL}"
unset SWEREBENCH_MAX_INSTANCES
uv run --locked --extra tinker python -m platoon.issue_resolution.train_tinker \
  --config platoon/issue_resolution/train_issue_resolution_fft_tinker.yaml \
  --tinker_base_url "$TINKER_BASE_URL" \
  --stats.trial_name "rebench-tinker-$(date +%Y%m%d-%H%M%S)"
```

Training settings are defined in
[`train_issue_resolution_fft_tinker.yaml`](platoon/issue_resolution/train_issue_resolution_fft_tinker.yaml).
Training overrides use the `--train.*` prefix.

Set `train.workflow_config.rollout_config.extra.use_binary_reward` to `true`
for resolved/not-resolved training rewards, or `false` for the existing weighted
dense reward. All reward components are computed in either mode.

By default, the launcher exposes its local Tinker proxy through a temporary
Modal HTTPS router. Agents call that URL; the proxy calls the upstream
`TINKER_BASE_URL`. No manual forwarding is needed. The router defaults to a
24-hour lifetime; increase `--inference_tunnel.modal.timeout` for longer runs.

## Local Apptainer workspaces (ABCI)

Merge the following fields into the chosen Tinker recipe. Each prebuilt image
must be named `<instance_id>.sif` inside `sif_dir`, using the dataset instance ID.
Each SIF must contain the prepared `/testbed` and an OpenHands agent-server
entrypoint compatible with the pinned SDK. Rollouts never build or pull images.

```yaml
train:
  workflow_config:
    rollout_config:
      extra:
        workspace:
          type: apptainer
          apptainer:
            sif_dir: null  # Required: directory containing the prebuilt SIFs.
            cache_dir: null  # Required: Apptainer cache directory.
            scratch_dir: null  # Required: existing node-local scratch directory.

inference_tunnel:
  backend: none
```

On the compute node running the Platoon client, select the installed Apptainer
and the current allocation's local scratch, following the Miles launch convention:

```bash
export PATH="$HOME/.local/bin:$PATH"
apptainer version
export ABCI_SCRATCH="${ABCI_SCRATCH:-${PBS_LOCALDIR:-}}"
: "${ABCI_SCRATCH:?Set ABCI_SCRATCH to this node's allocation directory under /local}"
```

This installation exposes `~/.local/apptainer/bin/apptainer` through
`~/.local/bin/apptainer`. Keep the SIF/cache storage under your user directory
in `/groups/gai51740`. The cache directory is created if absent, using default
permissions subject to the process umask; existing permissions are unchanged.
Replace the YAML's `null` path values with concrete paths, preferably absolute.
Missing or `null` required paths fail configuration validation. YAML path values
are literal: `$VARIABLE`, `${VARIABLE}`, and `~` are not expanded or checked for
unset variables. Shell variables can be expanded when passing CLI overrides,
for example `--train.workflow_config.rollout_config.extra.workspace.apptainer.scratch_dir "$ABCI_SCRATCH"`.
The scratch directory must already exist. There is no fallback to `/home` or `/tmp`.

Each workspace creates its own directory overlay on scratch, with no configured
size limit. With `--containall`, its private home, `/tmp`, and `/var/tmp` use
`--workdir` storage under that workspace's scratch directory. The runtime receives
`APPTAINER_CACHEDIR` from the YAML and `APPTAINER_TMPDIR` / `TMPDIR` pointing to
its own `runtime-tmp` directory on scratch. No launcher exports are required for
these temporary paths, and the parent process's environment is unchanged.
The per-workspace directories are removed after normal cleanup or a handled
startup failure. Abrupt process termination can leave scratch files; allocation
cleanup remains the final fallback.

Apptainer uses host networking. With the client and workspace on the same host,
agents call the existing LiteLLM proxy through localhost. Set `TINKER_BASE_URL`
to Miles' internal endpoint and pass it using the launch command above. Setting
`inference_tunnel.backend: none` creates no Platoon relay; the proxy itself still
adapts inference requests and records training interactions. Miles' own relay
configuration is separate.

Modal credentials and the deployed Modal evaluator are still required for test
rewards. Workspace selection does not change the reward evaluator, rollout
timeouts, or the three setup attempts. Apptainer startup uses the existing
600-second startup budget; port selection and health checks use the SDK defaults.

## Subprocess lifecycle

Each rollout runs in a fresh subprocess with its own interaction session. The
parent retains the proxy and collects the recorded training interactions.
The subprocess deadline defaults to the conversation timeout plus 480 seconds
for imports, workspace setup, reward evaluation, cleanup, and interpreter shutdown; override it with
`--train.workflow_config.rollout_config.extra.subprocess_timeout SECONDS`.
After the deadline or cancellation, the parent allows up to 180 additional
seconds for child cleanup before a forced kill.
The launcher stops the proxy and Modal router when training exits.

The child waits for the `conversation.run()` thread to exit, then refreshes and
reconciles once and freezes the events. The existing `reset/act/step` loop
consumes that history, preserving completion IDs and tool-call grouping, and
evaluates terminal rewards at the end. In PPO turn mode, unique captured patches
are scored after generation in `reset()`, and each `evaluate()` applies turn
penalties and score deltas. Steps appear in JSONL after generation and snapshot
scoring; `step_timeout` still applies to each action/step.
Run failures use the underlying error message without the conversation-ID prefix.
New subprocesses load this behavior immediately, but a running trainer keeps its
already-loaded parent subprocess deadline until restart.

Logs and checkpoint metadata go under
`/data/user_data/adityabs/miles_tinker_rebench/experiments/issue-resolution-rebench-tinker/<trial_name>/`; change `--log_path`
to choose another root. Weight and optimizer files use the Tinker service's
checkpoint storage. Reusing a trial directory resumes its checkpoint metadata.
