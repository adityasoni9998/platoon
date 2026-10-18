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
dense reward. All reward components are computed in either mode. The optional
length reward is added separately; keep it disabled for purely binary rewards.

The launcher automatically exposes its local Tinker proxy through a temporary
Modal HTTPS router. Agents call that URL; the proxy calls the upstream
`TINKER_BASE_URL`. No manual forwarding is needed. The router defaults to a
24-hour lifetime; increase `--inference_tunnel.modal.timeout` for longer runs.

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
evaluates rewards once at the end. Steps appear in JSONL after the run exits;
`step_timeout` still applies to each action/step, including final evaluation.
Run failures use the underlying error message without the conversation-ID prefix.
New subprocesses load this behavior immediately, but a running trainer keeps its
already-loaded parent subprocess deadline until restart.

Logs and checkpoint metadata go under
`/data/user_data/adityabs/miles_tinker_rebench/experiments/issue-resolution-rebench-tinker/<trial_name>/`; change `--log_path`
to choose another root. Weight and optimizer files use the Tinker service's
checkpoint storage. Reusing a trial directory resumes its checkpoint metadata.
