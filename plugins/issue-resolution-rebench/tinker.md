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

## Profiling inference on a client or server node

The same profiler works wherever the Platoon training client runs, including the
Tinker server node. Install the plugin with `uv sync --locked --extra tinker`
as above so the private SDK hooks use the tested Tinker version (0.16.1).
The node still needs the usual Modal, W&B, and model/dataset access for this
workload. Profiling does not require changes to the Tinker server.

To measure a local client-to-server path, set `TINKER_BASE_URL` to the server's
actual local HTTP endpoint and export its `TINKER_API_KEY`. Keeping the public
Modal URL still sends SDK requests through that relay. The OpenHands agents
continue to reach Platoon's inference router through Modal.

From `plugins/issue-resolution-rebench`, launch a fresh, two-batch profiling run:

```bash
: "${TINKER_BASE_URL:?Export the Tinker endpoint to measure}"
: "${TINKER_API_KEY:?Export your Tinker API key}"
unset TINKER_SUBPROCESS_SAMPLING
uv run --locked --extra tinker python -m platoon.issue_resolution.train_tinker \
  --config platoon/issue_resolution/train_issue_resolution_ppo_tinker.yaml \
  --tinker_base_url "$TINKER_BASE_URL" \
  --log_path "$HOME/platoon-profile-runs" \
  --stats.trial_name "rebench-ppo-profile-$(date +%Y%m%d-%H%M%S)" \
  --train.max_training_steps 2 \
  --profiling.enabled true
```

Choose a writable `--log_path` on that node; the YAML's default `/data/...` path
belongs to the original client machine. This command runs PPO training,
including the initial critic-only warmup batch. Adjust the batch limit as needed.
SDK subprocess sampling is unsupported by the profiler and is rejected when
enabled; this is separate from the OpenHands rollout subprocesses.

Profiling is disabled by default. When enabled, it records prompt preparation,
SDK scheduling and admission waits, submission acknowledgements, answer
retrieval, HTTP transport events, response parsing, and training spans. Separate
heartbeats measure delays on the proxy, SDK, and trainer event loops. Request
IDs link SDK calls to proxy traces, and proxy responses include an
`X-Platoon-Trace-ID` header.

Trace files are written to
`<log_path>/<experiment_name>/<trial_name>/profiling/trace-<timestamp>-<pid>.jsonl`.
From the repository root, summarize a run or inspect one request:

```bash
python3 scripts/profile_inference.py /path/to/trial/profiling/trace-*.jsonl
python3 scripts/profile_inference.py /path/to/trace.jsonl --trace-id TRACE_ID
```

The summary reports counts, mean, median, p95, and maximum durations. Spans
overlap: `sdk.bytes_wait` is part of `sdk.dispatch_wait`, and SDK HTTP calls are
inside submission/retrieval spans. Do not add those rows together. HTTP waiting
includes local connection handling, network communication, and remote work;
it does not isolate network latency or GPU generation time. The trace does not
directly measure individual HTTP connection locks. Timings use local monotonic
clocks; match request IDs for comparisons with server logs.

JSON encoding and disk writes run in a background thread. Event producers use a
bounded buffer and drop records instead of waiting when it is full or busy.
The summary reports dropped events and incomplete spans; tracing still has
some CPU overhead. Traces omit request bodies, prompts, tokens, and credentials.
The hooks are removed on normal session exit and do not edit installed packages
or change the SDK's admission budget or HTTP connection settings.
