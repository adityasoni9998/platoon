# Easy SWE-Smith training with Tinker

This recipe uses the same 257-instance easy subset, binary reward, rollout
limits, CISPO settings, and 11 epochs as the AReaL FSDP recipe. It requests
full-parameter training (`lora_rank: 0`), so the configured Tinker-compatible
service must support rank-zero training. Both recipes allow 32 concurrent task
workflows, each launching a group of eight rollouts (up to 256 agent rollouts).
The learning rates intentionally differ: Tinker uses `1e-6`, AReaL uses `3e-6`.

## Install and run

From the repository root:

```bash
cd plugins/issue-resolution-legacy
export UV_PROJECT_ENVIRONMENT=/tmp/issue-resolution-legacy-tinker-venv
uv sync --locked --extra tinker

# Configure TINKER_API_KEY, Modal credentials, and WANDB_API_KEY beforehand.
export OPENHANDS_SUPPRESS_BANNER=1
export MODAL_SANDBOX_V2=1
uv run --locked --extra tinker python -m platoon.issue_resolution.train_tinker \
  --config platoon/issue_resolution/train_issue_resolution_fft_tinker.yaml
```

Use `--tinker_base_url http://your-tinker-service:port` to select a custom
Tinker-compatible endpoint. With `tinker_base_url: null`, the Tinker SDK uses
its default endpoint selection. This upstream service URL is distinct from
the OpenAI-compatible proxy URL used by Modal.

The existing `tinker` extra provides `platoon[tinker]`; Platoon also supplies
FastAPI and Uvicorn. No AReaL installation is needed. Keep backend environments
separate: use `/tmp/issue-resolution-legacy-areal-venv` with `--extra areal`
for AReaL. The two extras are mutually exclusive.

## Inference transport

Both Tinker and AReaL select forwarding through the same YAML section.
The Tinker recipe selects Modal; the AReaL recipe selects Ogma. To use a
temporary Modal router, set:

```yaml
inference_tunnel:
  backend: modal
```

All other router settings have defaults; the complete section is:

```yaml
inference_tunnel:
  backend: modal  # ogma preserves the existing SSH relay
  modal:
    app_name: platoon-inference-router
    cpu: 0.125
    memory: 128  # MiB
    timeout: 86400  # maximum lifetime in seconds (24 hours)
```

These resource requests use [Modal's baseline CPU and memory allocation](https://modal.com/docs/guide/resources).
Actual usage can exceed the requests. The router's maximum lifetime must cover
the training run; it does not automatically rotate or renew when that lifetime
expires. Normal exit and startup failures terminate the sandbox and remove the
local temporary SSH keys.

With Modal forwarding, the request path is:

```text
Modal agent -> public HTTPS tunnel -> Modal router -> reverse SSH
            -> local OpenAI-compatible Tinker proxy -> Tinker service
```

The trainer initiates SSH outbound, so it needs no public IP or inbound port.
The router uses one public HTTPS tunnel per proxy, plus an SSH port authenticated
with a fresh per-run key and pinned host key. The Tinker API key stays on the
trainer host. Tinker uses one proxy; AReaL can forward all its worker proxies
through one router sandbox. The sandbox exposes [Modal tunnel URLs](https://modal.com/docs/guide/sandbox-networking),
not a dedicated public IP.

The Modal route uses the current Modal profile. For the `neulab` profile:

```bash
export MODAL_PROFILE=neulab
uv run --locked --extra tinker python -m platoon.issue_resolution.train_tinker \
  --inference_tunnel.backend modal
```

Local `ssh` and `ssh-keygen` executables are required. The router image installs
OpenSSH on first use. For the Ogma route, noninteractive SSH access to
`adityabs@ogma.lti.cs.cmu.edu` and public reverse forwarding must already work.

The training job owns **one Tinker HTTP proxy and one public `/v1` URL**, shared
by every rollout. Each rollout adds a unique session header so the proxy can
keep its training interactions separate. No per-rollout proxy or tunnel is
created. AReaL retains its existing per-worker proxies and session credentials.

`TinkerProxyRollout` starts a fresh Python subprocess for each `run_rollout`.
There is no rollout worker pool. Existing workflow concurrency bounds the
number of children: `num_concurrent_rollout_workflow_workers * group_size`
(up to 256 with this recipe). Training and the shared HTTP proxy stay in the
parent process. Synchronous conversation initialization, state polling, message
sending, and patch extraction can block only their own rollout's event loop.
The SDK may still use threads internally within each child.

The parent passes the already-loaded task and a copied config through private
local pickle files; the child does not reload the datasets. The child creates
its own trajectory collection and returns a trajectory dictionary. Python
context variables are not copied. HTTP requests carry the unique session header,
and the parent retrieves the corresponding interactions from its shared proxy.
No proxy object or Tinker sampling client is serialized into the child.

Each child exits after its rollout. On cancellation or subprocess timeout, the parent sends
SIGTERM so the child can run normal cleanup, allows 180 seconds, then kills and
reaps a stuck child before discarding its session and temporary files. Repeated
cancellation does not bypass this cleanup. Forced termination can leave remote
Modal resources alive until their sandbox timeout expires.

The existing rollout and SDK timeouts remain active. The parent additionally
bounds the whole child lifetime, including imports, workspace startup/retries,
cleanup, and interpreter shutdown. Its deadline defaults to the episode timeout
plus 7,200 seconds: 8,430 seconds for this recipe. The two-hour allowance covers
the three potentially lengthy workspace setup attempts and cleanup. Override
`train.workflow_config.rollout_config.extra.subprocess_timeout` to set a different
total limit in seconds. Termination can take an additional 180 seconds for cleanup.
This parent deadline still fires when a blocking call prevents the child's
asyncio timeout from running or a leftover thread prevents process exit.

This follows the existing AReaL/inference boundary of running rollouts in children
while the parent handles workflow results. Those workflows use process pools;
this plugin trades repeated interpreter/import startup for fresh process state
and independent termination of each rollout. AReaL's execution mode is unchanged.
Router setup and teardown remain outside the training loop.

The tunnel code uses ordinary context managers: `inference_tunnels` returns a
map of internal to public URLs, `ssh_connection` owns the SSH process, and
`modal_tunnels` additionally owns the sandbox and temporary keys. Resources
unwind in reverse order on normal exit, startup failure, or interruption. Both
routes check `/health` before training starts.

## Small run

```bash
uv run --locked --extra tinker python -m platoon.issue_resolution.train_tinker \
  --train.num_epochs 1 \
  --train.batch_size 2 \
  --train.workflow_config.group_size 2 \
  --train.num_concurrent_rollout_workflow_workers 1 \
  --stats.wandb.mode disabled
```

## Artifact paths and recovery

The Babel artifact root is `log_path`, defaulting to
`/data/user_data/adityabs/platoon_tinker/experiments`. The trainer appends
`stats.experiment_name` and `stats.trial_name`, so the default run directory is:

```text
/data/user_data/adityabs/platoon_tinker/experiments/easy_swesmith_257/easy_swesmith_257/
```

Within that directory:

- `rollouts/train/<model-version>/events/` contains rollout event JSONL files.
- `wandb/` contains local W&B files; the recipe also logs online to
  `issue_resolution-miles`.
- `checkpoints.jsonl` records server checkpoint URIs, training progress, and
  the W&B run ID. It does not contain model weights.

Override `--log_path`, `--stats.experiment_name`, and `--stats.trial_name` to
choose another run directory. The workflow derives its rollout directory from
these fields; `train.workflow_config.rollout_config.output_dir` would be
overridden. This YAML loader does not expand `${...}` expressions.
The logger and event sink create their output directories as needed.
Python console output goes to stderr; redirect or tee it explicitly if you
want a `client.log` file.

Full weight and optimizer checkpoints are saved on the miles-tinker server,
under its `--checkpoint-root`. The Babel YAML cannot move that storage. The
client requests full checkpoints initially, every 25 training steps, and at
normal completion or a graceful interrupt. Between full checkpoints it
refreshes the inference weights.

Reusing a run directory automatically resumes from its last checkpoint record,
even when `checkpoint.load_checkpoint_path` is null. Both the local metadata
and the corresponding server checkpoint must remain available. Use a new
`stats.trial_name` for a fresh run; changing the artifact root does not migrate
or resume runs from the old root.

Modal image and resource configuration is shared with
[the AReaL recipe](areal.md).
