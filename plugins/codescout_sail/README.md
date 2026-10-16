# CodeScout with Sail

This plugin is the SailWorkspace variant of `plugins/codescout`. It preserves
the same CodeScout task and Areal/Tinker recipes while running each OpenHands
environment in a Sailbox. Areal uses Ogma forwarding; Tinker supports Modal
or Ogma forwarding.

## Setup

Authenticate the Sail client once on the trainer host:

```bash
sail auth login
```

Create either backend environment from the repository root. To keep the virtual
environment off `/project`, place it under `/tmp`:

```bash
UV_PROJECT_ENVIRONMENT=/tmp/platoon-codescout-sail-venv \
  uv sync --project plugins/codescout_sail --extra areal
# Or use --extra tinker.
```

The plugin pins the OpenHands packages to
`adityasoni9998/software-agent-sdk@d3fd1c497860fdcba49e3304f348b10a4f1c370b`
from branch `sail-workspace-hardened`. `openhands-workspace[sail]` installs the
Sail client (`sail>=0.11.0,<0.12`).

## Network setup

Areal remains in default inline mode. Once its four DP proxy workers are ready,
the trainer creates one SSH ControlMaster to `adityabs@ogma.lti.cs.cmu.edu`
with four independent reverse forwards. Only the LLM URL sent into each
Sailbox is replaced by its corresponding public Ogma URL; Areal session control
continues to use the internal proxy URLs.

This requires passwordless SSH, externally bound reverse forwards on Ogma, and
firewall access from Sailboxes. The forwards expose plain HTTP, authenticated
by Areal session bearer keys but not encrypted between Sail and Ogma.

Tinker automatically creates one Modal router and forwards its local proxy to
a public HTTPS endpoint for all Sailbox rollouts. The launcher checks both the
local proxy and public endpoint before starting rollouts. Traffic follows:

```text
Sailbox -> Modal HTTPS endpoint -> reverse SSH -> local Tinker proxy -> Tinker
```

The Tinker config selects this route by default:

```yaml
inference_tunnel:
  backend: modal
  modal:
    app_name: codescout-sail-inference-router
    cpu: 0.125
    memory: 128
    timeout: 86400
```

Install with `--extra tinker`, which includes the Modal SDK. The host needs
configured Modal credentials plus `ssh` and `ssh-keygen`. `MODAL_PROFILE`
selects a profile, for example `neulab`. Temporary SSH keys, host-key checking,
public endpoint health checks, and resource cleanup follow the legacy plugin's
design, via shared helpers under `platoon/inference`.

The proxy binds to `127.0.0.1` by default. The launcher passes the public URL
to the agents; the Tinker service credential stays on the trainer. The existing
per-rollout session header continues to separate recorded interactions.

Normal completion, startup failure, and cancellation close the SSH connection,
terminate the router, stop the proxy, and restore the trainer's model settings.
The router's 24-hour lifetime must cover the run; it is not automatically
renewed. Set `--inference_tunnel.backend ogma` to use automatic Ogma forwarding.

The launcher always creates the configured tunnel automatically.
`CODESCOUT_TINKER_PROXY_HOST` can override the local bind address when needed.
Forwarding runs on the trainer and does not require rebuilding the CodeScout
agent-server image.

## Sailbox settings

The default Sail-specific agent-server image is:

```text
docker.io/adityasoni8/codescout-agent-server-sail-workspace@sha256:4cbd6c9b815dab1cb08a5fe85b1b7d1413bd1b6726d873f9b25003be0a765400
```

Override it without changing code:

```bash
export CODESCOUT_AGENT_SERVER_IMAGE="NEW_IMAGE_URL"
```

The image must provide the source agent-server at `/agent-server/.venv`, the
CodeScout system prompt at `/app/prompts_codescout/system_prompt.j2`, the custom
`LocalizationFinishTool`, Git, and ripgrep.

| Variable | Default | Meaning |
| --- | --- | --- |
| `SAIL_APP_NAME` | `codescout-agent-server` | Sail App owning Sailboxes |
| `SAILBOX_NAME` | generated | Optional fixed Sailbox name |
| `SAIL_IMAGE_ARCHITECTURE` | unset | Optional `amd64` or `arm64` image architecture |
| `SAIL_IMAGE_BUILD_TIMEOUT` | `1800` | OCI image import/build timeout in seconds |
| `SAIL_FORCE_IMAGE_BUILD` | `0` | Refresh a mutable image tag before creation |
| `SAIL_CREATE_TIMEOUT` | `600` | Sailbox provisioning timeout; `0` is unbounded |
| `SAIL_STARTUP_TIMEOUT` | `600` | Separate listener and health deadlines |
| `SAILBOX_SIZE` | `s` | Sail size ceiling: `s`, `m`, or `l` |
| `SAILBOX_MEMORY_GIB` | `2` | RAM ceiling in GiB |
| `SAILBOX_DISK_GIB` | `8` | Writable state-disk ceiling in GiB |
| `SAILBOX_PRIVATE` | `0` | Restrict operations to the creating Sail user |
| `SAILBOX_INBOUND_ALLOWLIST` | unset | Comma-separated HTTP ingress allowlist |
| `SAIL_WORKSPACE_OWNER` | `10001:10001` | Owner assigned to the workspace directory |
| `SAIL_EXPECTED_SERVER_GIT_SHA` | unset | Optional server revision check |

The size presets are ceilings rather than reservations: `s` provides 1 vCPU,
16 GiB default RAM, and 32 GiB default disk; `m` provides 4/32/128; and `l`
provides 8/64/256. Ongoing billing follows observed usage. This plugin selects
the `s` CPU tier and overrides its defaults to 2 GiB RAM and an 8 GiB writable
state disk, the smallest supported limits.

SailWorkspace has no overall lifetime parameter comparable to Modal's sandbox
timeout. The plugin explicitly terminates each Sailbox during rollout cleanup;
an abruptly killed trainer may leave Sailboxes requiring manual cleanup.

The import, provisioning, listener, health, and episode timeouts are separate
budgets. Workspace directory commands have a 60-second limit. A 600-second
episode timeout does not include workspace provisioning. Failed termination
retains the SDK handle and sandbox ID for another cleanup attempt.

## Train with Areal

```bash
cd plugins/codescout_sail
export UV_PROJECT_ENVIRONMENT=/tmp/platoon-codescout-sail-venv
uv run --extra areal python -m platoon.codescout_sail.train \
  --config platoon/codescout_sail/train_codescout_areal.yaml \
  trial_name="$trial"
```

## Train with Tinker

For PPO with one critic-only warmup step, use
[`train_codescout_ppo_tinker.yaml`](platoon/codescout_sail/train_codescout_ppo_tinker.yaml).
The [PPO launch instructions](tinker.md#ppo-on-swe-smith-code-search) describe
the critic-enabled server requirement, optimizer settings, and unchanged
CodeScout F1/turn-limit reward.

Each rollout runs in a fresh subprocess, following issue-resolution legacy.
The trainer retains the shared Tinker proxy and Modal tunnel. A blocking
Sailbox request therefore stalls only its rollout process. The parent enforces
`train.workflow_config.rollout_config.extra.subprocess_timeout` (seconds),
defaulting to the episode timeout plus 7,200 seconds for setup and cleanup.
Cancellation requests cleanup before killing an unresponsive child after
180 seconds; a forced kill cannot guarantee remote Sailbox cleanup.

The YAML includes the upstream Tinker service URL and all training settings.
Export `SAIL_API_KEY`, `TINKER_API_KEY`, and `WANDB_API_KEY` first. Choose the
Sail key for the intended organization explicitly: without `SAIL_API_KEY`,
the SDK falls back to the host's saved Sail login, which may belong to a
different organization. The service must support full-parameter training
(`lora_rank: 0`). Configure Modal credentials. From the repository root:

```bash
cd plugins/codescout_sail
export UV_PROJECT_ENVIRONMENT=/tmp/platoon-codescout-sail-venv
export MODAL_PROFILE=neulab
export MODAL_SANDBOX_V2=1
export OPENHANDS_SUPPRESS_BANNER=1
uv sync --locked --extra tinker

: "${TINKER_API_KEY:?Export the upstream Tinker API key}"
: "${SAIL_API_KEY:?Export the Sail API key for the intended organization}"
uv run --locked --extra tinker python -m platoon.codescout_sail.train_tinker \
  --config platoon/codescout_sail/train_codescout_fft_tinker.yaml
```

This uses CISPO, learning rate `1e-6`, batch size 32, rollout group size 8,
maximum staleness 1, and Qwen/Qwen3-4B-Instruct-2507. The 42,048-token total
context budget reserves 2,048 tokens for output (40,000 for the prompt).
32 workflow workers allow up to 256 simultaneous Sailbox rollouts. The
default runs one epoch of the current CodeScout dataset loader, which loads
`adityasoni17/SWE-smith-py-code-search`
Asynchronous rollout workers may start extra rollouts ahead of that step.

W&B uploads metrics to `codescout-platoon-tinker` (`mode: online`). Logs,
rollout events, W&B files, and `checkpoints.jsonl` go under
`/data/user_data/adityabs/miles_sail/codescout-platoon-tinker/qwen3_4b_cispo_sail/`.
Weight and optimizer checkpoints remain on the Tinker service; to store those
under the same root, configure the server with
`--checkpoint-root /data/user_data/adityabs/miles_sail/checkpoints` on its host.
The client YAML cannot change that server setting. Reusing the configured
trial directory automatically resumes from its checkpoint metadata; change
`stats.trial_name` in the YAML for a fresh run.

The YAML's `tinker_base_url` is separate from the automatically generated
agent-facing proxy URL. Update `tinker_base_url` in the YAML if the Tinker
service URL changes.
