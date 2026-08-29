# CodeScout training with Sail

This plugin is the SailWorkspace variant of `plugins/codescout`. It preserves
the same CodeScout task, Areal/Tinker recipes, inline rollout mode, and Ogma
reverse-port forwarding while running each OpenHands environment in a Sailbox.

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
`adityasoni9998/software-agent-sdk@7dd27078ec2dba29e5a0a9d96c1f2d1a519c99a8`
from branch `sail_modal_workspace`. `openhands-workspace[sail]` installs the
Sail client (`sail>=0.7.1,<1`).

## Network setup

Areal remains in default inline mode. Once its four DP proxy workers are ready,
the trainer creates one SSH ControlMaster to `adityabs@ogma.lti.cs.cmu.edu`
with four independent reverse forwards. Only the LLM URL sent into each
Sailbox is replaced by its corresponding public Ogma URL; Areal session control
continues to use the internal proxy URLs.

This requires passwordless SSH, externally bound reverse forwards on Ogma, and
firewall access from Sailboxes. The forwards expose plain HTTP, authenticated
by Areal session bearer keys but not encrypted between Sail and Ogma.

For Tinker, expose its single proxy and provide the public URL:

```bash
export CODESCOUT_TINKER_PROXY_HOST=0.0.0.0
export CODESCOUT_SAIL_MODEL_ENDPOINT="http://PUBLIC_HOST:PUBLIC_PORT/v1"
```

## Sailbox settings

The default Sail-specific agent-server image is:

```text
docker.io/adityasoni8/codescout-agent-server-sail-workspace:codescout-sail-source-minimal
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
| `SAIL_STARTUP_TIMEOUT` | `600` | Agent-server listener/health deadline |
| `SAILBOX_SIZE` | `s` | Sail size ceiling: `s`, `m`, or `l` |
| `SAILBOX_MEMORY_GIB` | `2` | RAM ceiling in GiB |
| `SAILBOX_DISK_GIB` | `16` | Writable state-disk ceiling in GiB |
| `SAILBOX_PRIVATE` | `0` | Restrict operations to the creating Sail user |
| `SAILBOX_INBOUND_ALLOWLIST` | unset | Comma-separated HTTP ingress allowlist |
| `SAIL_WORKSPACE_OWNER` | `10001:10001` | Owner assigned to the workspace directory |
| `SAIL_EXPECTED_SERVER_GIT_SHA` | unset | Optional server revision check |

The size presets are ceilings rather than reservations: `s` provides 1 vCPU,
16 GiB default RAM, and 32 GiB default disk; `m` provides 4/32/128; and `l`
provides 8/64/256. Ongoing billing follows observed usage. This plugin selects
the `s` CPU tier and overrides its defaults to 2 GiB RAM and a 16 GiB writable
state disk, matching the requested Modal resource profile.

SailWorkspace has no overall lifetime parameter comparable to Modal's sandbox
timeout. The plugin explicitly terminates each Sailbox during rollout cleanup;
an abruptly killed trainer may leave Sailboxes requiring manual cleanup.

## Train with Areal

```bash
cd plugins/codescout_sail
export UV_PROJECT_ENVIRONMENT=/tmp/platoon-codescout-sail-venv
uv run --extra areal python -m platoon.codescout_sail.train \
  --config platoon/codescout_sail/train_codescout_areal.yaml \
  trial_name="$trial"
```

## Train with Tinker

```bash
cd plugins/codescout_sail
export UV_PROJECT_ENVIRONMENT=/tmp/platoon-codescout-sail-venv
uv run --extra tinker python -m platoon.codescout_sail.train_tinker
```
