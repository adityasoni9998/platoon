# CodeScout training with Modal

This plugin implements the repository-level localization recipe from the
[CodeScout paper](https://arxiv.org/abs/2603.17829). It supports Platoon's Areal
and Tinker training backends while running each OpenHands environment in an
isolated Modal Sandbox.

## Setup

Authenticate the Modal client once on the trainer host:

```bash
modal setup
```

Create either backend environment from the repository root:

```bash
uv sync --project plugins/codescout --extra areal
# Or: uv sync --project plugins/codescout --extra tinker
```

The plugin pins the OpenHands packages to
`adityasoni9998/software-agent-sdk@0d404d2d804bc412d69ce13edf2aa6b6bed329a2`.
`openhands-workspace[modal]` installs the required Modal client dependency.

## Required network setup

The OpenHands agent and its LLM client run inside Modal. The Areal configuration
uses the default inline agent mode and the standard `GroupRolloutWorkflow`. This
initial port assumes each worker proxy URL injected by Areal is already publicly
reachable from Modal. Port forwarding, endpoint rewriting, and tunnel lifecycle
management are intentionally deferred.

For Tinker there is one CodeScout proxy, so provide its complete public URL:

```bash
export CODESCOUT_TINKER_PROXY_HOST=0.0.0.0
export CODESCOUT_MODAL_MODEL_ENDPOINT="http://PUBLIC_HOST:PUBLIC_PORT/v1"
```

The tunnel or firewall must restrict access appropriately. Areal uses its
session-scoped bearer key. The current Tinker proxy is intended for a trusted
network and should not be exposed openly to the internet.

## Modal sandbox settings

The default image is:

```text
docker.io/adityasoni8/codescout-agent-server-modal-workspace:codescout-modal-source-minimal
```

It must contain the source agent-server at `/agent-server/.venv`, the CodeScout
system prompt at `/app/prompts_codescout/system_prompt.j2`, the custom
`LocalizationFinishTool`, Git, and ripgrep. Override it with
`CODESCOUT_AGENT_SERVER_IMAGE` when using another compatible build.

The runtime settings can be changed with these environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODAL_APP_NAME` | `codescout-agent-server` | Modal App owning the sandboxes |
| `MODAL_ENVIRONMENT` | unset | Modal environment name |
| `MODAL_SANDBOX_TIMEOUT` | `1800` | Maximum sandbox lifetime, including setup and rollout |
| `MODAL_IDLE_TIMEOUT` | unset | Optional Modal idle timeout |
| `MODAL_STARTUP_TIMEOUT` | `600` | Agent-server health-check deadline |
| `MODAL_CPU` | `1` | CPU request per sandbox |
| `MODAL_MEMORY` | `2048` | Memory in MiB per sandbox |
| `MODAL_CLOUD` | unset | Optional cloud selection |
| `MODAL_REGION` | unset | Optional region selection |
| `MODAL_REGISTRY_SECRET_NAME` | unset | Secret for pulling a private image |
| `MODAL_EXPECTED_SERVER_GIT_SHA` | unset | Optional server revision check |
| `MODAL_VERBOSE` | `0` | Enable Modal provisioning output |

Unlike Apptainer, this workspace does not bind a trainer-host directory into the
sandbox. Each rollout shallow-clones the dataset's repository branch into
`/workspace`. Sandboxes are terminated after rollouts; `keep_alive` is not
enabled. Host environment variables and credentials are not forwarded into the
sandbox by default.

Concurrency has direct Modal cost implications. The configs preserve their
source-branch concurrency values; review them before launching Modal sandboxes.

## Train with Areal

```bash
cd plugins/codescout
python -m areal.launcher.local \
  platoon/codescout/train.py \
  --config platoon/codescout/train_codescout_areal.yaml
```

The example retains the original CodeScout 1.7B RFT checkpoint and uses the
current `GroupRolloutWorkflow` API with Areal's online proxy mode.

## Train with Tinker

```bash
cd plugins/codescout
python platoon/codescout/train_tinker.py
```

Adjust the model, output paths, group size, and concurrency in the corresponding
YAML files before a production run.
