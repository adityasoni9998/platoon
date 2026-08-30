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
`adityasoni9998/software-agent-sdk@29fafa4b5f0f41478b52f75b47423dfda99529d5`.
`openhands-workspace[modal]` installs the required Modal client dependency.

## Required network setup

The OpenHands agent and its LLM client run inside Modal. Areal remains in its
default inline agent mode. After Areal starts its four DP proxy workers, the
CodeScout trainer opens one background SSH ControlMaster to
`adityabs@ogma.lti.cs.cmu.edu`. It requests four independent reverse forwards
with remote port `0`, allowing Ogma to allocate an available public port for
each worker.

The resulting internal-proxy to Ogma URL map is serialized with the rollout
workflow. Areal session creation and interaction export continue to use the
internal proxy URL; only the LLM endpoint passed to OpenHands is replaced with
the corresponding `http://ogma.lti.cs.cmu.edu:<allocated-port>` URL. The SSH
connection and all four forwards are closed when training exits.

This requires:

- passwordless SSH authentication from the trainer host;
- Ogma's SSH server to allow externally bound reverse forwards
  (`GatewayPorts clientspecified` or `yes`);
- Ogma's firewall to admit the allocated ports from Modal.

The reverse forwards currently expose plain HTTP. Session bearer keys provide
authentication but not transport encryption between Modal and Ogma. Use TLS
termination on Ogma before treating this as suitable for an untrusted network.

For Tinker there is one CodeScout proxy, so provide its complete public URL:

```bash
export CODESCOUT_TINKER_PROXY_HOST=0.0.0.0
export CODESCOUT_MODAL_MODEL_ENDPOINT="http://PUBLIC_HOST:PUBLIC_PORT/v1"
```

The tunnel or firewall must restrict access appropriately. Areal uses its
session-scoped bearer key. The current Tinker proxy is intended for a trusted
network and should not be exposed openly to the internet.

## Modal sandbox settings

The default published Modal image is:

```text
docker.io__adityasoni8__codescout-agent-server-modal-workspace:29fafa4b
```

It must contain the source agent-server at `/agent-server/.venv`, the CodeScout
system prompt at `/app/prompts_codescout/system_prompt.j2`, the custom
`LocalizationFinishTool`, Git, and ripgrep. The workspace resolves it with
`modal.Image.from_name()`. Override it with `CODESCOUT_NAMED_AGENT_SERVER_IMAGE`
when using another compatible published Modal image.

Modal Sandbox V2 works through the same workspace code path. Opt into it in the
trainer process before launching a run:

```bash
export MODAL_SANDBOX_V2=1
```

Leave the variable unset (or set it to `0`) to use V1.

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
| `MODAL_SANDBOX_V2` | unset | Set to `1` to use Modal Sandbox V2 |
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

The Areal configuration translates the CodeScout Tinker recipe to the current
Areal schema and retains inline rollout execution.

## Train with Tinker

```bash
cd plugins/codescout
python platoon/codescout/train_tinker.py
```

Adjust the model, output paths, group size, and concurrency in the corresponding
YAML files before a production run.
