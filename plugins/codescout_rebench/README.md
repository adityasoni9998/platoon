# CodeScout on SWE-rebench with AReaL

This Platoon plugin ports the CodeScout task from
`babel-compute-node:/home/adityabs/hyperion_modal/plugins/codescout_rebench`
to the AReaL backend used by `plugins/issue-resolution-rebench`. It trains on
`adityasoni17/SWE-rebench`, split `filtered`, with a separate Modal V2 sandbox
for each trajectory.

## Training recipe

| Setting | Default |
| --- | --- |
| Model | `Qwen/Qwen3-4B-Instruct-2507` |
| Batch size | 32 tasks |
| Group size | 8 trajectories per task; 256 trajectories per full batch |
| Loss | CISPO, clipping thresholds 0 and 5 |
| Learning rate | `1e-6`, constant schedule |
| Off-policy limit | `rollout.max_head_offpolicyness: 1` |
| Concurrent task groups | 32; up to 256 sandbox trajectories |
| Actor | `fsdp:d2p1t1c2` |
| Inference | `sglang:d4p1t1` |
| Hardware | One node with eight GPUs, as in issue-resolution-rebench |
| Context | 40,000 input tokens + 2,048 completion tokens |
| Sampling | Temperature 1.0, independent random seed per trajectory |
| Turn budget | Eight turns in the system prompt; ten SDK turns allowed |
| Episode / conversation timeout | 600 seconds |

Reward preserves the Babel recipe: file F1 + module F1 + entity F1, plus 1
when the model uses at most eight turns. Parallel tool calls from one response
count as one turn. The turn bonus does not require a valid finish submission;
turns nine and ten can earn localization reward but no bonus. A valid submission
comes from exactly one `LocalizationFinishAction`. Empty gold sets score zero.
There is no patch/test-execution evaluator to deploy for this localization task.

As in issue-resolution-rebench, the trainer uses `PlatoonArealRLTrainer` and
inline `GroupRolloutWorkflow` execution, records completions through AReaL
proxy sessions, masks observations from the loss, and sets `filter_errors=False`.
Groups with identical rewards remain in accounting but are masked out before
model training by the shared backend. The plugin uses AReaL, with no Tinker
dependency or proxy.

## Modal workspaces and runtime assets

The per-instance named images and SDK pin (`5acdf05ae2fc6224f92114db6f557b181cd0f280`)
match issue-resolution-rebench. Each sandbox already contains `/testbed`;
startup resets it to its existing `HEAD` and removes untracked files.

The custom system prompt is read from `prompts/system_prompt.j2` on the trainer
and serialized inline with the agent. Before creating the conversation,
`runtime_assets.py` uploads the two-file `custom_tools.localization_finish`
package into the server's discovered site-packages directory and verifies its
import using isolated Python. It also installs and verifies ripgrep. This
requires access to the image's apt repositories, but no image rebuild or
publication. The agent receives only terminal and localization-finish tools;
condensation is disabled. Sandboxes are cleaned up after each rollout.

| Environment variable | Default |
| --- | --- |
| `CODESCOUT_REBENCH_MODAL_APP_NAME` | `codescout-rebench-agent-server` |
| `MODAL_ENVIRONMENT` | Default Modal environment |
| `MODAL_SANDBOX_TIMEOUT` | Episode timeout + 300 seconds |
| `MODAL_IDLE_TIMEOUT` | Unset |
| `MODAL_STARTUP_TIMEOUT` | 600 seconds |
| `MODAL_CPU` | 0.125 |
| `MODAL_MEMORY` | 288 MiB |
| `MODAL_CLOUD`, `MODAL_REGION` | Unset |
| `MODAL_EXPECTED_SERVER_GIT_SHA` | Pinned SDK SHA; empty disables the check |
| `MODAL_VERBOSE` | 0 |

Modal Sandbox V2 is enabled in code, matching issue-resolution-rebench.
All usable dataset rows are retained by default, including empty `file_changes`.
Set `CODESCOUT_REBENCH_SKIP_EMPTY_GOLD=1` to omit empty gold annotations, or
`CODESCOUT_REBENCH_MAX_INSTANCES` to cap the dataset for a smoke test.

## Network routing

The trainer reaches the public Modal agent-server endpoint directly. Only LLM
traffic from Modal needs Ogma to reach the private AReaL inference proxies:

```text
Trainer --------------------------------> Modal agent server
Modal LLM client --> Ogma SSH forward --> AReaL proxy --> SGLang
Trainer/workflow --> internal AReaL proxy (session control and interaction export)
```

After AReaL starts its four proxy workers, the launcher opens one SSH
ControlMaster and requests an automatically allocated public Ogma port per
proxy. The serialized workflow replaces only the LLM endpoint, preserving the
worker's session bearer key. Each forward is health-checked and all forwards
close when training exits. No fixed public model URL is needed.

This uses the same noninteractive SSH access and externally bound reverse
forwarding setup as issue-resolution-rebench. Defaults are
`adityabs@ogma.lti.cs.cmu.edu` and `ogma.lti.cs.cmu.edu`; override with
`PLATOON_OGMA_SSH_TARGET` and `PLATOON_OGMA_PUBLIC_HOST`.

## Setup and launch

From the repository root (all plugin virtual environments live under `/tmp`):

```bash
export UV_PROJECT_ENVIRONMENT=/tmp/adityabs-codescout-rebench/areal-venv
export UV_CACHE_DIR=/tmp/adityabs-codescout-rebench/uv-cache
uv sync --no-active --project plugins/codescout_rebench --extra areal --locked
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
modal setup
```

Configure W&B credentials and noninteractive SSH access as for
issue-resolution-rebench. Then launch from any directory:

```bash
/project/flame/adityabs/hyperion_modal/plugins/codescout_rebench/run_areal.sh
```

The launcher uses the prepared environment, sets the required SGLang/Modal
runtime variables, clears the FlashInfer build cache, and generates a trial name
containing the Git revision and UTC timestamp. Existing `HF_HOME` and
`UV_PROJECT_ENVIRONMENT` overrides are preserved. Additional arguments are
AReaL `key=value` config overrides.

A small training smoke run, still using the default eight-GPU layout:

```bash
CODESCOUT_REBENCH_MAX_INSTANCES=2 plugins/codescout_rebench/run_areal.sh \
  total_train_epochs=1 train_dataset.batch_size=2 workflow_config.group_size=2 \
  rollout.max_concurrent_rollouts=2 stats_logger.wandb.mode=disabled
```

Unset the dataset cap for the full run.

`smoke_rollout.py` is also available for a single rollout against an existing
public OpenAI-compatible endpoint. It checks prompt transport, custom finish
execution, and localization reward, and saves the trajectory:

```bash
python -m platoon.codescout_rebench.smoke_rollout \
  --endpoint http://PUBLIC_HOST:PORT/v1 --api-key "$SMOKE_API_KEY" \
  --minimum-localization-reward 0
```

The minimum defaults to 3 (perfect localization); use 0 to check integration
independently of model accuracy. The endpoint must support `/models` as well as
chat completions. For the AReaL session/recording path, use the training smoke run.

See [VALIDATION.md](VALIDATION.md) for the local test results, live Modal/Ogma
check, and the limits of the verification performed during this port.
