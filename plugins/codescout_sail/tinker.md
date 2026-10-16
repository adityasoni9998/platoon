# Train CodeScout SAIL with Tinker

## PPO on SWE-smith code search

Use `platoon/codescout_sail/train_codescout_ppo_tinker.yaml` for
`Qwen/Qwen3-4B-Instruct-2507` with full-parameter PPO. The dataset loader uses
`adityasoni17/SWE-smith-py-code-search` (all rows with a problem statement).
Rollout settings follow [reference run rfchcgg4](https://forge.coreweave.com/wandb/9998_team/codescout-platoon-tinker/runs/rfchcgg4):
batch size 32, group size 8, 32 workers (up to 256 simultaneous Sailboxes),
staleness 1, six rollout steps, a 42,048-token context, and 2,048 output tokens.
Training runs for one epoch, with paired actor/critic checkpoints every 25 batches.

The actor and critic use the number-search PPO optimizer settings: learning
rates `1e-6` and `1e-5`, respectively, Adam betas `(0.9, 0.98)`, epsilon `1e-8`,
weight decay `0.1`, and gradient clipping `1.0`. PPO uses ratio clipping
`[0.8, 1.2]`, value clipping `0.2`, turn-level values and batch advantage
whitening over turns, `gamma=0.99`, and `gae_lambda=0.9`.
The first accepted batch performs exactly one critic optimizer step and no
actor step. Each later batch performs one critic step followed by one actor step.

The existing CodeScout reward is assigned at the terminal turn: file F1 +
module F1 + entity F1, plus `1` for finishing in exactly four model turns.
The six-step rollout cap remains separate from that four-turn bonus.
No number-search reward shaping is enabled.

The Miles-Tinker server must run with `--tinker-enable-critic` and support
full-parameter training. Export `TINKER_BASE_URL`, `TINKER_API_KEY`,
`SAIL_API_KEY` for the intended Sail organization, and `WANDB_API_KEY`.
Configure Modal credentials as for the CISPO recipe below. From the repository root:

```bash
cd plugins/codescout_sail
export UV_PROJECT_ENVIRONMENT=/tmp/platoon-codescout-sail-venv
export MODAL_PROFILE=neulab
export MODAL_SANDBOX_V2=1
export OPENHANDS_SUPPRESS_BANNER=1
uv sync --locked --extra tinker

: "${TINKER_BASE_URL:?Export the critic-enabled Tinker service URL}"
: "${TINKER_API_KEY:?Export the upstream Tinker API key}"
: "${SAIL_API_KEY:?Export the Sail API key for the intended organization}"
trial="codescout-sail-ppo-$(date +%Y%m%d-%H%M%S)"
uv run --locked --extra tinker python -m platoon.codescout_sail.train_tinker \
  --config platoon/codescout_sail/train_codescout_ppo_tinker.yaml \
  --tinker_base_url "$TINKER_BASE_URL" \
  --stats.trial_name "$trial"
```

For a two-batch check covering both critic warmup and the first actor update,
add `--train.num_epochs null --train.max_training_steps 2` to set the duration
solely by batch count.
Asynchronous workers may start additional rollouts ahead of the final batch.
Use a fresh trial directory for PPO; resume only from paired
`ppo_checkpoints.jsonl` metadata, not the reference run's CISPO checkpoints.

## CISPO reference recipe

Requires `uv`, `ssh`, `ssh-keygen`, configured Modal credentials, and exported
`SAIL_API_KEY`, `TINKER_API_KEY`, `WANDB_API_KEY`.

From the repository root:

```bash
cd plugins/codescout_sail
export UV_PROJECT_ENVIRONMENT=/tmp/platoon-codescout-sail-venv
export MODAL_PROFILE=neulab
export MODAL_SANDBOX_V2=1
export OPENHANDS_SUPPRESS_BANNER=1
uv sync --locked --extra tinker
trial="codescout-sail-tinker-$(date +%Y%m%d-%H%M%S)"
uv run --locked --extra tinker python -m platoon.codescout_sail.train_tinker \
  --config platoon/codescout_sail/train_codescout_fft_tinker.yaml \
  --tinker_base_url "$TINKER_BASE_URL" \
  --stats.trial_name "$trial"
```

Change `MODAL_PROFILE` to your configured profile. The Modal tunnel is automatic.
The CLI URL overrides the YAML; exporting the variable alone does not.
The service must support `lora_rank: 0`. Each launch uses a timestamped trial name. Add
`--train.num_epochs null --train.max_training_steps 1` for a short CISPO check.

Each rollout runs in a separate process. As in legacy, its default timeout is
the episode timeout plus two hours for setup/cleanup; override with
`--train.workflow_config.rollout_config.extra.subprocess_timeout SECONDS`.
