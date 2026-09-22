# Train CodeScout SAIL with Tinker

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
`--train.max_training_steps 1` for a short check.

Each rollout runs in a separate process. As in legacy, its default timeout is
the episode timeout plus two hours for setup/cleanup; override with
`--train.workflow_config.rollout_config.extra.subprocess_timeout SECONDS`.
