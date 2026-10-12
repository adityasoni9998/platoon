# DAPO-Math-17K with Tinker PPO

This plugin feeds single-turn math trajectories into `GroupRolloutWorkflow` and
`PlatoonTinkerRLTrainer`. It uses the existing actor/critic training, async rollout,
metrics, and checkpoint code. There are no trainer overrides or monkey patches.

The reference is Miles' [DAPO PPO client](https://github.com/neulab/miles-tinker/blob/soni/hyperion/deploy/clients/dapo_ppo.py)
and its [launch recipe](https://github.com/neulab/miles-tinker/blob/soni/hyperion/docs/fork/tinker-ppo-client.md),
inspected at revision `de71e430d4de297ca7c3534f06e64433ced29b72`.

## Setup

From the repository root, enter the plugin directory and sync its environment:

```bash
cd plugins/dapo-math
uv sync --all-extras
```

The plugin includes the DeepScaler grader and math helpers copied from the Miles
revision above, with attribution and the upstream license. No Miles checkout is
needed. Configure the Tinker connection:

```bash
export TINKER_BASE_URL='http://your-server:10613'
export TINKER_API_KEY='your-key'
```

Set the URL in the environment, keeping YAML `tinker_base_url: null`: the existing
rollout proxy creates its own SDK service client and reads this environment
variable. A YAML-only URL does not configure that initial proxy client.

The server must support full-model actor/critic training, use
`--tinker-enable-critic`, and match `--rollout-temperature 1.0`. Model initialization,
GPU placement, offloading, and checkpoint retention remain server settings.

## Run

Run these commands from `plugins/dapo-math`. Choose a trial name using the current
UTC timestamp in Bash, then pass it as a YAML configuration override:

```bash
DAPO_TRIAL_NAME="tinker-ppo-platoon-client-$(date -u +%Y%m%d-%H%M%SZ)"
uv run python -m platoon.dapo_math.train_tinker \
  --stats.trial_name "$DAPO_TRIAL_NAME"
```

The dataset is downloaded through Hugging Face and read directly from its cache:
`zhuzilin/dapo-math-17k/dapo-math-17k.jsonl`. No separate dataset copy is written.
Dataset messages are passed through without adding instructions. The Qwen3
thinking renderer formats them for sampling.

Logs are written under
`/data/user_data/adityabs/miles_tinker_smoke/dapo-math-ppo/<trial_name>`.
The commands above produce `tinker-ppo-platoon-client-<YYYYMMDD-HHMMSSZ>`.
W&B uses the same name and logs to
[`9998_team/miles-amd-ppo`](https://forge.coreweave.com/wandb/9998_team/miles-amd-ppo).
Use your existing W&B login or set `WANDB_API_KEY` before launching.

For a small smoke run that reaches actor updates after critic warmup:

```bash
DAPO_TRIAL_NAME="tinker-ppo-platoon-client-$(date -u +%Y%m%d-%H%M%SZ)"
uv run python -m platoon.dapo_math.train_tinker \
  --stats.trial_name "$DAPO_TRIAL_NAME" \
  --train.max_training_steps 3 \
  --train.batch_size 2 \
  --train.num_concurrent_rollout_workflow_workers 2 \
  --train.workflow_config.group_size 2 \
  --train.workflow_config.rollout_config.inference_params.max_completion_tokens 1024 \
  --checkpoint.every 1
```

The shorter response limit can produce mostly zero rewards; it is a plumbing
check, not a learning-quality comparison. To disable W&B, pass
`--stats.wandb.mode disabled`. Rollout events include raw answers and completion
IDs, and metrics include `train/reward/correct` and `train/reward/truncated` in
addition to the trainer's standard metrics. No held-out evaluation is configured.

Resume in the same run directory by passing `--stats.trial_name <previous-trial-name>`,
or initialize another run using
`--checkpoint.load_checkpoint_path /path/to/ppo_checkpoints.jsonl`.

## Reference settings and differences

The YAML matches Qwen3-4B, 8 prompts × 8 responses, 8192 response tokens,
top-p 1, actor/critic learning rates `1e-6`/`1e-5`, Adam betas
`0.9`/`0.98`, epsilon `1e-8`, weight decay `0.1`, gradient clipping `1.0`, policy
ratio bounds `0.8`/`1.2`, value clipping `0.2`, gamma/lambda `1`, one critic-only
batch, 300 training batches, and checkpointing every 10 batches from the launch
recipe. Rewards use the copied Miles grader on raw sampled tokens, preserving the
`</think>` delimiter. Truncated responses are graded normally with zero terminal
bootstrap, matching the example.

Sampling temperature is set to `1.0` for this experiment, instead of the reference's
`0.8`; configure the server's rollout temperature to `1.0` as well.

These differences are left in place rather than changing core training code:

| Reference | This plugin with the existing trainer |
|---|---|
| Two disjoint 32-response optimizer batches from each 64-response rollout | One 64-response optimizer batch per role. Setting update epochs to two would reuse all 64 responses and is not equivalent. |
| Equal weight per response: each token gets `1 / (32 × response_length)` | Equal weight per action token across the batch. |
| Whitening with sample variance (`N−1`) | Whitening with population variance (`N`). |
| Synchronous by default; optional async mode prefetches one batch and waits for it before publishing weights | Existing GRPO async workers with `max_staleness: 1`; this is not an exact one-batch publication barrier. |
| Python shuffle with seed 42 and reference-DP sequence balancing | Existing trainer's dataset shuffle and arrival order; no client-side reference-DP balancing. |

For a native Miles Megatron comparison, `--global-batch-size 64` gives one optimizer
batch for 64 rollout responses, and `--calculate-per-token-loss` selects token-mean
aggregation for both actor and critic. The whitening difference is negligible at
these token counts. The Tinker DAPO client instead uses `--batch-size 64` for batch
size and hardcodes per-response weights; it has no aggregation CLI switch. Do not
pass `--calculate-per-token-loss` to the Tinker server, which expects clients to
provide normalized weights/advantages and rejects that flag.

This is a smoke-test task for the current PPO implementation, not an exact
reproduction of Miles' optimization schedule. Only static validation was performed;
the plugin has not been run.
