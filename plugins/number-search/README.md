# Number Search

Number Search is a small binary-search-style environment for quick Platoon training and rollout smoke tests. The agent guesses a hidden number and receives "too low", "too high", or "correct" feedback.

## Install

```bash
cd plugins/number-search
uv sync --extra areal
```

Use `--extra tinker` instead of `--extra areal` for Tinker experiments.

## Train

Tinker:

```bash
uv run python -m platoon.number_search.train_tinker \
  --config platoon/number_search/number_search_tinker.yaml
```

For Qwen3-4B-Instruct on the MI350 Miles-Tinker gateway:

```bash
uv run python -m platoon.number_search.train_tinker \
  --config platoon/number_search/number_search_qwen3_4b_instruct_tinker.yaml
```

AReaL:

```bash
uv run python3 platoon/number_search/train.py \
  --config platoon/number_search/nv_number_search_cispo_areal.yaml
```

Useful configs:

- `platoon/number_search/number_search_tinker.yaml`
- `platoon/number_search/number_search_qwen3_4b_instruct_tinker.yaml`
- `platoon/number_search/number_search_cispo_tinker.yaml` — Qwen3-4B-Instruct CISPO with outcome-only rewards.
- `platoon/number_search/number_search_ppo_tinker.yaml` — Qwen3-4B-Instruct PPO with turn-level rewards, critic values, and advantage whitening (`gamma=0.99`, `gae_lambda=0.9`). Requires a critic-enabled Tinker server.
- `platoon/number_search/number_search_areal.yaml`
- `platoon/number_search/nv_number_search_areal.yaml`
- `platoon/number_search/nv_number_search_cispo_areal.yaml`

## Environment

Each task contains a range and target number. The only action is `guess(n: int)`. By default, rewards are `1.0` for the correct guess and `0.0` otherwise.

The PPO config enables reward shaping: `+0.2` for either middle integer of the
current inclusive range, `-0.2` for an out-of-range guess, and zero otherwise.
Feedback narrows the range after scoring. A correct guess adds `+1` on that
same turn, exactly once. Both modes permit only one integer guess per agent turn.
Additional calls raise an error before producing feedback or changing reward;
the first guess's reward and success, if any, are retained. An invalid-type
attempt also consumes the turn's guess allowance. The allowance resets on the
next agent turn. There is no per-turn
penalty: only an integer guess outside the current valid range receives `-0.2`.
The midpoint and out-of-range rewards are configurable under `reward_shaping`.
The CISPO config leaves shaping disabled. Tinker training uses the full
50,000-task training pool.

Both configs log **`train/root_success_rate/avg`** as binary task success.
With shaping enabled, `train/task_reward/avg` instead
contains the total shaped return. `train/root_shaping_return/avg` reports its
shaping component separately. Success and shaping components are also stored in
each step's `misc.reward_misc` in the rollout JSONL logs. Eval uses the same
metrics under `eval/` when enabled.

Datasets live next to the package:

- `platoon/number_search/number_search_train.jsonl`
- `platoon/number_search/number_search_val.jsonl`
