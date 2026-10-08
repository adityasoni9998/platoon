# Reward plots

## Resolution probability by auxiliary-reward bucket

Plot `P(issue resolved | auxiliary reward in bucket)` for one or more runs:

```bash
uv run scripts/reward_plots/plot_reward_buckets.py \
  /path/to/dense_run /path/to/sparse_run \
  --labels "Dense ORM" "Sparse ORM" \
  --skip-last-steps 2 --overlay --output-dir artifacts/reward_buckets
```

Input directories may be run roots, `rollouts/train`, or `train_rollout`.
Localization (overall, file, module, function) uses width-0.2 buckets;
JSON-format and agent-error rewards use binary buckets; file-editor/str-replace
reward uses width-0.25 buckets. Buckets include the left endpoint and exclude
the right endpoint, except the final bucket includes 1.

Each bar is resolved trajectories divided by trajectories in that bucket.
Labels show the probability and sample count; whiskers are 95% task-cluster
bootstrap intervals, keeping repeated attempts on the same issue together.
Dotted lines show the baseline resolution probability among trajectories with
that metric available. These are descriptive associations, not estimates of
the causal benefit of training on an auxiliary reward.

Use `--overlay` to additionally create `comparison_all` and `comparison_nonempty`
PNG/PDF grids with side-by-side bars for each run in distinct colours.
The default produces separate all-trajectory and nonempty-patch views per run,
in PNG and PDF. It also saves `buckets.csv`, `rollouts.csv`, and `summary.json`.
Only completed root trajectories with a nonempty root reward dictionary and
a recorded binary label are included. Missing/invalid auxiliary values are
excluded per metric rather than defaulted to zero. Unknown patch status is
excluded from the nonempty view. Repeated attempts are retained and counts
are rollout-weighted; there is no smoothing or per-step averaging.

To redraw a saved per-rollout snapshot without reading the event logs:

```bash
uv run scripts/reward_plots/plot_reward_buckets.py \
  --rollout-csv artifacts/reward_buckets/rollouts.csv \
  --output-dir artifacts/reward_buckets_redraw
```

The localization-audit `rollouts.csv` uses the same input column names and can
also be read. Step-aggregated reward-grid `metrics.csv` cannot be used to
reconstruct conditional probabilities. Optional controls include
`--population all|nonempty|both`, inclusive `--min-step` / `--max-step`,
`--bootstrap-samples` (default 1000; zero disables intervals), `--seed`,
`--workers`, and `--y-max`. `--skip-last-steps` defaults to zero and removes the
last available steps separately per run before applying the step range.

## Rewards over training steps

From the repository root, generate the vertical 11×3 figure:

```bash
uv run scripts/reward_plots/plot.py /path/to/train_rollout
```

Defaults: centered 16-step averages, skip the two highest-numbered training-step directories, all remaining groups retained. Missing rewards default to 0, except P2P fail rate defaults to 1. The figure displays P2P **pass** rate (`1 - fail rate`).

Overall reward is pinned to the first row and uses the saved `trajectory_finished.reward` (not a recomputed component sum). Its Max@8 and non-zero advantages are computed from these total rewards; missing totals default to 0. Other rows are sorted by descending overall rollout-weighted mean reward. Columns show mean reward, Max@8 (mean of within-group maxima), and the fraction of rollouts with non-zero component-wise advantages. Advantages are component reward minus its group mean; the non-zero tolerance is `1e-10`. All axes span 0–1. The legend is at the top and training-step tick labels appear on every subplot.

Groups are identified by training-step directory and task ID. All recorded rollouts are used, including zero-variance groups. Groups with fewer than eight recorded rollouts use those present without padding. Centered windows contain eight preceding steps, the current step, and seven following steps; edges use available steps. Smoothing weights per-step statistics equally.

Optional arguments: `--window 8`, `--skip-last-steps 0`, `--output-dir /path/to/output`, and `--horizontal`.

To redraw the same snapshot without rereading trajectories, pass its output directory:

```bash
uv run scripts/reward_plots/plot.py /path/to/group_reward_plots_centered_k16
```

When redrawing, the saved smoothing window and skipped steps are retained. The default figure is `all_rewards_grid.png`; horizontal output is `all_rewards_grid_horizontal.png`. Older saved CSVs without overall reward must be regenerated from raw rollouts. Raw-rollout processing also saves per-component PNGs, `metrics.csv`, `group_rewards.jsonl`, and `summary.json` alongside the figure. The individual renderer can also produce three category grids:

```bash
uv run scripts/reward_plots/plot_reward_grids.py /path/to/group_reward_plots_centered_k16
```

Overall localization is the per-rollout arithmetic mean of file, module, and function F1 (missing F1 values default to 0). Its Max@8 is computed after averaging the three F1s within each rollout, then taking the group maximum.

To create a separate running-average-only figure with automatic per-subplot y-limits:

```bash
uv run scripts/reward_plots/plot.py /path/to/group_reward_plots_centered_k16 --smooth-only --dynamic-ylim
```

This writes `all_rewards_grid_smooth_only_dynamic_ylim.png`, preserving the default figure.

## Dense/sparse ORM comparison and restarts

`plot_orm_comparison.py DENSE_RUN SPARSE_RUN --output-dir OUTPUT_DIR` samples
at most eight eligible rollouts without replacement for each `(training step,
task ID)` group. This prevents repeated rollout batches from trainer restarts
from turning max@8 into max@16 or max@24. Selection happens after completion
and optional missing-reward/infrastructure-error filtering. The same selected
rollouts are used for every reward component and for mean reward, max@8, and
nonzero advantage. Groups with fewer than eight eligible members keep them all.

Sampling is reproducible for unchanged group membership, independent of file
iteration order, using `--group-sample-seed` (default 42) and a seed derived
separately for each run/step/task. Changing the seed selects a different sample;
cache entries are separated by seed. `summary.json` reports eligible counts
before sampling and `rollouts_excluded_by_group_sampling`. Existing commands
need no new flag to apply this cap; rerun them to rebuild old pooled metrics.
