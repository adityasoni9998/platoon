# Reward plots

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
