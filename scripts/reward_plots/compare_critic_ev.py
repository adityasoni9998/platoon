"""Fetch unsampled W&B PPO history and overlay turn-level EV curves."""
import argparse
import csv
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean

RUNS = [
    ("CodeScout", "codescout-platoon-tinker", "222s5jnu"),
    ("CodeScout", "codescout-platoon-tinker", "4ot77kzn"),
    ("CodeScout", "codescout-platoon-tinker", "tlz73uxj"),
    ("CodeScout", "codescout-platoon-tinker", "ap4n7px7"),
]
ADV = "train/critic/raw_advantages_std"
RET = "train/critic/returns_std"


def fetch(spec, out, refresh):
    task, project, run_id = spec
    cache = out / f"{run_id}_history.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    import wandb
    run = wandb.Api(timeout=60).run(f"9998_team/{project}/{run_id}")
    train = run.config["train"]
    ppo = train["ppo"]
    if ppo["value_granularity"] != "turn" or ppo["advantage_normalization_granularity"] != "turn":
        raise ValueError(f"{run_id}: return and advantage statistics must both use turn masks")
    # No required-key filtering: retain all history to audit missing metrics.
    history = list(run.scan_history(page_size=1000))
    wanted = ["_step", "_timestamp", "train/step", ADV, RET,
              "train/critic/old_values_std", "train/critic/raw_advantages_std__count",
              "train/critic/returns_std__count", "train/critic/skipped"]
    result = dict(task=task, project=project, run_id=run_id, name=run.name, state=run.state,
                  batch_size=train["batch_size"], group_size=train["workflow_config"]["group_size"],
                  critic_minibatches=ppo["critic"]["num_minibatches"],
                  critic_learning_rate=float(ppo["critic"]["optimizer"]["learning_rate"]),
                  ppo={k: ppo[k] for k in ("value_granularity", "advantage_normalization_granularity",
                                          "gamma", "gae_lambda", "reward_mode")},
                  history=[{k: h[k] for k in wanted if k in h} for h in history])
    cache.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ev_codescout_number_search"))
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--fetch-only", action="store_true")
    args = parser.parse_args()
    if args.window < 1:
        parser.error("window must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        runs = list(pool.map(lambda spec: fetch(spec, args.output_dir, args.refresh), RUNS))
    if args.fetch_only:
        for run in runs:
            print(run["run_id"], run["batch_size"], run["group_size"], len(run["history"]))
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11.5, 6), layout="constrained")
    rows, summaries = [], []
    for run, color in zip(runs, ["#1f77b4", "#ff7f0e", "#d62728", "#17becf"]):
        label = f"{run['task']} · BS{run['batch_size']}, GS{run['group_size']}"
        if run.get("critic_minibatches", 1) != 1:
            label += f" · K={run['critic_minibatches']}, critic LR={run['critic_learning_rate']:g}"
        by_step, missing, duplicates = {}, [], []
        for h in run["history"]:
            if "train/step" not in h:
                continue
            step = int(h["train/step"])
            if ADV not in h or RET not in h:
                missing.append(step)
                continue
            a, r = h[ADV], h[RET]
            if not (isinstance(a, (float, int)) and isinstance(r, (float, int))
                    and math.isfinite(a) and math.isfinite(r) and r > 0):
                raise ValueError(f"Invalid variance inputs: {run['run_id']}, step {step}")
            for key in [ADV + "__count", RET + "__count"]:
                if h.get(key, 1) != 1:
                    raise ValueError(f"Cannot derive EV from averaged standard deviations: {key}")
            if step in by_step:
                duplicates.append(step)
            by_step[step] = dict(run_id=run["run_id"], label=label, step=step,
                                 explained_variance=1-(a/r)**2, advantage_std=a, return_std=r)
        selected = [by_step[s] for s in sorted(by_step)]
        if not selected:
            ax.plot([], [], color=color, label=f"{label} (awaiting metrics)")
            summary = {k: v for k, v in run.items() if k != "history"}
            summary.update(label=label, updates=0)
            summaries.append(summary)
            continue
        for row in selected:
            row["trailing_mean"] = fmean(r["explained_variance"] for r in selected
                                         if row["step"]-args.window < r["step"] <= row["step"])
        x = [r["step"] for r in selected]
        y = [r["explained_variance"] for r in selected]
        smooth = [r["trailing_mean"] for r in selected]
        ax.plot(x, y, color=color, linewidth=.8, alpha=.18)
        ax.plot(x, smooth, color=color, linewidth=2.3, marker="o" if len(x) == 1 else None, label=label)
        rows.extend(selected)
        missing_steps = sorted(set(range(x[0], x[-1]+1))-set(x))
        summary = {k: v for k,v in run.items() if k != "history"}
        summary.update(label=label, updates=len(selected), first_step=x[0], last_step=x[-1],
                       missing_metric_steps=sorted(set(missing)), gaps=missing_steps,
                       duplicate_steps=duplicates, latest_raw=y[-1], latest_trailing=smooth[-1],
                       first_16_mean=fmean(y[:16]), min_ev=min(y), max_ev=max(y),
                       negative_steps=[r["step"] for r in selected if r["explained_variance"]<0])
        summaries.append(summary)
    ax.axhline(0, color="0.5", linewidth=.8, linestyle="--")
    ax.set(title=f"PPO critic explained variance · {args.window}-step trailing mean",
           xlabel="Completed training batch (including warmup)", ylabel="Explained variance (turn-level, pre-update)")
    ax.set_xlim(min(r["step"] for r in rows), max(r["step"] for r in rows))
    ax.grid(alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="lower right", fontsize=10)
    for ext in ("png", "pdf"):
        fig.savefig(args.output_dir / f"explained_variance_comparison.{ext}", dpi=180)
    with (args.output_dir / "per_step.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(formula="1 - (raw_advantages_std / returns_std)^2", window=args.window,
                  precision="Full-precision unsampled W&B scan_history",
                  weighting="One observation per trained turn; one pre-update EV per training batch; "
                            "trailing mean weights batches equally.",
                  target="GAE lambda-return (gamma 0.99, lambda 0.9), before critic update",
                  runs=summaries)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
