# /// script
# dependencies = ["matplotlib", "wandb"]
# ///
"""Overlay critic EV from W&B and turn-weighted group-baseline EV from JSONL."""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean, pstdev

RUNS = [
    ("CodeScout", "codescout-platoon-tinker", "222s5jnu"),
    ("CodeScout", "codescout-platoon-tinker", "4ot77kzn"),
    ("CodeScout", "codescout-platoon-tinker", "tlz73uxj"),
    ("CodeScout", "codescout-platoon-tinker", "ap4n7px7"),
    ("CodeScout", "codescout-platoon-tinker", "a221kfp1"),
    ("CodeScout", "codescout-platoon-tinker", "abzvwgct"),
    ("CodeScout", "codescout-platoon-tinker", "70v8xbkq"),
    ("CodeScout", "codescout-platoon-tinker", "kexcf96w"),
]
GROUP_RUN = Path("/data/user_data/adityabs/miles_sail/codescout-platoon-tinker/"
                 "codescout-sail-cispo-b32-g8-20261024-124001")
ADV = "train/critic/raw_advantages_std"
RET = "train/critic/returns_std"


def read_group_rollout(path):
    """Count unique LLM completions, excluding the initial environment-only step."""
    roots, tasks, turns, finished = set(), {}, defaultdict(set), {}
    with path.open() as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # A live file can end with an incomplete line.
            kind = event.get("type")
            if kind == "trajectory_created":
                trajectory = event["trajectory"]
                if trajectory.get("parent_info") is None:
                    roots.add(trajectory["id"])
            elif kind == "trajectory_task_set":
                tasks[event["trajectory_id"]] = event["task"]["id"]
            elif kind == "trajectory_step_added":
                completion = event["step"].get("misc", {}).get("action_misc", {}).get("completion_id")
                if completion:
                    turns[event["trajectory_id"]].add(completion)
            elif kind == "trajectory_finished":
                finished[event["trajectory_id"]] = event.get("reward")
    return [{"id": tid, "task": tasks.get(tid), "turns": len(turns[tid]),
             "reward": finished.get(tid)} for tid in roots]


def group_variances(groups, group_size):
    """Compute population moments over turns, with an unweighted rollout baseline."""
    rewards, advantages = [], []
    audit = Counter()
    for group in groups:
        # No group invocation ID is logged. Do not guess how to partition repeated tasks.
        if len(group) != group_size:
            audit["incomplete_or_ambiguous_groups_excluded"] += 1
            continue
        if any(not row["turns"] or isinstance(row["reward"], bool)
               or not isinstance(row["reward"], (int, float))
               or not math.isfinite(row["reward"]) for row in group):
            audit["unfinished_or_invalid_groups_excluded"] += 1
            continue
        baseline = fmean(row["reward"] for row in group)
        for row in group:
            rewards.extend([row["reward"]] * row["turns"])
            advantages.extend([row["reward"] - baseline] * row["turns"])
        audit["groups"] += 1
        audit["rollouts"] += len(group)
    audit["turns"] = len(rewards)
    if not rewards:
        return None, dict(audit)
    return (pstdev(advantages), pstdev(rewards)), dict(audit)


def group_history(source, group_size):
    history, audits = [], {}
    folders = sorted((source / "rollouts/train").glob("[0-9]*"), key=lambda p: int(p.name))
    with ThreadPoolExecutor(max_workers=8) as pool:
        for folder in folders:
            groups, seen = defaultdict(list), set()
            for records in pool.map(read_group_rollout, sorted((folder / "events").glob("*.jsonl"))):
                for row in records:
                    if row["id"] not in seen and row["task"] is not None:
                        seen.add(row["id"])
                        groups[row["task"]].append(row)
            moments, audit = group_variances(groups.values(), group_size)
            audits[folder.name] = audit
            if moments is not None:
                a, r = moments
                if r == 0:
                    audit["undefined_ev_zero_return_variance"] = True
                    continue
                history.append({"train/step": int(folder.name), ADV: a, RET: r})
    return history, audits


def fetch(spec, out, refresh):
    task, project, run_id = spec
    cache = out / f"{run_id}_history.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    if run_id == "a221kfp1":
        # This run uses 32 tasks x 8 rollouts with an ordinary group-mean baseline.
        # Read rewards and turns locally; W&B is unnecessary even for metadata.
        history, audit = group_history(GROUP_RUN, group_size=8)
        result = dict(task=task, project=project, run_id=run_id, name=GROUP_RUN.name,
                      estimator="group", actor_loss="cispo", batch_size=32,
                      group_size=8, history=history, audit=audit,
                      source=str(GROUP_RUN), x_axis="Sampling policy version",
                      target="Final ORM reward repeated once per unique agent completion",
                      population="Complete groups of eight finished root rollouts, including zero-variance "
                                 "groups and budget endings. JSONL cannot identify trainer-dropped datums.")
        cache.write_text(json.dumps(result, indent=2) + "\n")
        return result
    # Rollout JSONL does not contain the critic's pre-update GAE/value moments.
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
                  estimator="critic", actor_loss=train["loss_fn"], x_axis="Completed training batch",
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
    parser.add_argument("--skip-last-steps", type=int, default=2,
                        help="Exclude the newest sampling versions (group) or training batches (critic)")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--fetch-only", action="store_true")
    args = parser.parse_args()
    if args.window < 1 or args.skip_last_steps < 0:
        parser.error("window must be positive; skip-last-steps must be nonnegative")
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
    for run, color in zip(runs, ["#1f77b4", "#ff7f0e", "#d62728", "#17becf", "#9467bd", "#2ca02c", "#8c564b", "#e377c2"]):
        group = run.get("estimator") == "group"
        algorithm = "CISPO group" if group else (
            "CISPO + critic" if run.get("actor_loss") == "cispo" else "PPO")
        label = f"{algorithm} · BS{run['batch_size']}, GS{run['group_size']}"
        if run.get("critic_minibatches", 1) != 1:
            label += f" · K={run['critic_minibatches']}, critic LR={run['critic_learning_rate']:g}"
        if run["run_id"] == "70v8xbkq":
            label += " · β₂=.95, WD=.017"
        if run["run_id"] == "kexcf96w":
            label = f"{algorithm} · BS{run['batch_size']}, GS{run['group_size']}, K={run['critic_minibatches']} · β₂=.95, WD=.017, γ=λ=1"
        known_steps = {int(h["train/step"]) for h in run["history"] if "train/step" in h}
        if group:
            # Include live versions with no completed groups yet, just like the reward plot.
            known_steps.update(int(s) for s in run["audit"])
        excluded = sorted(known_steps)[-args.skip_last_steps:] if args.skip_last_steps else []
        by_step, missing, duplicates = {}, [], []
        for h in run["history"]:
            if "train/step" not in h:
                continue
            step = int(h["train/step"])
            if step in excluded:
                continue
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
                                 explained_variance=1-(a/r)**2, advantage_std=a, return_std=r,
                                 x_axis="sampling_policy_version" if group else "completed_training_batch")
        selected = [by_step[s] for s in sorted(by_step)]
        if not selected:
            ax.plot([], [], color=color, label=f"{label} (awaiting metrics)")
            summary = {k: v for k, v in run.items() if k != "history"}
            summary.update(label=label, updates=0, excluded_steps=excluded)
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
                       excluded_steps=excluded,
                       missing_metric_steps=sorted(set(missing)), gaps=missing_steps,
                       duplicate_steps=duplicates, latest_raw=y[-1], latest_trailing=smooth[-1],
                       first_16_mean=fmean(y[:16]), min_ev=min(y), max_ev=max(y),
                       negative_steps=[r["step"] for r in selected if r["explained_variance"]<0])
        summaries.append(summary)
    ax.axhline(0, color="0.5", linewidth=.8, linestyle="--")
    ax.set(title=f"CodeScout baseline explained variance · {args.window}-step trailing mean",
           xlabel="Training progress (critic: completed batch incl. warmup; group: sampling version)",
           ylabel="Explained variance (one observation per turn)")
    if len({r["step"] for r in rows}) > 1:
        ax.set_xlim(min(r["step"] for r in rows), max(r["step"] for r in rows))
    ax.grid(alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="lower right", fontsize=10)
    for ext in ("png", "pdf"):
        fig.savefig(args.output_dir / f"explained_variance_comparison.{ext}", dpi=180)
    with (args.output_dir / "per_step.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["run_id", "label", "step", "explained_variance",
                                "advantage_std", "return_std", "x_axis", "trailing_mean"])
        writer.writeheader()
        writer.writerows(rows)
    result = dict(formula="1 - (raw_advantages_std / returns_std)^2", window=args.window,
                  skip_last_steps=args.skip_last_steps,
                  precision="Critic: unsampled W&B scan_history. Group: rollout JSONL.",
                  weighting="One observation per turn; trailing mean weights available x-axis points equally.",
                  target="Critic: pre-update GAE lambda-return using each run's configured gamma and lambda. "
                         "Group: final ORM reward, with reward minus group mean as advantage.",
                  caveat="Group EV uses an in-group outcome baseline, not learned out-of-sample predictions; "
                         "its target and sampling-version aggregation differ from critic EV.",
                  runs=summaries)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
