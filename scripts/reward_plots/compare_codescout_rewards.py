# /// script
# dependencies = ["matplotlib"]
# ///
"""Overlay CodeScout rewards read directly from rollout event JSONL files."""

import argparse
import csv
import json
import math
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

RUN_ROOT = Path("/data/user_data/adityabs/miles_sail/codescout-platoon-tinker")
DEFAULT_RUNS = [
    ("codescout-sail-ppo-20261015-203723", "PPO · BS32, GS8, K=1, critic LR=1e-5"),
    ("codescout-sail-ppo-b256-g1-20261016-223443", "PPO · BS256, GS1, K=1, critic LR=1e-5"),
    ("codescout-sail-ppo-b256-g1-k4-20261022-222813", "PPO · BS256, GS1, K=4, critic LR=5e-6"),
    ("codescout-sail-ppo-b256-g1-k8-20261022-233113", "PPO · BS256, GS1, K=8, critic LR=3.54e-6"),
    ("codescout-sail-cispo-b32-g8-20261024-124001", "CISPO group · BS32, GS8"),
    ("codescout-sail-cispo-critic-b256-g1-k4-20261024-124518", "CISPO + critic · BS256, GS1, K=4, critic LR=5e-6"),
]
# Keep each run's colour consistent with compare_critic_ev.py, including subsets.
RUN_COLORS = dict(zip(
    (name for name, _ in DEFAULT_RUNS),
    ["#1f77b4", "#ff7f0e", "#d62728", "#17becf", "#9467bd", "#2ca02c"],
))


def read_rollout(path):
    roots = set()
    finished = {}
    audit = Counter(files_read=1)
    with path.open() as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                audit["malformed_lines_skipped"] += 1
                continue
            if event.get("type") == "trajectory_created":
                trajectory = event["trajectory"]
                if trajectory.get("parent_info") is None:
                    roots.add(trajectory["id"])
            elif event.get("type") == "trajectory_finished":
                finished[event["trajectory_id"]] = event
    audit["unfinished_roots_excluded"] += len(roots - finished.keys())
    rewards = []
    for trajectory_id in roots & finished.keys():
        event = finished[trajectory_id]
        audit["finished_roots"] += 1
        reward = event.get("reward")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
            audit["missing_or_invalid_rewards_excluded"] += 1
            continue
        rewards.append((trajectory_id, float(reward), bool(event.get("error_message"))))
    return rewards, audit


def collect_run(path, label, args):
    source = path.resolve()
    if (source / "rollouts/train").is_dir():
        source /= "rollouts/train"
    if not source.is_dir():
        raise ValueError(f"Directory not found: {source}")
    folders = sorted((p for p in source.iterdir() if p.is_dir() and p.name.isdigit()),
                     key=lambda p: int(p.name))
    if not folders:
        raise ValueError(f"{label}: no rollout policy-version directories found in {source}; "
                         "wait for the run to write rollout JSONL logs")
    skipped = folders[-args.skip_last_steps:] if args.skip_last_steps else []
    retained = folders[:-args.skip_last_steps] if args.skip_last_steps else folders
    if args.max_step is not None:
        retained = [p for p in retained if int(p.name) <= args.max_step]
    # Snapshot membership before scanning a live run.
    files = [(int(p.name), sorted((p / "events").glob("*.jsonl"))) for p in retained]
    skipped_counts = {p.name: len(list((p / "events").glob("*.jsonl"))) for p in skipped}
    audit = Counter()
    seen = set()
    rows = []
    history = deque()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for step, paths in files:
            rewards = []
            errors = 0
            for results, counts in pool.map(read_rollout, paths):
                audit.update(counts)
                for trajectory_id, reward, has_error in results:
                    if trajectory_id in seen:
                        audit["duplicate_roots_excluded"] += 1
                        continue
                    seen.add(trajectory_id)
                    rewards.append(reward)
                    errors += has_error
            mean = fmean(rewards) if rewards else None
            history.append((step, mean))
            while history and history[0][0] < step - args.window + 1:
                history.popleft()
            valid = [value for _, value in history if value is not None]
            rows.append({"run": label, "policy_version": step, "rollouts": len(rewards),
                         "error_endings_included": errors, "mean_reward": mean,
                         "trailing_mean_reward": fmean(valid) if valid else None})
    if not seen and retained:
        raise ValueError(f"{label}: no finished root trajectories with finite rewards remain")
    summary = {"label": label, "source": str(source), "rollouts_plotted": len(seen),
               "excluded_policy_versions": [int(p.name) for p in skipped],
               "files_in_skipped_versions_at_scan": skipped_counts,
               "first_policy_version": rows[0]["policy_version"] if rows else None,
               "last_policy_version": rows[-1]["policy_version"] if rows else None, "audit": dict(audit)}
    print(f"{label}: {len(seen)} rollouts; skipped {[int(p.name) for p in skipped]}", flush=True)
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path,
                        help="Run roots or rollouts/train directories; defaults to six CodeScout comparison runs")
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--skip-last-steps", type=int, default=1)
    parser.add_argument("--max-step", type=int, help="Maximum policy version, applied after skipping newest versions")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/codescout_reward_comparison"))
    args = parser.parse_args()
    if not args.runs:
        args.runs = [RUN_ROOT / name for name, _ in DEFAULT_RUNS]
        if args.labels is None:
            args.labels = [label for _, label in DEFAULT_RUNS]
    if args.window < 1 or args.workers < 1 or args.skip_last_steps < 0:
        parser.error("window/workers must be positive; skip-last-steps must be nonnegative")
    labels = args.labels or [p.resolve().name for p in args.runs]
    if len(labels) != len(args.runs) or len(set(labels)) != len(labels):
        parser.error("Provide one unique label per run")
    rows = []
    summaries = []
    for path, label in zip(args.runs, labels):
        try:
            run_rows, summary = collect_run(path, label, args)
        except ValueError as exc:
            parser.error(str(exc))
        rows.extend(run_rows)
        summaries.append(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
    for path, label in zip(args.runs, labels):
        source = path.resolve()
        run_name = source.parent.parent.name if source.name == "train" else source.name
        color = RUN_COLORS.get(run_name) or ax._get_lines.get_next_color()
        data = {row["policy_version"]: row for row in rows if row["run"] == label}
        if not data:
            ax.plot([], [], color=color, label=f"{label} (awaiting retained versions)")
            continue
        # Missing versions break plotted lines; they are never filled with zero.
        steps = list(range(min(data), max(data) + 1))
        smooth = [data.get(s, {}).get("trailing_mean_reward") for s in steps]
        ax.plot(steps, [math.nan if v is None else v for v in smooth],
                color=color, linewidth=2.3, marker=".", label=label)
    ax.set(title=f"CodeScout reward comparison · trailing {args.window}-step averages",
           xlabel="Sampling policy version (step)", ylabel="Mean final rollout reward")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.2)
    ax.legend()
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"reward_comparison.{suffix}", dpi=180)
    plt.close(fig)
    with (args.output_dir / "rewards.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["run", "policy_version", "rollouts",
                                "error_endings_included", "mean_reward", "trailing_mean_reward"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {"window": args.window, "skip_last_steps": args.skip_last_steps,
               "max_step": args.max_step, "runs": summaries,
               "reward_source": "trajectory_finished.reward; no substituted or recomputed rewards",
               "population": "Finished root trajectories with finite rewards, including zero/error rewards",
               "smoothing": "Equal-weight means over policy versions in [step-window+1, step]; "
                            "available values at edges",
               "x_axis": "Sampling policy version, not optimizer consumption step"}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved PNG, PDF, CSV, and exclusion counts to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
