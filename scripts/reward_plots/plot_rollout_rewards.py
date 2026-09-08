# /// script
# dependencies = ["matplotlib"]
# ///
"""Plot per-training-step reward means and trailing or centered k-step averages.

Run with: uv run analysis/plot_rollout_rewards.py TRAIN_ROLLOUT --window 8
Missing rewards are excluded, never replaced with zero. Edge windows use
available steps. Centered windows include k//2 preceding steps and (k-1)//2
following steps, plus the current step. Only finished trajectories contribute to the plots.
"""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = {
    "file_f1": ("File F1", ("localization", "file_f1_reward")),
    "function_f1": ("Function F1", ("localization", "entity_f1_reward")),
    "module_f1": ("Module F1", ("localization", "module_f1_reward")),
    "binary_issue_resolution": ("Binary issue resolution", ("binary_reward",)),
    "f2p_pass_rate": ("F2P pass rate", ("f2p_pass_fraction",)),
    "p2p_fail_rate": ("P2P fail rate", ("p2p_fail_fraction",)),
    "json_format_tool": ("JSON format tool reward", ("tool_json_error_reward",)),
    "agent_error_reward": ("Agent error reward", ("agent_error_event_reward",)),
    "str_replace_reward": ("Str replace reward", ("str_replace_reward",)),
}


def running_average(values, k, alignment="trailing"):
    before = k // 2 if alignment == "centered" else k - 1
    after = (k - 1) // 2 if alignment == "centered" else 0
    return [fmean(valid) if (valid := [v for v in values[max(0, i-before):i+after+1]
                                     if math.isfinite(v)]) else math.nan
            for i in range(len(values))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout_dir", type=Path)
    parser.add_argument("-k", "--window", type=int, default=8)
    parser.add_argument("--alignment", choices=["trailing", "centered"], default="trailing")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.window < 1:
        parser.error("--window must be positive")
    folders = sorted((p for p in args.rollout_dir.iterdir() if p.is_dir() and p.name.isdigit()),
                     key=lambda p: int(p.name))
    if not folders:
        parser.error("No numeric training-step directories found")
    suffix = "_centered" if args.alignment == "centered" else ""
    output = args.output_dir or args.rollout_dir.parent / f"reward_plots{suffix}_k{args.window}"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    skipped = malformed = finished = 0
    for folder in folders:
        samples = {name: [] for name in METRICS}
        for path in sorted((folder / "events").glob("*.jsonl")):
            rewards = {}
            complete = False
            with path.open() as stream:
                for line in stream:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        malformed += 1
                        continue
                    if event.get("type") == "trajectory_step_added":
                        info = event["step"].get("misc", {}).get("reward_misc", {})
                        if info:
                            rewards.update(info)
                    elif event.get("type") == "trajectory_finished":
                        complete = True
            if not complete:
                skipped += 1
                continue
            finished += 1
            for name, (_, keys) in METRICS.items():
                value = rewards
                for key in keys:
                    value = value.get(key) if isinstance(value, dict) else None
                if isinstance(value, (int, float)) and math.isfinite(value):
                    samples[name].append(value)
        row = {"step": int(folder.name)}
        for name, values in samples.items():
            row[name] = fmean(values) if values else math.nan
            row[name + "_count"] = len(values)
        rows.append(row)
    # Include gaps so the window spans training steps rather than observed folders.
    by_step = {row["step"]: row for row in rows}
    steps = list(range(rows[0]["step"], rows[-1]["step"] + 1))
    for name, (title, _) in METRICS.items():
        values = [by_step.get(step, {}).get(name, math.nan) for step in steps]
        smooth = running_average(values, args.window, args.alignment)
        if not any(math.isfinite(v) for v in values):
            raise ValueError(f"No recorded values for {name}")
        for step, avg in zip(steps, smooth):
            if step in by_step:
                by_step[step][name + "_running_average"] = avg
        fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
        ax.plot(steps, values, color="#9cb9cf", linewidth=1, alpha=0.8, label="Training-step mean")
        ax.plot(steps, smooth, color="#1263a0", linewidth=2.3,
                label=f"{args.alignment.title()} {args.window}-step average")
        ax.set(title=title, xlabel="Training step", ylabel="Mean reward / rate")
        ax.grid(alpha=0.22)
        ax.legend()
        ax.margins(x=0.015, y=0.1)
        fig.savefig(output / f"{name}.png", dpi=180)
        plt.close(fig)
    with (output / "reward_means.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"source": str(args.rollout_dir.resolve()), "window": args.window,
               "alignment": args.alignment,
               "training_steps": len(rows), "finished_trajectories": finished,
               "unfinished_trajectories_excluded": skipped, "malformed_lines_skipped": malformed,
               "aggregation": f"Mean of recorded rewards per step; {args.alignment} mean of step means; partial edge windows",
               "counts": {name: sum(row[name + "_count"] for row in rows) for name in METRICS}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved nine PNGs and underlying CSV to {output}")


if __name__ == "__main__":
    main()
