# /// script
# dependencies = ["matplotlib", "numpy"]
# ///
"""Plot P(issue resolved | auxiliary reward bucket) from rollouts or a CSV snapshot."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import math
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_group_reward_metrics import read_rollout


# Canonical CSV column, display title, number of equal-width bins (None = binary).
METRICS = [
    ("localization_reward", "Overall localization", 5),
    ("file_f1", "File F1", 5),
    ("module_f1", "Module F1", 5),
    ("function_f1", "Function F1", 5),
    ("tool_json", "JSON tool format", None),
    ("agent_error", "Agent error reward", None),
    ("str_replace", "File editor reward", 4),
]
COLUMNS = ["run", "step", "task", "binary", "empty_patch", "path"] + [m[0] for m in METRICS]


def number(value):
    try:
        value = float(value)
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) and 0 <= value <= 1 else None


def bucket_index(value, bins):
    """Left-inclusive bins; the final bin includes 1. Missing values stay missing."""
    value = number(value)
    if value is None:
        return None
    if bins is None:
        return int(value) if value in (0, 1) else None
    return min(bins - 1, math.floor(value * bins + 1e-12))


def bucket_labels(bins):
    if bins is None:
        return ["0", "1"]
    return [f"[{i / bins:g}, {(i + 1) / bins:g}{']' if i == bins - 1 else ')'}"
            for i in range(bins)]


def read_record(item):
    run, step, path = item
    audit = Counter()
    task, _, finished, rewards = read_rollout(path, audit)
    if not finished or not audit["rollouts_with_reward_dict"]:
        return None
    binary = number(rewards.get("binary_reward"))
    if binary not in (0, 1):
        return None
    loc = rewards.get("localization") or {}
    patch = rewards.get("model_patch")
    # Empty-patch terminal results omit model_patch but have this explicit error.
    empty = (not patch.strip()) if isinstance(patch, str) else None
    if str(rewards.get("error", "")).startswith("Empty model patch"):
        empty = True
    if task is None:
        match = re.fullmatch(r"events_(.+)_([0-9a-fA-F-]{36})\.jsonl", path.name)
        task = match.group(1) if match else str(path)
    values = {
        "localization_reward": rewards.get("localization_reward"),
        "file_f1": loc.get("file_f1_reward"),
        "module_f1": loc.get("module_f1_reward"),
        "function_f1": loc.get("entity_f1_reward"),
        "tool_json": rewards.get("tool_json_error_reward"),
        "agent_error": rewards.get("agent_error_event_reward"),
        "str_replace": rewards.get("str_replace_reward"),
    }
    return dict(run=run, step=step, task=task, binary=binary, empty_patch=empty,
                path=str(path), **{k: number(v) for k, v in values.items()})


def read_csv(path):
    with path.open() as stream:
        reader = csv.DictReader(stream)
        required = {"run", "step", "task", "binary", "empty_patch"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"CSV requires columns {sorted(required)}; this must be per-rollout data")
        rows = []
        for record in reader:
            binary = number(record["binary"])
            if binary not in (0, 1):
                continue
            empty = {"true": True, "false": False, "1": True, "0": False}.get(
                record["empty_patch"].lower())
            rows.append(dict(run=record["run"], step=int(record["step"]),
                             task=record["task"], binary=binary, empty_patch=empty,
                             path=record.get("path", ""),
                             **{key: number(record.get(key)) for key, _, _ in METRICS}))
        return rows


def select_steps(steps, skip_last, min_step, max_step):
    steps = sorted(set(steps))
    if skip_last:
        steps = steps[:-skip_last]
    return [s for s in steps if s >= min_step and (max_step is None or s <= max_step)]


def bootstrap_interval(task_counts, samples, rng):
    """Resample task clusters, retaining repeated attempts together."""
    if not task_counts or not samples:
        return None, None
    values = np.asarray(list(task_counts.values()), dtype=float)
    estimates = []
    for _ in range(samples):
        selected = values[rng.integers(0, len(values), len(values))]
        estimates.append(selected[:, 0].sum() / selected[:, 1].sum())
    return tuple(float(x) for x in np.quantile(estimates, [.025, .975]))


def aggregate(rows, populations=("all", "nonempty"), bootstrap_samples=1000, seed=42):
    result = []
    rng = np.random.default_rng(seed)
    for run in dict.fromkeys(r["run"] for r in rows):
        run_rows = [r for r in rows if r["run"] == run]
        for population in populations:
            selected = [r for r in run_rows if population == "all" or r["empty_patch"] is False]
            if not selected:
                continue
            for metric, _, bins in METRICS:
                eligible = [(r, bucket_index(r.get(metric), bins)) for r in selected]
                eligible = [(r, idx) for r, idx in eligible if idx is not None]
                baseline = sum(r["binary"] for r, _ in eligible) / len(eligible) if eligible else None
                for idx, label in enumerate(bucket_labels(bins)):
                    bucket = [r for r, j in eligible if j == idx]
                    resolved = sum(r["binary"] for r in bucket)
                    task_counts = defaultdict(lambda: [0, 0])
                    for r in bucket:
                        task_counts[r["task"]][0] += r["binary"]
                        task_counts[r["task"]][1] += 1
                    low, high = bootstrap_interval(task_counts, bootstrap_samples, rng)
                    result.append(dict(
                        run=run, population=population, metric=metric, bucket_index=idx,
                        bucket=label, n=len(bucket), tasks=len(task_counts), resolved=int(resolved),
                        p_resolved=resolved / len(bucket) if bucket else None,
                        ci_low=low, ci_high=high, eligible_n=len(eligible),
                        excluded_missing_metric=len(selected) - len(eligible), baseline=baseline,
                    ))
    return result


def write_csv(path, rows, fieldnames=None):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(buckets, rows, output, y_max=None):
    """One 2x4 bar grid per run/population, matching the existing bucket dashboard."""
    keys = list(dict.fromkeys((r["run"], r["population"]) for r in buckets))
    paths = []
    for plot_id, (run, population) in enumerate(keys):
        subset = [r for r in buckets if (r["run"], r["population"]) == (run, population)]
        max_value = max((r["ci_high"] if r["ci_high"] is not None else r["p_resolved"] or 0)
                        for r in subset)
        limit = y_max or min(1.12, max(.4, max_value * 1.25 + .04))
        fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=True)
        for ax, (metric, title, bins) in zip(axes.flat, METRICS):
            values = [r for r in subset if r["metric"] == metric]
            x = np.arange(len(values))
            probs = [r["p_resolved"] if r["p_resolved"] is not None else 0 for r in values]
            ax.bar(x, probs, color="#2165a3", width=.78)
            for i, r in enumerate(values):
                p = r["p_resolved"]
                if p is None:
                    ax.text(i, .012, "n=0", ha="center", fontsize=8)
                    continue
                high = p
                if r["ci_low"] is not None:
                    # Percentile cluster intervals need not contain the observed point estimate.
                    ax.vlines(i, r["ci_low"], r["ci_high"], color="#333333", lw=.8)
                    ax.hlines([r["ci_low"], r["ci_high"]], i-.05, i+.05, color="#333333", lw=.8)
                    high = max(p, r["ci_high"])
                ax.text(i, high + limit*.018, f"{p:.1%}\nn={r['n']:,}",
                        ha="center", va="bottom", fontsize=8)
            baseline = values[0]["baseline"]
            if baseline is not None:
                ax.axhline(baseline, color="#777777", ls=":", lw=1)
            ax.set(title=title, xlabel="Auxiliary reward bucket", ylim=(0, limit),
                   xticks=x, xticklabels=bucket_labels(bins))
            ax.tick_params(axis="x", labelsize=8)
            ax.grid(axis="y", alpha=.15)
            ax.set_axisbelow(True)
        for ax in axes[:, 0]:
            ax.set_ylabel("P(issue resolved)")
        axes.flat[-1].axis("off")
        steps = sorted({r["step"] for r in rows if r["run"] == run})
        population_label = "all included trajectories" if population == "all" else "nonempty patches only"
        fig.suptitle(f"{run}: issue resolution conditional on auxiliary reward\n"
                     f"Steps {steps[0]}–{steps[-1]} • {population_label}", fontsize=14)
        fig.text(.02, .015,
                 "Bars: resolved / trajectories in bucket. Whiskers: 95% task-cluster bootstrap CI. "
                 "Dotted lines: baseline for available metric.\n"
                 "Bins include their left edge; the final bin includes 1. "
                 "Missing auxiliary values are excluded separately for each metric.", fontsize=9)
        fig.tight_layout(rect=(0, .055, 1, .93))
        slug = re.sub(r"[^a-z0-9]+", "_", run.lower()).strip("_") or "run"
        stem = f"{plot_id:02d}_{slug}_{population}"
        for extension in ("png", "pdf"):
            target = output / f"{stem}.{extension}"
            fig.savefig(target, dpi=180)
            paths.append(str(target))
        plt.close(fig)
    return paths


def render_overlay(buckets, rows, output, y_max=None):
    """Compare runs in the same bins with side-by-side bars and shared axes."""
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter
    runs = list(dict.fromkeys(r['run'] for r in buckets))
    colors = ['#1673b8', '#e27718', '#27955b', '#9361b0']
    if len(runs) > len(colors):
        raise ValueError('Overlay supports at most four runs')
    paths = []
    for population in dict.fromkeys(r['population'] for r in buckets):
        subset = [r for r in buckets if r['population'] == population]
        largest = max((r['ci_high'] if r['ci_high'] is not None else r['p_resolved'] or 0)
                      for r in subset)
        limit = y_max or min(1.12, max(.4, largest * 1.25 + .04))
        fig, axes = plt.subplots(2, 4, figsize=(22, 10), sharey=True)
        width = .8 / len(runs)
        for ax, (metric, title, bins) in zip(axes.flat, METRICS):
            labels = bucket_labels(bins)
            for run_index, run in enumerate(runs):
                color = colors[run_index]
                values = {r['bucket_index']: r for r in subset if r['run'] == run and r['metric'] == metric}
                offset = (run_index - (len(runs)-1)/2) * width
                for idx in range(len(labels)):
                    r = values.get(idx)
                    x = idx + offset
                    if r is None or r['p_resolved'] is None:
                        ax.text(x, .012, 'n=0', ha='center', fontsize=7, color=color)
                        continue
                    prob = r['p_resolved']
                    ax.bar(x, prob, width=width*.92, color=color)
                    high = prob
                    if r['ci_low'] is not None:
                        ax.vlines(x, r['ci_low'], r['ci_high'], color='#333333', lw=.8)
                        ax.hlines([r['ci_low'], r['ci_high']], x-width*.14, x+width*.14, color='#333333', lw=.8)
                        high = max(prob, r['ci_high'])
                    ax.text(x, high + limit*.012, f"{prob:.1%}\nn={r['n']:,}",
                            ha='center', va='bottom', fontsize=6.5, color=color)
                if values:
                    baseline = next(iter(values.values()))['baseline']
                    if baseline is not None:
                        ax.axhline(baseline, color=color, ls=':', lw=1, alpha=.8)
            ax.set(title=title, xlabel='Auxiliary reward bucket', ylim=(0, limit),
                   xticks=np.arange(len(labels)), xticklabels=labels)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(axis='y', alpha=.15)
            ax.set_axisbelow(True)
        for ax in axes[:, 0]:
            ax.set_ylabel('P(issue resolved)')
        legend_ax = axes.flat[-1]
        legend_ax.axis('off')
        legend_ax.legend(handles=[Patch(color=colors[i], label=run) for i, run in enumerate(runs)],
                         loc='upper left', frameon=False, fontsize=14)
        ranges = []
        for run in runs:
            steps = [r['step'] for r in rows if r['run'] == run]
            ranges.append(f'{run}: steps {min(steps)}–{max(steps)}')
        legend_ax.text(0, .64, '\n'.join(ranges) + '\n\nWhiskers: 95% task-cluster bootstrap CI\nDotted lines: per-run metric baseline',
                       transform=legend_ax.transAxes, fontsize=11, va='top')
        pop_label = 'all included trajectories' if population == 'all' else 'nonempty patches only'
        fig.suptitle(f'Issue resolution conditional on auxiliary reward • {pop_label}', fontsize=18)
        fig.text(.02, .015, 'Repeated attempts retained; missing auxiliary values excluded per metric. '
                 'Association does not establish a causal training benefit.', fontsize=10)
        fig.tight_layout(rect=(0, .045, 1, .95))
        for extension in ('png', 'pdf'):
            target = output / f'comparison_{population}.{extension}'
            fig.savefig(target, dpi=180)
            paths.append(str(target))
        plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="*", help="Run root, rollouts/train, or train_rollout directories")
    parser.add_argument("--rollout-csv", type=Path, help="Per-rollout CSV from this plotter or the localization audit")
    parser.add_argument("--labels", nargs="+", help="Display labels, matching run directories or CSV run order")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--population", choices=["all", "nonempty", "both"], default="both")
    parser.add_argument("--skip-last-steps", type=int, default=0)
    parser.add_argument("--min-step", type=int, default=0)
    parser.add_argument("--max-step", type=int, help="Inclusive upper step bound")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--y-max", type=float, help="Optional shared y-axis maximum")
    parser.add_argument('--overlay', action='store_true', help='Also create combined charts with differently coloured bars for each run')
    args = parser.parse_args()
    if bool(args.runs) == bool(args.rollout_csv):
        parser.error("Supply run directories OR --rollout-csv")
    if min(args.skip_last_steps, args.min_step, args.bootstrap_samples) < 0 or args.workers < 1:
        parser.error("Counts must be nonnegative; workers must be positive")
    if args.max_step is not None and args.max_step < args.min_step:
        parser.error("max-step must be >= min-step")
    if args.y_max is not None and args.y_max <= 0:
        parser.error("y-max must be positive")
    observed = None
    if args.rollout_csv:
        rows = read_csv(args.rollout_csv)
        names = list(dict.fromkeys(r["run"] for r in rows))
        selected = {name: select_steps([r["step"] for r in rows if r["run"] == name],
                                      args.skip_last_steps, args.min_step, args.max_step) for name in names}
        rows = [r for r in rows if r["step"] in selected[r["run"]]]
    else:
        names = [str(p) for p in args.runs]
        if len(names) != len(set(names)):
            parser.error("Run directories must be distinct")
        items = []
        selected = {}
        for name, root in zip(names, args.runs):
            folder = next((p for p in [root/"rollouts/train", root/"train_rollout", root]
                           if p.is_dir() and any(c.is_dir() and c.name.isdigit() for c in p.iterdir())), None)
            if folder is None:
                parser.error(f"No numeric step directories in {root}")
            selected[name] = select_steps([int(p.name) for p in folder.iterdir() if p.is_dir() and p.name.isdigit()],
                                          args.skip_last_steps, args.min_step, args.max_step)
            for step in selected[name]:
                items.extend((name, step, p) for p in sorted((folder/str(step)/"events").glob("*.jsonl")))
        observed = len(items)
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, row in enumerate(pool.map(read_record, items, chunksize=16), 1):
                if row is not None:
                    rows.append(row)
                if i % 10000 == 0:
                    print(f"Read {i:,}/{observed:,} trajectories", flush=True)
    if not rows:
        parser.error("No eligible trajectories in the selected steps")
    labels = args.labels or names
    if len(labels) != len(names) or len(labels) != len(set(labels)):
        parser.error("Supply one unique label per run")
    label_map = dict(zip(names, labels))
    for row in rows:
        row["run"] = label_map[row["run"]]
    populations = ("all", "nonempty") if args.population == "both" else (args.population,)
    buckets = aggregate(rows, populations, args.bootstrap_samples, args.seed)
    if not buckets:
        parser.error("No trajectories qualify for the requested population")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Permit redraws from the output snapshot without changing its source schema.
    snapshot = args.output_dir / "rollouts.csv"
    if not args.rollout_csv or snapshot.resolve() != args.rollout_csv.resolve():
        write_csv(snapshot, rows, COLUMNS)
    write_csv(args.output_dir / "buckets.csv", buckets)
    plots = render(buckets, rows, args.output_dir, args.y_max)
    if args.overlay:
        plots.extend(render_overlay(buckets, rows, args.output_dir, args.y_max))
    summary = dict(
        source=str(args.rollout_csv) if args.rollout_csv else [str(p) for p in args.runs],
        selected_steps={label_map[k]: v for k, v in selected.items()},
        observed_files=observed, included_rollouts=len(rows),
        excluded_files=observed-len(rows) if observed is not None else None,
        population="Finished root trajectory, nonempty root reward dictionary, recorded binary label 0 or 1",
        nonempty="Explicit nonempty model_patch (or empty_patch=False in CSV); unknown patch status excluded",
        missing="Missing/nonfinite/out-of-range auxiliary values excluded separately for each metric",
        aggregation="Rollout-weighted conditional probability; repeated attempts retained; no smoothing",
        bins={key: bucket_labels(bins) for key, _, bins in METRICS},
        bootstrap_samples=args.bootstrap_samples, seed=args.seed,
        uncertainty="95% percentile task-cluster bootstrap within each run/bucket",
        per_run={label: {"n": sum(r["run"] == label for r in rows),
                         "resolved": int(sum(r["binary"] for r in rows if r["run"] == label))}
                 for label in labels}, plots=plots,
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved {len(plots)} plots, buckets.csv, rollouts.csv and summary.json to {args.output_dir}")


if __name__ == "__main__":
    main()
