# /// script
# dependencies = ["matplotlib"]
# ///
"""Plot component-wise reward and GRPO group diagnostics from rollout events.

uv run analysis/plot_group_reward_metrics.py TRAIN_ROLLOUT --window 16
Keep every observed (training step, task ID) group, including unfinished and
zero-variance groups. Default missing P2P fail rates to one and other component rewards to zero. Groups with
fewer than eight recorded rollouts use the rollouts present (no invented rows).
Instances means individual rollouts, not distinct task IDs. Non-zero advantage
means abs(reward - group mean) > 1e-10: standard-deviation normalization does not
change which advantages are zero. These are component-wise diagnostics, not
advantages computed from the weighted total reward used during training.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
import re
from pathlib import Path
from statistics import fmean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_rollout_rewards import METRICS as COMPONENT_METRICS, running_average

METRICS = {"overall_reward": ("Overall reward", ("overall_reward",)), "overall_localization": ("Overall localization", ("overall_localization",)), **COMPONENT_METRICS}

STATISTICS = {
    "reward": ("Mean reward", "Reward / rate"),
    "best_at_8": ("Mean group max@8", "Reward / rate"),
    "nonzero_advantage_instances_pct": ("Instances with non-zero advantage", "Rollouts (%)"),
    "nonzero_advantage_groups_pct": ("Groups with non-zero advantages", "Groups (%)"),
}


def group_statistics(groups, tolerance=1e-10, minimize=False):
    if not groups:
        return {key: math.nan for key in STATISTICS}
    active = [[abs(value - fmean(group)) > tolerance for value in group] for group in groups]
    return {
        "reward": fmean(value for group in groups for value in group),
        "best_at_8": fmean((min if minimize else max)(group) for group in groups),
        "nonzero_advantage_instances_pct": 100 * fmean(flag for flags in active for flag in flags),
        "nonzero_advantage_groups_pct": 100 * fmean(any(flags) for flags in active),
    }


def read_rollout(path, audit):
    task_id = None
    root_id = None
    rewards = {}
    finished = False
    with path.open() as stream:
        for line in stream:
            # Avoid decoding the large action/observation records with no rewards.
            if not any(marker in line for marker in
                       ('"trajectory_created"', '"trajectory_task_set"',
                        '"trajectory_finished"', '"localization_reward"', '"binary_reward"')):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                audit['malformed_relevant_lines'] += 1
                continue
            kind = event.get('type')
            if kind == 'trajectory_created' and event['trajectory'].get('parent_info') is None:
                root_id = event['trajectory']['id']
            if event.get('trajectory_id') != root_id or root_id is None:
                continue
            if kind == 'trajectory_task_set':
                task_id = event['task']['id']
            elif kind == 'trajectory_step_added':
                rewards.update(event['step'].get('misc', {}).get('reward_misc', {}))
            elif kind == 'trajectory_finished':
                finished = True
                rewards['overall_reward'] = event.get('reward')
    localization = rewards.get('localization') or {}
    f1_values = [localization.get(key) for key in
                 ('file_f1_reward', 'module_f1_reward', 'entity_f1_reward')]
    rewards['overall_localization'] = fmean(
        value if isinstance(value, (int, float)) and math.isfinite(value) else 0.0
        for value in f1_values
    )
    return task_id, root_id, finished, rewards


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rollout_dir', type=Path)
    parser.add_argument('-k', '--window', type=int, default=16)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--skip-last-steps', type=int, default=2)
    args = parser.parse_args()
    if args.window < 1:
        parser.error('--window must be positive')
    folders = sorted((p for p in args.rollout_dir.iterdir() if p.is_dir() and p.name.isdigit()),
                     key=lambda p: int(p.name))
    if not folders:
        parser.error('No training-step folders found')
    if args.skip_last_steps < 0 or args.skip_last_steps >= len(folders):
        parser.error('--skip-last-steps must be nonnegative and leave at least one step')
    skipped_steps = [int(p.name) for p in folders[-args.skip_last_steps:]] if args.skip_last_steps else []
    if args.skip_last_steps:
        folders = folders[:-args.skip_last_steps]
    print(f'Skipping training steps {skipped_steps}; plotting {folders[0].name} through {folders[-1].name}', flush=True)
    # Snapshot file membership before scanning the actively growing run.
    files = {int(p.name): sorted((p/'events').glob('*.jsonl')) for p in folders}
    output = args.output_dir or args.rollout_dir.parent / f'group_reward_plots_centered_k{args.window}'
    output.mkdir(parents=True, exist_ok=True)
    audit = Counter()
    rows = []
    samples = []
    for step, paths in files.items():
        groups = defaultdict(list)
        seen = set()
        for path in paths:
            task, trajectory, finished, rewards = read_rollout(path, audit)
            if task is None:
                match = re.fullmatch(r'events_(.+)_([0-9a-fA-F-]{36})\.jsonl', path.name)
                if not match:
                    raise ValueError(f'Cannot identify task from {path}')
                task = match.group(1)
                audit['task_ids_recovered_from_filename'] += 1
            trajectory = trajectory or str(path)
            if trajectory in seen:
                raise ValueError(f'Duplicate trajectory {trajectory} at step {step}')
            seen.add(trajectory)
            groups[task].append((finished, rewards))
        complete = {task: group for task, group in groups.items()
                    if len(group) == 8 and all(done for done, _ in group)}
        audit['groups_observed'] += len(groups)
        audit['complete_groups'] += len(complete)
        audit['groups_not_exactly_eight'] += sum(len(g) != 8 for g in groups.values())
        audit['groups_with_unfinished_rollouts'] += sum(not all(done for done, _ in g) for g in groups.values())
        for name, (_, keys) in METRICS.items():
            valid_groups = []
            defaulted = 0
            groups_defaulted = 0
            for task, group in groups.items():
                values = []
                missing = 0
                for _, reward in group:
                    value = reward
                    for key in keys:
                        value = value.get(key) if isinstance(value, dict) else None
                    if not isinstance(value, (int, float)) or not math.isfinite(value):
                        value = 1.0 if name == 'p2p_fail_rate' else 0.0
                        missing += 1
                    values.append(value)
                defaulted += missing
                groups_defaulted += missing > 0
                valid_groups.append(values)
                samples.append({'step': step, 'task_id': task, 'component': name,
                                'rewards': values, 'defaulted_rewards': missing})
            row = {'step': step, 'component': name, 'observed_groups': len(groups),
                   'complete_groups': len(complete), 'eligible_groups': len(valid_groups),
                   'eligible_instances': sum(map(len, valid_groups)),
                   'groups_with_defaulted_rewards': groups_defaulted,
                   'rewards_defaulted': defaulted,
                   'default_reward': 1.0 if name == 'p2p_fail_rate' else 0.0}
            row.update(group_statistics(valid_groups, minimize=name == 'p2p_fail_rate'))
            rows.append(row)
        if step % 16 == 0:
            print(f'Processed training step {step}', flush=True)
    steps = list(range(min(files), max(files) + 1))
    for name, (title, _) in METRICS.items():
        component_rows = {r['step']: r for r in rows if r['component'] == name}
        for metric, (metric_title, ylabel) in STATISTICS.items():
            file_metric = metric
            if metric == 'best_at_8':
                direction = 'min' if name == 'p2p_fail_rate' else 'max'
                file_metric = f'{direction}_at_8'
                metric_title = f'Mean group {direction}@8'
            values = [component_rows.get(step, {}).get(metric, math.nan) for step in steps]
            if not any(math.isfinite(v) for v in values):
                raise ValueError(f'No observed groups for {name}')
            smooth = running_average(values, args.window, 'centered')
            for step, value in zip(steps, smooth):
                if step in component_rows:
                    component_rows[step][metric+'_centered_average'] = value
            fig, ax = plt.subplots(figsize=(10, 5.5), layout='constrained')
            ax.plot(steps, values, color='#9cb9cf', lw=1, alpha=.8, label='Per training step')
            ax.plot(steps, smooth, color='#1263a0', lw=2.3,
                    label=f'Centered {args.window}-step average')
            ax.set(title=f'{title}: {metric_title}', xlabel='Training step', ylabel=ylabel)
            ax.grid(alpha=.22)
            ax.legend()
            ax.margins(x=.015, y=.1)
            fig.savefig(output/f'{name}__{file_metric}.png', dpi=180)
            plt.close(fig)
    (output/'p2p_fail_rate__max_at_8.png').unlink(missing_ok=True)
    with (output/'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output/'group_rewards.jsonl').open('w') as stream:
        for sample in samples:
            stream.write(json.dumps(sample)+'\n')
    summary = {'source': str(args.rollout_dir.resolve()), 'window': args.window,
               'alignment': 'centered', 'preceding_steps': args.window//2,
               'following_steps': (args.window-1)//2, 'partial_edge_windows': True,
               'training_step_count': len(files), 'skipped_training_steps': skipped_steps, 'png_count': len(METRICS)*len(STATISTICS),
               'expected_group_size': 8, 'partial_groups': 'Use every recorded rollout; do not pad absent rollouts', 'group_key': ['training step', 'task ID'],
               'instance_definition': 'individual rollout', 'nonzero_tolerance': 1e-10,
               'advantage': 'component reward minus within-group component mean; nonzero status unaffected by std normalization',
               'missing_values': 'Default missing or nonfinite P2P fail rate to 1; default other component rewards to 0; keep group and rollout',
               'reward_population': 'All observed groups and rollouts, including unfinished and zero-variance groups, for all four metrics',
               'best_at_8': 'Minimum within each group for P2P fail rate; maximum for all other components; averaged over groups',
               'smoothing': 'Equal-weight mean of available per-step statistics',
               'audit': dict(audit),
               'eligible_groups_by_component': {name: sum(r['eligible_groups'] for r in rows if r['component']==name) for name in METRICS}}
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    print(f'Saved {summary["png_count"]} PNGs to {output}')


if __name__ == '__main__':
    main()
