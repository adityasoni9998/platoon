# /// script
# dependencies = ["matplotlib"]
# ///
"""Create three reward grids from plot_group_reward_metrics.py's saved CSV.

Usage: uv run analysis/plot_reward_grids.py GROUP_PLOTS_DIRECTORY
Uses saved metrics without rescanning rollouts; supports overlaid runs and trailing averages.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plot_rollout_rewards import running_average

GRIDS = {
    'localization': [('overall_localization', 'Overall localization'), ('file_f1', 'File F1'), ('module_f1', 'Module F1'), ('function_f1', 'Function F1')],
    'testing': [('binary_issue_resolution', 'Binary resolution'), ('f2p_pass_rate', 'F2P pass rate'),
                ('p2p_fail_rate', 'P2P pass rate')],
    'tool_formatting': [('json_format_tool', 'JSON tool format'), ('agent_error_reward', 'Agent error reward'),
                        ('str_replace_reward', 'Str replace reward')],
}
COLUMNS = [('reward', 'Mean reward'), ('best_at_8', 'Max@8'),
           ('nonzero_advantage_instances_pct', 'Non-zero advantage instances (fraction)')]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('--combined', action='store_true', help='Create one 11-by-3 figure')
    parser.add_argument('--horizontal', action='store_true', help='Transpose metrics and components')
    parser.add_argument('--smooth-only', action='store_true', help='Show only running averages')
    parser.add_argument('--dynamic-ylim', action='store_true', help='Autoscale each subplot y-axis')
    parser.add_argument('--window', type=int, help='Override averaging window')
    parser.add_argument('--alignment', choices=['centered', 'trailing'])
    parser.add_argument('--output-dir', type=Path, help='Save figures separately from the input metrics')
    parser.add_argument('--skip-first-steps', nargs=2, action='append', default=[],
                        metavar=('RUN', 'N'), help='For RUN, omit zero-based training steps below N before smoothing (repeatable)')
    parser.add_argument('--rebase-skipped-steps', action='store_true',
                        help='Subtract each run\'s skipped step count from its plotted step numbers')
    args = parser.parse_args(argv)
    summary = json.loads((args.input_dir/'summary.json').read_text())
    with (args.input_dir/'metrics.csv').open() as f:
        rows = list(csv.DictReader(f))
    original_runs = list(dict.fromkeys(r.get('run', '') for r in rows))
    skip_first = {}
    for run, count in args.skip_first_steps:
        if run not in original_runs:
            parser.error(f'Unknown run: {run}')
        try:
            count = int(count)
        except ValueError:
            parser.error('--skip-first-steps requires an integer count')
        if count < 0 or run in skip_first:
            parser.error('Skip counts must be nonnegative and run labels must be unique')
        skip_first[run] = count
    if args.rebase_skipped_steps and not skip_first:
        parser.error('--rebase-skipped-steps requires --skip-first-steps')
    rows = [r for r in rows if int(r['step']) >= skip_first.get(r.get('run', ''), 0)]
    for run in original_runs:
        if not any(r.get('run', '') == run for r in rows):
            parser.error(f'No steps remain for {run}')
    if args.rebase_skipped_steps:
        for row in rows:
            row['step'] = str(int(row['step']) - skip_first.get(row.get('run', ''), 0))
    output_dir = args.output_dir or args.input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    axis_label = 'Training step (warmup removed for selected runs)' if args.rebase_skipped_steps else 'Training step'
    if args.combined and not any(r['component'] == 'overall_reward' for r in rows):
        parser.error('Saved metrics lack overall reward. Regenerate from train_rollout using plot.py.')
    if not any(r['component'] == 'overall_localization' for r in rows):
        parser.error('Saved metrics lack overall localization. Regenerate from train_rollout using plot.py.')
    comparison = bool(rows and 'run' in rows[0])
    # Accept the saved dense/sparse comparison snapshot as well as the original CSV.
    if comparison and 'mean' in rows[0]:
        for row in rows:
            row['eligible_instances'] = row['rollouts']
            row['reward'] = float(row['mean'])
            row['best_at_8'] = float(row['max_at_8'])
            row['nonzero_advantage_instances_pct'] = 100 * float(row['nonzero_advantage'])
            if row['component'] == 'p2p_pass_rate':
                row['component'] = 'p2p_fail_rate'
                row['reward'] = 1 - row['reward']
                row['best_at_8'] = 1 - row['best_at_8']
    window = args.window or summary.get('window', 8 if comparison else 16)
    if window < 1:
        parser.error('--window must be positive')
    alignment = args.alignment or summary.get('alignment', 'trailing' if comparison else 'centered')
    runs = list(dict.fromkeys(r.get('run', '') for r in rows))
    colors = dict(zip(runs, ['#1673b8', '#e27718', '#27955b', '#9361b0']))
    if len(runs) > len(colors):
        parser.error('At most four overlaid runs are supported')
    grids = {'all_rewards': [component for group in GRIDS.values() for component in group]} if args.combined else GRIDS
    def overall_mean(component):
        data = [r for r in rows if r['component'] == component and int(r['eligible_instances']) > 0]
        count = sum(int(r['eligible_instances']) for r in data)
        mean = sum(float(r['reward']) * int(r['eligible_instances']) for r in data) / count if count else float('nan')
        return 1 - mean if component == 'p2p_fail_rate' else mean

    for grid, components in grids.items():
        components = sorted(components, key=lambda item: overall_mean(item[0]), reverse=True)
        if args.combined:
            components.insert(0, ('overall_reward', 'Overall reward'))
        print('Row order (overall rollout mean): ' + ', '.join(
            f'{label}={overall_mean(component):.4f}' for component, label in components))
        nrows = len(components)
        if args.horizontal:
            fig, axes = plt.subplots(3, nrows, figsize=(6 * nrows + 3, 17), sharex=True)
        else:
            fig, axes = plt.subplots(nrows, 3, figsize=(26, 4 * nrows + 3) if args.combined else (17, 3 * nrows + 2), sharex=True)
        for row_index, (component, label) in enumerate(components):
            for col_index, (metric, title) in enumerate(COLUMNS):
                ax = axes[col_index, row_index] if args.horizontal else axes[row_index, col_index]
                for run in runs:
                    if run in summary.get('mean_only_runs', []) and metric != 'reward':
                        continue
                    data = sorted((r for r in rows if r['component'] == component and r.get('run', '') == run),
                                  key=lambda r: int(r['step']))
                    by_step = {int(r['step']): float(r[metric]) for r in data}
                    steps = list(range(min(by_step), max(by_step) + 1))
                    raw = [by_step.get(step, float('nan')) for step in steps]
                    # E[pass] = 1-E[fail]; max(pass) = 1-min(fail).
                    if component == 'p2p_fail_rate' and metric != 'nonzero_advantage_instances_pct':
                        raw = [1-v for v in raw]
                    if metric == 'nonzero_advantage_instances_pct':
                        raw = [v / 100 for v in raw]
                    smooth = running_average(raw, window, alignment)
                    # Do not draw a value at steps with no included rollouts.
                    empty_steps = {int(r['step']) for r in data if int(r['eligible_instances']) == 0}
                    smooth = [float('nan') if step in empty_steps else value
                              for step, value in zip(steps, smooth)]
                    if not args.smooth_only:
                        ax.plot(steps, raw, color=colors[run], lw=1, alpha=.3)
                    ax.plot(steps, smooth, color=colors[run], lw=2.2)
                if args.horizontal:
                    if col_index == 0:
                        ax.set_title(label, fontsize=26, pad=20)
                    if row_index == 0:
                        ax.set_ylabel(title.replace(' instances (fraction)', '\ninstances (fraction)'), fontsize=26, labelpad=20)
                    if col_index == 2:
                        ax.set_xlabel(axis_label, fontsize=22)
                else:
                    if row_index == 0:
                        ax.set_title(title.replace(' instances (fraction)', ' instances\n(fraction)'), fontsize=26, pad=20)
                    if col_index == 0:
                        ax.set_ylabel(label, fontsize=26, labelpad=20)
                    if row_index == nrows - 1:
                        ax.set_xlabel(axis_label, fontsize=22)
                ax.grid(alpha=.22)
                ax.margins(x=.015, y=.1)
                if not args.dynamic_ylim:
                    ax.set_ylim(0, 1)
                ax.tick_params(labelsize=20, labelbottom=True)
        heading = 'Dense vs sparse ORM · All rewards' if comparison else grid.replace('_', ' ').title()
        if comparison and len(runs) > 2:
            heading = 'ORM run comparison · All rewards'
        subtitle = ' · common dense overall reward' if comparison else ''
        if skip_first:
            subtitle += ' · selected warmup excluded'
        fig.suptitle(f'{heading} · {alignment.title()} {window}-step averages{subtitle}', fontsize=30, y=.99)
        handles = [Line2D([0], [0], color=colors[run], lw=2.2,
                          label=(run + ' (mean only)' if run in summary.get('mean_only_runs', []) else run)
                          or f'{alignment.title()} {window}-step average') for run in runs]
        if not args.smooth_only:
            handles.append(Line2D([0], [0], color='gray', lw=1, alpha=.3, label='Per training step'))
        fig.legend(handles=handles, loc='upper center', ncol=len(handles), frameon=False,
                   bbox_to_anchor=(.5, .974), fontsize=22)
        fig.subplots_adjust(left=.04 if args.horizontal else .065, right=.985,
                            bottom=.08 if args.horizontal else (.035 if args.combined else .09),
                            top=.88 if args.horizontal else (.94 if args.combined else .84),
                            hspace=.30, wspace=.28)
        suffix = '_horizontal' if args.horizontal else ''
        if args.smooth_only:
            suffix += '_smooth_only'
        if args.dynamic_ylim:
            suffix += '_dynamic_ylim'
        path = output_dir / ('overlapping_reward_grid.png' if comparison else f'{grid}_grid{suffix}.png')
        fig.savefig(path, dpi=180, bbox_inches='tight', pad_inches=.25)
        if comparison:
            fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight', pad_inches=.25)
        plt.close(fig)
        print(path)
    if skip_first or args.output_dir:
        (output_dir/'plot_options.json').write_text(json.dumps({
            'source_metrics': str((args.input_dir/'metrics.csv').resolve()),
            'skip_first_steps': skip_first,
            'rebase_skipped_steps': args.rebase_skipped_steps,
            'window': window, 'alignment': alignment,
            'plotted_step_ranges': {run: [min(int(r['step']) for r in rows if r.get('run', '') == run),
                                          max(int(r['step']) for r in rows if r.get('run', '') == run)]
                                    for run in original_runs},
            'smoothing': 'Computed after excluding warmup; excluded steps never enter running averages',
        }, indent=2)+'\n')


if __name__ == '__main__':
    main()
