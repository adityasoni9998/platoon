# /// script
# dependencies = ["matplotlib"]
# ///
"""Create three reward grids from plot_group_reward_metrics.py's saved CSV.

Usage: uv run analysis/plot_reward_grids.py GROUP_PLOTS_DIRECTORY
Uses the existing data snapshot and centered averages without rescanning rollouts.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

GRIDS = {
    'localization': [('overall_localization', 'Overall localization'), ('file_f1', 'File F1'), ('module_f1', 'Module F1'), ('function_f1', 'Function F1')],
    'testing': [('binary_issue_resolution', 'Binary resolution'), ('f2p_pass_rate', 'F2P pass rate'),
                ('p2p_fail_rate', 'P2P pass rate')],
    'tool_formatting': [('json_format_tool', 'JSON tool format'), ('agent_error_reward', 'Agent error reward'),
                        ('str_replace_reward', 'Str replace reward')],
}
COLUMNS = [('reward', 'Mean reward'), ('best_at_8', 'Max@8'),
           ('nonzero_advantage_instances_pct', 'Non-zero advantage instances (fraction)')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('--combined', action='store_true', help='Create one 11-by-3 figure')
    parser.add_argument('--horizontal', action='store_true', help='Transpose metrics and components')
    parser.add_argument('--smooth-only', action='store_true', help='Show only running averages')
    parser.add_argument('--dynamic-ylim', action='store_true', help='Autoscale each subplot y-axis')
    args = parser.parse_args()
    summary = json.loads((args.input_dir/'summary.json').read_text())
    with (args.input_dir/'metrics.csv').open() as f:
        rows = list(csv.DictReader(f))
    if args.combined and not any(r['component'] == 'overall_reward' for r in rows):
        parser.error('Saved metrics lack overall reward. Regenerate from train_rollout using plot.py.')
    if not any(r['component'] == 'overall_localization' for r in rows):
        parser.error('Saved metrics lack overall localization. Regenerate from train_rollout using plot.py.')
    window = summary['window']
    grids = {'all_rewards': [component for group in GRIDS.values() for component in group]} if args.combined else GRIDS
    def overall_mean(component):
        data = [r for r in rows if r['component'] == component]
        count = sum(int(r['eligible_instances']) for r in data)
        mean = sum(float(r['reward']) * int(r['eligible_instances']) for r in data) / count
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
            data = sorted((r for r in rows if r['component'] == component), key=lambda r: int(r['step']))
            steps = [int(r['step']) for r in data]
            for col_index, (metric, title) in enumerate(COLUMNS):
                raw = [float(r[metric]) for r in data]
                smooth = [float(r[metric+'_centered_average']) for r in data]
                # E[pass] = 1 - E[fail]; max(pass) = 1 - min(fail).
                # Centering reverses advantage signs, preserving nonzero status.
                if component == 'p2p_fail_rate' and metric != 'nonzero_advantage_instances_pct':
                    raw = [1-v for v in raw]
                    smooth = [1-v for v in smooth]
                if metric == 'nonzero_advantage_instances_pct':
                    raw = [v / 100 for v in raw]
                    smooth = [v / 100 for v in smooth]
                ax = axes[col_index, row_index] if args.horizontal else axes[row_index, col_index]
                if not args.smooth_only:
                    ax.plot(steps, raw, color='#9cb9cf', lw=1, alpha=.65)
                ax.plot(steps, smooth, color='#1263a0', lw=2.2)
                if args.horizontal:
                    if col_index == 0:
                        ax.set_title(label, fontsize=26, pad=20)
                    if row_index == 0:
                        ax.set_ylabel(title.replace(' instances (fraction)', '\ninstances (fraction)'), fontsize=26, labelpad=20)
                    if col_index == 2:
                        ax.set_xlabel('Training step', fontsize=22)
                else:
                    if row_index == 0:
                        ax.set_title(title.replace(' instances (fraction)', ' instances\n(fraction)'), fontsize=26, pad=20)
                    if col_index == 0:
                        ax.set_ylabel(label, fontsize=26, labelpad=20)
                    if row_index == nrows - 1:
                        ax.set_xlabel('Training step', fontsize=22)
                ax.grid(alpha=.22)
                ax.margins(x=.015, y=.1)
                if not args.dynamic_ylim:
                    ax.set_ylim(0, 1)
                ax.tick_params(labelsize=20, labelbottom=True)
        fig.suptitle(f'{grid.replace("_", " ").title()} · Centered {window}-step averages', fontsize=38, y=.99)
        handles = [Line2D([0], [0], color='#1263a0', lw=2.2,
                          label=f'Centered {window}-step average')]
        if not args.smooth_only:
            handles.insert(0, Line2D([0], [0], color='#9cb9cf', lw=1.5, label='Per training step'))
        fig.legend(handles=handles, loc='upper center', ncol=len(handles), frameon=False,
                   bbox_to_anchor=(.5, .974), fontsize=22)
        fig.subplots_adjust(left=.04 if args.horizontal else .12, right=.985,
                            bottom=.08 if args.horizontal else (.035 if args.combined else .09),
                            top=.88 if args.horizontal else (.94 if args.combined else .84),
                            hspace=.30, wspace=.28)
        suffix = '_horizontal' if args.horizontal else ''
        if args.smooth_only:
            suffix += '_smooth_only'
        if args.dynamic_ylim:
            suffix += '_dynamic_ylim'
        path = args.input_dir/f'{grid}_grid{suffix}.png'
        fig.savefig(path, dpi=180, bbox_inches='tight', pad_inches=.25)
        plt.close(fig)
        print(path)


if __name__ == '__main__':
    main()
