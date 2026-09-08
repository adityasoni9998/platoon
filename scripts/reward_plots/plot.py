# /// script
# dependencies = ["matplotlib"]
# ///
"""Generate the vertical reward dashboard from rollouts or saved metrics.

uv run scripts/reward_plots/plot.py /path/to/train_rollout
uv run scripts/reward_plots/plot.py /path/to/group_reward_plots_centered_k16
"""
import argparse
from pathlib import Path
import sys

from plot_group_reward_metrics import main as aggregate
from plot_reward_grids import main as render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('--window', type=int, default=16)
    parser.add_argument('--skip-last-steps', type=int, default=2)
    parser.add_argument('--output-dir', type=Path, help='Output directory when processing raw rollouts')
    parser.add_argument('--horizontal', action='store_true')
    parser.add_argument('--smooth-only', action='store_true')
    parser.add_argument('--dynamic-ylim', action='store_true')
    args = parser.parse_args()
    if args.window < 1 or args.skip_last_steps < 0:
        parser.error('window must be positive and skip-last-steps must be nonnegative')
    original_argv = sys.argv
    try:
        if (args.input_dir / 'metrics.csv').is_file():
            if args.output_dir:
                parser.error('--output-dir applies only to raw rollout input')
            output = args.input_dir
        else:
            output = args.output_dir or args.input_dir.parent / f'group_reward_plots_centered_k{args.window}'
            sys.argv = ['plot_group_reward_metrics.py', str(args.input_dir), '--window', str(args.window),
                        '--skip-last-steps', str(args.skip_last_steps), '--output-dir', str(output)]
            aggregate()
        sys.argv = ['plot_reward_grids.py', str(output), '--combined']
        if args.horizontal:
            sys.argv.append('--horizontal')
        if args.smooth_only:
            sys.argv.append('--smooth-only')
        if args.dynamic_ylim:
            sys.argv.append('--dynamic-ylim')
        render()
    finally:
        sys.argv = original_argv


if __name__ == '__main__':
    main()
