"""Refresh five-run reward, critic EV, and limit overlays including the hariom run.

Uses the established detectors and formulas. Writes new comparison directories,
preserving the original four-run figures. Run with a Python environment containing
numpy, matplotlib, and tokenizers (e.g. the base Miniconda environment).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/data/user_data/adityabs/miles_tinker_rebench/experiments/issue-resolution-rebench-tinker')
NEW_RUN = BASE/'rebench-cispo-critic-b512-g1-k4-20261026-123718'
LABEL = 'CISPO + Turn critic (K=4)'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--new-run', type=Path, default=NEW_RUN)
    parser.add_argument('--label', default=LABEL)
    parser.add_argument('--workers', type=int, default=64, help='Limit-scan worker processes')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    logs = ROOT/'artifacts/comparison_with_cispo_critic_logs'
    logs.mkdir(parents=True, exist_ok=True)
    grid = 'artifacts/orm_comparison_with_cispo_critic'
    ev = 'artifacts/ppo_ev_comparison_with_cispo_critic'
    limits = 'artifacts/limit_audit_with_cispo_critic'
    individual = 'artifacts/cispo_turn_critic_k4_ev'
    dense = str(BASE/'rebench-tinker-dense-orm-20260927-223359')
    sparse = str(BASE/'rebench-tinker-sparse-orm-20260927-224119')
    ppo = str(BASE/'rebench-tinker-ppo-dense-orm-20261017-193039')
    turn = str(BASE/'rebench-ppo-turn-b512-g1-20261021-182628')

    def run(name, command):
        path = logs/(name+'.log')
        print('Starting '+name, flush=True)
        with path.open('w') as stream:
            result = subprocess.run([sys.executable, *map(str, command)], cwd=ROOT,
                                    stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{name} failed: '+path.read_text()[-6000:])
        print('Completed '+name, flush=True)

    warmup = ['--skip-first-steps','PPO + Turn rewards','12',
              '--skip-first-steps',args.label,'12',
              '--rebase-skipped-steps','--allow-empty-after-skip']

    def rewards_and_limits():
        run('reward_grid', ['scripts/reward_plots/plot_orm_comparison.py',dense,sparse,
            '--extra-run','PPO + Dense ORM',ppo,
            '--turn-reward-run','PPO + Turn rewards',turn,
            '--turn-reward-run',args.label,args.new_run,
            '--output-dir',grid,'--skip-last-steps','2',
            '--skip-missing-rewards','--skip-infra-errors','--group-sample-seed','42'])
        run('reward_grid_no_warmup', ['scripts/reward_plots/plot_reward_grids.py',grid,
            '--output-dir',grid+'_no_warmup','--combined','--smooth-only','--dynamic-ylim',
            '--window','16','--alignment','trailing',*warmup])
        run('limit_scan', ['artifacts/limit_audit_with_turn_ppo/refresh.py',
            '--grid-dir',grid,'--output-dir',limits,'--workers',args.workers])
        run('limits', ['artifacts/limit_audit_with_ppo/plot_limits.py',
            '--recover-output','--input-dir',limits])
        run('limits_no_warmup', ['artifacts/limit_audit_with_ppo/plot_limits.py',
            '--recover-output','--input-dir',limits,'--output-dir',limits+'_no_warmup',*warmup])

    def explained_variance():
        run('ppo_ev',['artifacts/ppo_explained_variance/plot.py',ppo])
        run('turn_ppo_ev',['artifacts/ppo_explained_variance/plot.py',turn,
            '--output-dir','artifacts/ppo_ev_oct21_b512_g1','--title','PPO + Turn rewards — critic EV'])
        run('new_critic_ev',['artifacts/ppo_explained_variance/plot.py',args.new_run,
            '--output-dir',individual,'--title',args.label+' — pre-update turn-level EV'])
        run('cispo_baseline_ev',['artifacts/ev_baseline_comparison/plot.py',
            '--granularity','turn','--output-dir','artifacts/ev_baseline_comparison_turn'])
        extra=['--extra-run',args.label,individual+'/per_step.csv']
        run('ev_overlay',['artifacts/ppo_ev_comparison/plot.py','--output-dir',ev,*extra])
        run('ev_overlay_no_warmup',['artifacts/ppo_ev_comparison/plot.py',
            '--output-dir',ev+'_no_warmup',*extra,'--skip-turn-warmup','12',
            '--skip-run-warmup',args.label,'12','--allow-empty-after-skip'])

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(rewards_and_limits),pool.submit(explained_variance)]
        for future in futures:
            future.result()
    for folder, filename in [(grid,'overlapping_reward_grid.png'),(ev,'explained_variance_comparison.png'),
                             (limits,'limit_frequencies_recovered.png')]:
        for suffix in ['', '_no_warmup']:
            print(ROOT/(folder+suffix)/filename, flush=True)


if __name__ == '__main__':
    main()
