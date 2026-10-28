"""Extract common dense ORM scores, then render with the existing grid plotter."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import csv
import json
import hashlib
import time
import math
import re
import random
from pathlib import Path
from statistics import fmean

from plot_group_reward_metrics import read_rollout, group_statistics
from plot_rollout_rewards import running_average

COMPONENTS = [
    ('overall_reward', 'Overall reward (dense formula)'),
    ('overall_localization', 'Overall localization'),
    ('file_f1', 'File F1'), ('module_f1', 'Module F1'), ('function_f1', 'Function F1'),
    ('binary_issue_resolution', 'Binary resolution'), ('f2p_pass_rate', 'F2P pass rate'),
    ('p2p_pass_rate', 'P2P pass rate'), ('json_format_tool', 'JSON tool format'),
    ('agent_error_reward', 'Agent error reward'), ('str_replace_reward', 'Str replace reward'),
]
COLUMNS = [('mean', 'Mean reward'), ('max_at_8', 'Max@8'),
           ('nonzero_advantage', 'Nonzero advantage (rollout fraction)')]
FORMULA = '0.70*(0.5*binary + 0.5*F2P_pass - 0.1*P2P_fail) + 0.15*mean(file_F1,module_F1,function_F1) + 0.05*(JSON_tool + agent_error + str_replace)'


def extract(path, *, skip_missing_rewards=False, skip_infra_errors=False, turn_rewards=False):
    audit = Counter()
    if turn_rewards:
        from turn_reward_orm import read_turn_rollout
        task, trajectory, finished, r = read_turn_rollout(path, audit)
    else:
        task, trajectory, finished, r = read_rollout(path, audit)
    if not finished:
        return task, trajectory, False, None, None, dict(audit)
    if skip_infra_errors and audit.get('llm_infrastructure_error_rollouts', 0):
        audit['infrastructure_error_rollouts_excluded'] += 1
        return task, trajectory, True, None, r.get('overall_reward'), dict(audit)
    if skip_missing_rewards and not audit.get('rollouts_with_reward_dict', 0):
        audit['missing_reward_dict_rollouts'] += 1
        return task, trajectory, True, None, r.get('overall_reward'), dict(audit)
    if task is None:
        match = re.fullmatch(r'events_(.+)_([0-9a-fA-F-]{36})\.jsonl', path.name)
        if not match:
            raise ValueError(f'Cannot identify task: {path}')
        task = match.group(1)
        audit['task_ids_recovered_from_filename'] += 1
    trajectory = trajectory or str(path)
    loc = r.get('localization') or {}
    def val(mapping, key):
        value = mapping.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            audit['defaulted_' + key] += 1
            audit['defaulted_component_values'] += 1
            return 1.0 if key == 'p2p_fail_fraction' else 0.0
        return float(value)
    v = {
        'file_f1': val(loc, 'file_f1_reward'),
        'module_f1': val(loc, 'module_f1_reward'),
        'function_f1': val(loc, 'entity_f1_reward'),
        'binary_issue_resolution': val(r, 'binary_reward'),
        'f2p_pass_rate': val(r, 'f2p_pass_fraction'),
        'p2p_pass_rate': 1-val(r, 'p2p_fail_fraction'),
        'json_format_tool': val(r, 'tool_json_error_reward'),
        'agent_error_reward': val(r, 'agent_error_event_reward'),
        'str_replace_reward': val(r, 'str_replace_reward'),
    }
    v['overall_localization'] = fmean(v[k] for k in ('file_f1', 'module_f1', 'function_f1'))
    test = .5*v['binary_issue_resolution'] + .5*v['f2p_pass_rate'] - .1*(1-v['p2p_pass_rate'])
    v['overall_reward'] = .7*test + .15*v['overall_localization'] + .05*sum(v[k] for k in ('json_format_tool', 'agent_error_reward', 'str_replace_reward'))
    if audit['defaulted_component_values']:
        audit['rollouts_with_defaults'] += 1
    else:
        if isinstance(r.get('composite_reward'), (int, float)):
            assert math.isclose(test, r['composite_reward'], abs_tol=1e-10)
        if isinstance(r.get('localization_reward'), (int, float)):
            assert math.isclose(v['overall_localization'], r['localization_reward'], abs_tol=1e-10)
    return task, trajectory, finished, v, r.get('overall_reward'), dict(audit)



def sample_group(group, *, run, step, task, seed=42):
    """Select up to eight distinct rollout IDs, independent of traversal order."""
    ordered = sorted(group, key=lambda item: item[0])
    if len(ordered) <= 8:
        return ordered
    key = json.dumps([seed, run, step, task], separators=(',', ':'))
    rng = random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest(), 'big'))
    return rng.sample(ordered, 8)


def step_signature(paths):
    """Detect new, removed, replaced, or appended event files without reading logs."""
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        digest.update(f'{path.name}:{stat.st_size}:{stat.st_mtime_ns}:{stat.st_ctime_ns}\n'.encode())
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dense', type=Path)
    p.add_argument('sparse', type=Path)
    p.add_argument('--extra-run', nargs=2, action='append', default=[],
                   metavar=('LABEL', 'PATH'),
                   help='Overlay another run using the same calculations; accepts its root or rollouts/train directory (repeatable, at most five runs total)')
    p.add_argument('--turn-reward-run', nargs=2, action='append', default=[],
                   metavar=('LABEL', 'PATH'),
                   help='Overlay reconstructed dense ORM from a turn-reward run in mean-reward panels only; final patch state plus whole-trajectory formatting rewards')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--dense-steps', type=int, help='Optional cap; default includes all available steps')
    p.add_argument('--sparse-steps', type=int, help='Optional cap; default includes all available steps')
    p.add_argument('--skip-last-steps', type=int, default=0, help='Omit the last N available steps from each run')
    p.add_argument('--skip-first-steps', nargs=2, action='append', default=[],
                   metavar=('RUN', 'N'), help='Exclude steps below N for the named run in the rendered grid, before smoothing; saved raw metrics remain complete')
    p.add_argument('--rebase-skipped-steps', action='store_true',
                   help='Subtract skipped initial steps from the selected runs\' plotted step numbers')
    p.add_argument('--skip-missing-rewards', action='store_true',
                   help='Exclude finished trajectories without a nonempty root step.misc.reward_misc dictionary; default missing fields in partial dictionaries')
    p.add_argument('--skip-infra-errors', action='store_true',
                   help='Exclude root finish events with LLMServiceUnavailableError or LLMTimeoutError; retain ordinary tool/agent failures')
    p.add_argument('--group-sample-seed', type=int, default=42,
                   help='Seed for selecting eight eligible rollouts per task/step when restarts produce larger groups (default: 42)')
    args = p.parse_args()
    if args.skip_last_steps < 0:
        p.error('--skip-last-steps must be nonnegative')
    runs = [('Dense ORM', args.dense, args.dense_steps),
            ('Sparse ORM', args.sparse, args.sparse_steps)]
    turn_runs = {label for label, _ in args.turn_reward_run}
    for label, path in args.extra_run + args.turn_reward_run:
        root = Path(path)
        if root.name == 'train' and root.parent.name == 'rollouts':
            root = root.parent.parent
        runs.append((label, root, None))
    if len(runs) > 5 or len({name for name, _, _ in runs}) != len(runs):
        p.error('Use at most five runs with distinct labels')
    if len({root.resolve() for _, root, _ in runs}) != len(runs):
        p.error('Run directories must be distinct')
    skip_labels = set()
    for label, count in args.skip_first_steps:
        if label not in {name for name, _, _ in runs} or label in skip_labels:
            p.error('--skip-first-steps requires distinct, existing run labels')
        try:
            if int(count) < 0:
                raise ValueError
        except ValueError:
            p.error('--skip-first-steps requires nonnegative integer counts')
        skip_labels.add(label)
    if args.rebase_skipped_steps and not args.skip_first_steps:
        p.error('--rebase-skipped-steps requires --skip-first-steps')
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    # Include calculation dependencies so code changes cannot reuse stale metrics.
    code_hash = hashlib.sha256(b''.join(
        (Path(__file__).parent / filename).read_bytes()
        for filename in ('plot_orm_comparison.py', 'plot_group_reward_metrics.py', 'plot_rollout_rewards.py', 'turn_reward_orm.py')
    )).hexdigest()
    cache_path = out / 'step_cache.json'
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    if cache.get('code_hash') != code_hash:
        cache = {'code_hash': code_hash, 'steps': {}}
    rows, audits, selected, skipped = [], {}, {}, {}
    cache_hits = rescanned = 0
    for name, root, count in runs:
        run_audit = Counter()
        steps = sorted(int(folder.name) for folder in (root/'rollouts/train').iterdir()
                       if folder.is_dir() and folder.name.isdigit()
                       and (count is None or int(folder.name) < count))
        skipped[name] = steps[-args.skip_last_steps:] if args.skip_last_steps else []
        if args.skip_last_steps:
            steps = steps[:-args.skip_last_steps]
        if not steps:
            p.error(f'No selected steps for {name}')
        selected[name] = steps
        for step in steps:
            audit = Counter()
            paths = sorted((root/'rollouts'/'train'/str(step)/'events').glob('*.jsonl'))
            if not paths:
                raise ValueError(f'No events for {name} step {step}')
            key = f'{root.resolve()}:{step}:skip_missing_rewards={args.skip_missing_rewards}:skip_infra_errors={args.skip_infra_errors}:group_sample_seed={args.group_sample_seed}'
            if name not in ('Dense ORM', 'Sparse ORM'):
                key += f':run={name}'
            if name in turn_runs:
                key += ':reward_mode=turn_reconstructed_mean_only'
            signature = step_signature(paths)
            cached = cache['steps'].get(key)
            if cached and cached['signature'] == signature:
                rows.extend(dict(row) for row in cached['rows'])
                run_audit.update(cached['audit'])
                cache_hits += 1
                continue
            rescanned += 1
            first_row = len(rows)
            groups = defaultdict(list)
            seen = set()
            # Turn reconstruction parses every event and is CPU-bound; processes
            # parallelize JSON decoding while ordinary-run behavior stays intact.
            executor = ProcessPoolExecutor if name in turn_runs else ThreadPoolExecutor
            with executor(max_workers=8) as pool:
                for task, trajectory, finished, v, logged, errors in pool.map(partial(extract, skip_missing_rewards=args.skip_missing_rewards, skip_infra_errors=args.skip_infra_errors, turn_rewards=name in turn_runs), paths, chunksize=8):
                    audit.update(errors)
                    audit['observed_rollouts'] += 1
                    if not finished:
                        audit['unfinished_rollouts'] += 1
                        audit['excluded_rollouts'] += 1
                        continue
                    if v is None:
                        audit['excluded_rollouts'] += 1
                        continue
                    if trajectory in seen:
                        raise ValueError(f'Duplicate trajectory: {trajectory}')
                    seen.add(trajectory)
                    groups[task].append((trajectory, v))
                    if name == 'Dense ORM' and not errors.get('rollouts_with_defaults', 0):
                        assert math.isclose(v['overall_reward'], logged, abs_tol=1e-9), (step, trajectory, v['overall_reward'], logged)
                        audit['dense_formula_matches'] += 1
            # Select once before component statistics so every panel uses the same rollouts.
            audit['eligible_rollouts_before_sampling'] += sum(map(len, groups.values()))
            audit['groups_over_eight_before_sampling'] += sum(len(g) > 8 for g in groups.values())
            eligible = [[v for _, v in sample_group(group, run=name, step=step, task=task,
                                                    seed=args.group_sample_seed)]
                        for task, group in groups.items()]
            dropped = sum(map(len, groups.values())) - sum(map(len, eligible))
            audit['rollouts_excluded_by_group_sampling'] += dropped
            audit['excluded_rollouts'] += dropped
            audit['included_rollouts'] += sum(map(len, eligible))
            audit['excluded_rollouts'] += 0
            audit['eligible_groups'] += len(eligible)
            audit['groups_not_exactly_eight'] += sum(len(g) != 8 for g in eligible)
            for component, _ in COMPONENTS:
                samples = [[v[component] for v in g] for g in eligible]
                # Reuse the existing plotter's aggregation, including partial and zero-variance groups.
                original = group_statistics(samples)
                stats = {'mean': original['reward'],
                         'max_at_8': original['best_at_8'],
                         'nonzero_advantage': original['nonzero_advantage_instances_pct'] / 100}
                if name in turn_runs:
                    stats['max_at_8'] = stats['nonzero_advantage'] = math.nan
                rows.append(dict(run=name, step=step, component=component, groups=len(eligible), rollouts=sum(map(len, eligible)), **stats))
            run_audit.update(audit)
            after_paths = sorted((root/'rollouts/train'/str(step)/'events').glob('*.jsonl'))
            if signature == step_signature(after_paths):
                cache['steps'][key] = {'signature': signature, 'audit': dict(audit),
                                       'rows': [dict(row) for row in rows[first_row:]]}
            else:
                cache['steps'].pop(key, None)
            temporary = cache_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(cache))
            temporary.replace(cache_path)
            print(f'{name}: step {step}, {audit['included_rollouts']}/{len(paths)} finished rollouts, {len(eligible)} groups; {audit['excluded_rollouts']} excluded ({audit.get('unfinished_rollouts', 0)} unfinished, {audit.get('missing_reward_dict_rollouts', 0)} missing reward dict, {audit.get('infrastructure_error_rollouts_excluded', 0)} infrastructure errors, {dropped} excess group members)', flush=True)
        audits[name] = dict(run_audit)
    for name, _, _ in runs:
        for component, _ in COMPONENTS:
            data = [r for r in rows if r['run'] == name and r['component'] == component]
            for metric, _ in COLUMNS:
                for r, value in zip(data, running_average([r[metric] for r in data], 16, 'centered')):
                    r[metric+'_centered_16'] = value if r['rollouts'] else math.nan
    with (out/'metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'dense_source': str(args.dense), 'sparse_source': str(args.sparse),
               'dense_steps': [min(selected['Dense ORM']), max(selected['Dense ORM'])],
               'sparse_steps': [min(selected['Sparse ORM']), max(selected['Sparse ORM'])],
               'selected_steps': selected,
               'skipped_steps': skipped,
               'refresh': {'cached_steps': cache_hits, 'rescanned_steps': rescanned,
                           'data_seconds': round(time.monotonic()-started, 2)},
               'formula': FORMULA, 'window': 16, 'alignment': 'centered',
               'smoothing': '16-step centered mean; partial initial windows; no extrapolation',
               'skip_missing_rewards': args.skip_missing_rewards,
               'skip_infra_errors': args.skip_infra_errors,
               'infrastructure_error_filter': 'Exclude root finish error_message starting with LLMServiceUnavailableError: or LLMTimeoutError:' if args.skip_infra_errors else 'disabled',
               'population': 'Root trajectory_finished required; partial and zero-variance groups retained.' + (' Nonempty root step.misc.reward_misc required; partial dictionaries use defaults.' if args.skip_missing_rewards else ' Finished trajectories without reward diagnostics retained with defaults.'),
               'nonzero_advantage': 'abs(component reward - within-group mean) > 1e-10; overall uses recomputed dense reward for both runs',
               'defaults': 'Missing/nonfinite component rewards = 0; P2P fail fraction = 1. Overall recomputed from defaulted components (all missing gives -0.07).',
               'group_sample_seed': args.group_sample_seed,
               'group_sampling': 'After filtering, sample eight rollouts without replacement per step/task when more than eight are available; retain all smaller groups. One reproducible selection shared by all components and statistics.',
               'max_at_8': 'Mean maximum of the selected up-to-eight rollouts per step/task; partial groups are not padded.',
               'audit': audits}
    if args.extra_run or args.turn_reward_run:
        summary['extra_runs'] = {name: str(root) for name, root, _ in runs[2:]}
    if turn_runs:
        summary['mean_only_runs'] = sorted(turn_runs)
        summary['turn_reward_reconstruction'] = (
            'Final completion-bearing root turn must have reward_mode=turn and final_settlement=true. '
            'Use its test/localization state with the original dense weights, not turn deltas or accumulated reward. '
            'Recompute JSON format from the final agent response, agent-error reward from all unique root events, '
            'and editor reward as 1-total_editor_errors/total_editor_calls (zero if no calls). '
            'Missing final state is treated as missing reward diagnostics; ordinary component defaults are unchanged. '
            'Max@8 and nonzero-advantage statistics are omitted for these runs.')
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    from plot_reward_grids import main as render
    render_options = [value for pair in args.skip_first_steps for value in ('--skip-first-steps', *pair)]
    if args.rebase_skipped_steps:
        render_options.append('--rebase-skipped-steps')
    render([str(out), '--combined', '--smooth-only', '--dynamic-ylim',
            '--window', '16', '--alignment', 'trailing', *render_options])
    print(json.dumps(summary, indent=2), flush=True)
    print(f'Total including rendering: {time.monotonic()-started:.2f}s', flush=True)


if __name__ == '__main__':
    main()
