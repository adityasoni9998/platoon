"""Extract common dense ORM scores, then render with the existing grid plotter."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
import re
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


def extract(path):
    audit = Counter()
    task, trajectory, finished, r = read_rollout(path, audit)
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
        assert math.isclose(test, r['composite_reward'], abs_tol=1e-10)
        assert math.isclose(v['overall_localization'], r['localization_reward'], abs_tol=1e-10)
    return task, trajectory, finished, v, r.get('overall_reward'), dict(audit)



def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dense', type=Path)
    p.add_argument('sparse', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--dense-steps', type=int, default=20)
    p.add_argument('--sparse-steps', type=int, default=50)
    args = p.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rows, audits = [], {}
    for name, root, count in [('Dense ORM', args.dense, args.dense_steps), ('Sparse ORM', args.sparse, args.sparse_steps)]:
        audit = Counter()
        for step in range(count):
            paths = sorted((root/'rollouts'/'train'/str(step)/'events').glob('*.jsonl'))
            if not paths:
                raise ValueError(f'No events for {name} step {step}')
            groups = defaultdict(list)
            seen = set()
            with ThreadPoolExecutor(max_workers=8) as pool:
                for task, trajectory, finished, v, logged, errors in pool.map(extract, paths):
                    audit.update(errors)
                    audit['observed_rollouts'] += 1
                    if trajectory in seen:
                        raise ValueError(f'Duplicate trajectory: {trajectory}')
                    seen.add(trajectory)
                    groups[task].append(v)
                    if not finished:
                        audit['unfinished_rollouts'] += 1
                    elif name == 'Dense ORM' and not errors.get('rollouts_with_defaults', 0):
                        assert math.isclose(v['overall_reward'], logged, abs_tol=1e-9), (step, trajectory, v['overall_reward'], logged)
                        audit['dense_formula_matches'] += 1
            eligible = list(groups.values())
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
                rows.append(dict(run=name, step=step, component=component, groups=len(eligible), rollouts=sum(map(len, eligible)), **stats))
            print(f'{name}: step {step}, {len(paths)} rollouts, {len(eligible)} groups (all included)', flush=True)
        audits[name] = dict(audit)
    for name in ('Dense ORM', 'Sparse ORM'):
        for component, _ in COMPONENTS:
            data = [r for r in rows if r['run'] == name and r['component'] == component]
            for metric, _ in COLUMNS:
                for r, value in zip(data, running_average([r[metric] for r in data], 8, 'trailing')):
                    r[metric+'_trailing_8'] = value
    with (out/'metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'dense_source': str(args.dense), 'sparse_source': str(args.sparse),
               'dense_steps': [0, args.dense_steps-1], 'sparse_steps': [0, args.sparse_steps-1],
               'formula': FORMULA, 'smoothing': '8-step trailing mean; partial initial windows; no extrapolation',
               'population': 'Every recorded rollout, including unfinished, missing-reward, and zero-variance groups; identical population for all columns',
               'nonzero_advantage': 'abs(component reward - within-group mean) > 1e-10; overall uses recomputed dense reward for both runs',
               'defaults': 'Missing/nonfinite component rewards = 0; P2P fail fraction = 1. Overall recomputed from defaulted components (all missing gives -0.07).',
               'max_at_8': 'Mean observed group maximum; groups keyed by step/task. Use all recorded members for groups not exactly eight, without padding or truncation.',
               'audit': audits}
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    from plot_reward_grids import main as render
    render([str(out), '--combined', '--smooth-only', '--dynamic-ylim',
            '--window', '8', '--alignment', 'trailing'])
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
