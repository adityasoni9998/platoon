import asyncio
import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest
from openhands.sdk.event import ActionEvent, AgentErrorEvent, MessageEvent, ObservationEvent
from openhands.sdk.event.hook_execution import HookExecutionEvent
from openhands.sdk.llm import Message, MessageToolCall, TextContent
from openhands.tools.terminal import TerminalAction
from platoon.issue_resolution import env as env_module
from platoon.issue_resolution import turn_snapshots
from platoon.issue_resolution.hooks.capture_patch import capture
from platoon.openhands.types import OpenHandsTrajectoryStep

from platoon.envs.base import Task


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args])


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / 'repo'
    path.mkdir()
    git(path, 'init', '-q')
    git(path, 'config', 'user.email', 'test@example.com')
    git(path, 'config', 'user.name', 'Test')
    (path / 'code.py').write_bytes(b'def f():\r\n    return 1\r\n')
    (path / 'deleted.py').write_text('x = 1\n')
    git(path, 'add', '.')
    git(path, 'commit', '-qm', 'base')
    return path


def read_snapshots(out):
    return [json.loads(line) for line in (out / 'snapshots.jsonl').read_text().splitlines()]


def test_capture_is_independent_of_agent_index_and_applies_to_dirty_testbed(repo, tmp_path):
    base = git(repo, 'rev-parse', 'HEAD').decode().strip()
    out = tmp_path / 'captures'
    # Image leftovers are removed by the existing rollout preparation.
    (repo / 'leftover.md').write_text('image leftover\n')
    git(repo, 'add', 'leftover.md')
    (repo / 'unstaged.md').write_text('another leftover\n')
    git(repo, 'reset', '--hard', base)
    git(repo, 'clean', '-fd')
    baseline = capture(repo, base, out)
    (repo / 'code.py').write_bytes(b'def f():\r\n    return 2\r\n')
    (repo / 'code.py').chmod(0o755)
    (repo / 'new.py').write_text('x = 2\n')
    (repo / 'deleted.py').unlink()
    index = (repo / '.git/index').read_bytes()
    edited = capture(repo, base, out)
    assert (repo / '.git/index').read_bytes() == index
    git(repo, 'add', '-A')
    assert capture(repo, base, out) == edited
    git(repo, 'reset')
    assert capture(repo, base, out) == edited
    git(repo, 'add', '-A')
    git(repo, 'commit', '-qm', 'agent commit')
    index = (repo / '.git/index').read_bytes()
    head = git(repo, 'rev-parse', 'HEAD')
    assert capture(repo, base, out) == edited
    assert (repo / '.git/index').read_bytes() == index
    assert git(repo, 'rev-parse', 'HEAD') == head
    assert len(read_snapshots(out)) == 2
    snapshot = read_snapshots(out)[1]
    assert 'leftover.md' not in snapshot['patch'] and 'unstaged.md' not in snapshot['patch']
    assert snapshot['before_sources']['code.py'] == 'def f():\n    return 1\n'
    assert snapshot['after_sources']['code.py'] == 'def f():\n    return 2\n'
    assert set(snapshot['before_sources']) == {'code.py', 'deleted.py'}
    assert set(snapshot['after_sources']) == {'code.py', 'new.py'}
    assert '+    return 2\r\n' in snapshot['patch']

    testbed = tmp_path / 'testbed'
    subprocess.run(['git', 'clone', '-q', str(repo), str(testbed)], check=True)
    git(testbed, 'reset', '--hard', base)
    (testbed / 'leftover.md').write_text('image leftover\n')
    git(testbed, 'add', 'leftover.md')
    (testbed / 'unstaged.md').write_text('another leftover\n')
    subprocess.run(['git', '-C', str(testbed), 'apply', '--verbose'],
                   input=snapshot['patch'].encode(), check=True, capture_output=True)
    for name in ['code.py', 'new.py']:
        assert (testbed / name).read_bytes() == (repo / name).read_bytes()
    assert not (testbed / 'deleted.py').exists()
    (repo / 'data.bin').write_bytes(b'\0\x01\x02binary')
    capture(repo, base, out)
    binary_patch = read_snapshots(out)[-1]['patch']
    assert 'Binary files /dev/null and b/data.bin differ' in binary_patch
    assert 'GIT binary patch' not in binary_patch
    assert (repo / 'data.bin').read_bytes() == b'\0\x01\x02binary'
    git(repo, 'reset', '--hard', base)
    git(repo, 'clean', '-fd')
    assert capture(repo, base, out) == baseline
    assert len(read_snapshots(out)) == 3


def test_capture_keeps_rename_target_despite_unreadable_added_deleted_sources(repo, tmp_path):
    (repo / 'deleted.py').write_bytes(b'# caf\xe9\n')
    git(repo, 'add', '-A')
    git(repo, 'commit', '-qm', 'non-UTF8 source in baseline')
    base = git(repo, 'rev-parse', 'HEAD').decode().strip()
    (repo / 'deleted.py').unlink()
    (repo / 'added.py').write_bytes(b'# invalid \xff\n')
    (repo / 'code.py').rename(repo / 'renamed.txt')
    out = tmp_path / 'captures'
    capture(repo, base, out)
    snapshot = read_snapshots(out)[0]
    assert snapshot['before_sources'] == {'code.py': 'def f():\n    return 1\n'}
    assert snapshot['after_sources'] == {'renamed.txt': 'def f():\n    return 1\n'}
    assert snapshot['source_errors']
    assert 'a/deleted.py' in snapshot['patch'] and 'b/added.py' in snapshot['patch']


def test_capture_non_utf8_patch_uses_orm_replacement_decoding(repo, tmp_path):
    base = git(repo, 'rev-parse', 'HEAD').decode().strip()
    out = tmp_path / 'captures'
    (repo / 'corpus.txt').write_bytes(b'caf\xe9\n')
    result = capture(repo, base, out)
    snapshot = read_snapshots(out)[0]
    assert snapshot['patch_hash'] == result['patch_hash']
    assert '+caf\ufffd\n' in snapshot['patch']
    assert (repo / 'corpus.txt').read_bytes() == b'caf\xe9\n'


def test_timeout_evaluation_diagnostics_are_json_serializable(monkeypatch, tmp_path):
    from contextlib import nullcontext
    from dataclasses import make_dataclass
    from pathlib import Path
    from unittest.mock import AsyncMock

    from platoon.issue_resolution.test_execution_reward import test_execution_reward as evaluator

    result = make_dataclass('TimeoutResult', [
        ('resolved', bool), ('report', dict), ('log_dir', Path), ('timed_out', bool),
    ])(False, {'timed_out': True}, tmp_path, True)
    remote = SimpleNamespace(aio=AsyncMock(return_value=result))
    monkeypatch.setattr(evaluator.modal.Function, 'from_name', lambda *args: SimpleNamespace(remote=remote))
    monkeypatch.setattr(evaluator.modal, 'enable_output', nullcontext)
    monkeypatch.setattr(evaluator, 'named_agent_server_image_for_instance', lambda instance: 'image')

    reward, info = asyncio.run(evaluator.compute_test_execution_reward('patch', {'instance_id': 'task'}))
    saved = json.loads(json.dumps({'snapshot': info}))['snapshot']
    assert reward == saved['composite_reward'] == 0.0
    assert saved['evaluation_logs']['timed_out'] is True
    assert saved['evaluation_logs']['log_dir'] == str(tmp_path)


class LocalWorkspace:
    def execute_command(self, command, timeout):
        p = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
        return SimpleNamespace(exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr, timeout_occurred=False)

    def file_upload(self, source, destination):
        shutil.copyfile(source, destination)
        return SimpleNamespace(success=True)

    file_download = file_upload


def test_install_and_jsonl_download_keep_sources_and_report_capture_errors(repo, tmp_path, monkeypatch):
    monkeypatch.setattr(turn_snapshots, 'HOOK_DIR', str(tmp_path / 'hooks'))
    workspace = LocalWorkspace()
    instance = {'repo_path': str(repo), 'base_commit': git(repo, 'rev-parse', 'HEAD').decode().strip()}
    config = turn_snapshots.install_capture_hook(workspace, instance)
    hook = config.post_tool_use[0].hooks[0]
    assert not hook.async_
    initial = turn_snapshots.download_snapshots(workspace, tmp_path / 'download')
    assert initial[turn_snapshots.EMPTY_PATCH_HASH]['patch'] == ''
    assert list((tmp_path / 'download').iterdir()) == []
    (repo / 'code.py').write_bytes(b'def f():\r\n    return 3\r\n')
    result = workspace.execute_command(hook.command, timeout=10)
    assert result.exit_code == 0
    patch_hash = json.loads(result.stdout)['patch_hash']
    (repo / 'code.py').write_text('def f():\n    return 4\n')
    snapshots = turn_snapshots.download_snapshots(workspace, tmp_path / 'download')
    assert list((tmp_path / 'download').iterdir()) == []
    assert snapshots[patch_hash]['after_sources']['code.py'].endswith('return 3\n')
    assert '\r\n' in snapshots[patch_hash]['patch']
    assert (repo / 'code.py').read_text().endswith('return 4\n')
    failed = workspace.execute_command(hook.command + ' --base-commit nonexistent-commit', timeout=10)
    assert failed.exit_code == 1
    assert 'nonexistent-commit' in json.loads(failed.stderr)['capture_error']
    monkeypatch.setattr(workspace, 'file_download', lambda *args: SimpleNamespace(success=False, error='failed'))
    with pytest.raises(RuntimeError, match='Cannot download turn snapshots'):
        turn_snapshots.download_snapshots(workspace, tmp_path / 'download')
    assert list((tmp_path / 'download').iterdir()) == []


@pytest.mark.parametrize('unreadable_required_source', [False, True])
def test_captured_sources_match_existing_localization_processor(
    repo, tmp_path, monkeypatch, unreadable_required_source,
):
    import sys

    from platoon.issue_resolution.localization_reward import processing
    from platoon.issue_resolution.localization_reward.localization_reward import (
        compute_localization_reward,
    )

    monkeypatch.setattr(processing, 'REMOTE_PYTHON', sys.executable)
    (repo / 'deleted.py').write_bytes(b'# invalid \xff\n')
    git(repo, 'add', '-A')
    git(repo, 'commit', '-qm', 'unreadable deleted source')
    base = git(repo, 'rev-parse', 'HEAD').decode().strip()
    (repo / 'deleted.py').unlink()
    (repo / 'added.py').write_bytes(b'# different added content \xfe\n')
    (repo / 'code.py').write_bytes(b'def f():\r\n    return 2\r\n')
    if unreadable_required_source:
        (repo / 'code.py').write_bytes(b'def f():\n    return 2 # \xff\n')
    out = tmp_path / 'captures'
    capture(repo, base, out)
    snapshot = read_snapshots(out)[0]
    gold = [{'file': 'code.py', 'changes': {'edited_entities': ['code.f']}}]

    async def test_reward(*args):
        return 0.7, {}

    monkeypatch.setattr(env_module, 'compute_test_execution_reward', test_reward)
    env = SimpleNamespace(_task=SimpleNamespace(misc={'file_changes': gold}),
                          _test_execution_semaphore=asyncio.Semaphore(4))
    actual = asyncio.run(env_module.SWERebenchEnv._score_snapshot(env, snapshot))
    expected = compute_localization_reward(snapshot['patch'], {
        'repo_path': str(repo), 'base_commit': base, 'file_changes': gold,
    }, LocalWorkspace())
    assert actual['localization_reward'] == expected[0]
    assert actual['composite_reward'] == 0.7
    if unreadable_required_source:
        assert actual['localization_reward'] == 0
        assert actual['localization']['status'] == expected[1]['status'] == 'error'
        assert actual['localization']['source_errors']
    else:
        assert actual['localization_reward'] > 0
        assert actual['localization'] == expected[1]


def test_snapshot_test_execution_concurrency(monkeypatch):
    env = env_module.SWERebenchEnv(
        task=Task(id='task'), agent=None, workspace='/tmp',
    )
    active = peak = 0

    async def test_reward(patch, instance):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return float(patch), {}

    monkeypatch.setattr(env_module, 'compute_test_execution_reward', test_reward)
    monkeypatch.setattr(env_module, 'score_sources', lambda *args: (0.0, {}))

    async def score_all():
        return await asyncio.gather(*(env._score_snapshot({
            'patch': str(i), 'before_sources': {}, 'after_sources': {},
        }) for i in range(12)))

    results = asyncio.run(score_all())
    assert peak == 8 and active == 0
    assert [result['composite_reward'] for result in results] == list(range(12))


def test_workspace_preparation_installs_hook_after_cleanup(monkeypatch):
    from platoon.issue_resolution import rollout

    commands = []

    def execute(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(exit_code=0)

    workspace = SimpleNamespace(execute_command=execute)
    monkeypatch.setattr(rollout, '_modal_workspace', lambda *args, **kwargs: workspace)

    def install(ws, instance):
        assert ws is workspace
        assert 'git reset --hard HEAD && git clean -fd' in commands[-1]
        assert instance['base_commit'] == 'base' and instance['repo_path'] == '/testbed'
        return 'hook-config'

    monkeypatch.setattr(rollout, 'install_capture_hook', install)
    result = rollout.prepare_workspace({'instance_id': 'test', 'base_commit': 'base'}, turn_rewards=True)
    assert result == (workspace, 'hook-config')
    assert rollout.prepare_workspace({'instance_id': 'test'}) == (workspace, None)


def action(cid, call, *, valid=True):
    return ActionEvent(
        thought=[], action=TerminalAction(command='true') if valid else None,
        tool_name='terminal', tool_call_id=call, llm_response_id=cid,
        tool_call=MessageToolCall(id=call, name='terminal', arguments='{"command":"true"}', origin='completion'),
    )


def hook(a, patch_hash, *, success=True):
    return HookExecutionEvent(hook_event_type='PostToolUse', hook_command='capture',
        action_id=a.id, success=success, exit_code=0 if success else 1,
        stdout=json.dumps({'patch_hash': patch_hash}), stderr='' if success else 'git failed')


def test_existing_completion_mapping_handles_parallel_tools_and_missing_captures():
    a, b, rejected = action('one', 'a'), action('one', 'b'), action('two', 'c', valid=False)
    events = [a, b, hook(b, 'edited'), hook(a, 'edited'), rejected]
    assert turn_snapshots.completion_snapshots(events, 'capture') == {'one': 'edited'}
    # A runtime error has no hook, but a sibling's hook captures the whole batch.
    error = AgentErrorEvent(tool_name='terminal', tool_call_id=b.tool_call_id, error='failed')
    assert turn_snapshots.completion_snapshots([a, b, hook(a, 'edited'), error], 'capture') == {'one': 'edited'}
    with pytest.raises(ValueError, match='git failed'):
        turn_snapshots.completion_snapshots([a, hook(a, '', success=False)], 'capture')
    assert turn_snapshots.completion_snapshots([a], 'capture') == {}


def turn_env(monkeypatch):
    env = env_module.SWERebenchEnv(task=Task(id='task'), agent=None, workspace='/tmp')
    env._turn_rewards_enabled = True
    env._state = SimpleNamespace(final=False)
    monkeypatch.setattr(env_module, 'is_finished', lambda state: state.final)
    return env


def evaluate_scores(env, scores):
    result = []
    for i, (loc, test) in enumerate(scores):
        cid = str(i)
        env._snapshot_scores[cid] = {'localization_reward': loc, 'composite_reward': test}
        env._completion_snapshots[cid] = cid
        env._current_step = OpenHandsTrajectoryStep(misc={'action_misc': {'completion_id': cid}})
        env._state.final = i == len(scores) - 1
        result.append(asyncio.run(env.evaluate()))
    return result


@pytest.mark.parametrize('scores,expected', [
    ([(.4, .8), (.1, .3), (.4, .8), (.2, .3)], [.62, 0, 0, -.38]),
    ([(.4, .8), (.1, .3), (.1, .3)], [.62, 0, -.395]),
    ([(0, -.1)], [-.07]),
    ([(.4, .8), (0, 0)], [.62, -.62]),
])
def test_per_step_settlement(monkeypatch, scores, expected):
    env = turn_env(monkeypatch)
    result = evaluate_scores(env, scores)
    assert [reward for reward, _ in result] == pytest.approx(expected)
    assert sum(reward for reward, _ in result) == pytest.approx(.15 * scores[-1][0] + .70 * scores[-1][1])


def test_no_hook_inherits_previous_state_and_format_penalties_are_per_category(monkeypatch):
    env = turn_env(monkeypatch)
    env._previous_patch_hash = 'edited'
    env._snapshot_scores['edited'] = {'localization_reward': .4, 'composite_reward': .8}
    env._best_localization, env._best_test = .4, .8
    env._current_step = OpenHandsTrajectoryStep(
        misc={'action_misc': {'completion_id': 'message'}},
        action_events=[MessageEvent(source='agent', llm_response_id='message',
            llm_message=Message(role='assistant', content=[TextContent(text='<tool_call>bad')]))],
        observation_events=[AgentErrorEvent(tool_name='terminal', tool_call_id=str(i), error='bad') for i in range(2)]
        + [ObservationEvent.model_construct(tool_name='file_editor',
                observation=SimpleNamespace(is_error=True)) for _ in range(2)],
    )
    reward, info = asyncio.run(env.evaluate())
    assert reward == pytest.approx(-.15)
    assert info['snapshot_unchanged'] and info['localization_delta'] == 0


def test_runtime_error_without_capture_allows_later_recovery(monkeypatch):
    env = turn_env(monkeypatch)
    failed, recovered = action('failed', 'a'), action('recovered', 'b')
    error = AgentErrorEvent(tool_name='terminal', tool_call_id=failed.tool_call_id, error='failed')
    observation = ObservationEvent.model_construct(tool_name='terminal', action_id=recovered.id)
    env._completion_snapshots = turn_snapshots.completion_snapshots(
        [failed, error, recovered, observation, hook(recovered, 'edited')], 'capture',
    )
    env._snapshot_scores = {
        turn_snapshots.EMPTY_PATCH_HASH: {'localization_reward': 0.0, 'composite_reward': 0.0},
        'edited': {'localization_reward': .4, 'composite_reward': .8},
    }
    env._current_step = OpenHandsTrajectoryStep(
        action_events=[failed], observation_events=[error],
        misc={'action_misc': {'completion_id': 'failed'}},
    )
    reward, info = asyncio.run(env.evaluate())
    assert reward == pytest.approx(-.05)
    assert info['snapshot_unchanged'] and info['test_execution_delta'] == info['localization_delta'] == 0
    env._state.final = True
    env._current_step = OpenHandsTrajectoryStep(
        action_events=[recovered], observation_events=[observation],
        misc={'action_misc': {'completion_id': 'recovered'}},
    )
    reward, info = asyncio.run(env.evaluate())
    assert reward == pytest.approx(.62)
    assert info['patch_hash'] == 'edited' and env.turn_reward_error is None


@pytest.mark.parametrize('gamma,gae_lambda,expected', [
    (1.0, 1.0, [.03, -.03, -.03]),
    (.99, .9, [.03618357, -.02673, -.03]),
])
def test_final_settlement_reaches_earlier_turns_through_ppo_gae(monkeypatch, gamma, gae_lambda, expected):
    import tinker
    import torch

    from platoon.train.tinker.ppo import prepare_batch
    from platoon.train.tinker.proxy import TinkerLLMInteraction
    from platoon.utils.tinker_data_processing import trajectory_to_data

    env = turn_env(monkeypatch)
    deltas = [r for r, _ in evaluate_scores(env, [(.4, 0), (.1, 0), (.2, 0)])]
    interactions = {str(i): TinkerLLMInteraction(
        tinker.ModelInput.from_ints([10 + i, 20 + i]),
        SimpleNamespace(tokens=[30 + i], logprobs=[-.1]),
    ) for i in range(3)}
    result = trajectory_to_data(
        trajectory={'steps': [{'reward': r, 'misc': {'action_misc': {'completion_id': str(i)}}}
                              for i, r in enumerate(deltas)]},
        interactions=interactions, task_id='task', trajectory_id='trajectory',
        trajectory_advantage=sum(deltas), checkpoint_version=0,
        trajectory_reward=sum(deltas), ppo_reward_mode='turn',
    )
    predictions = [{'values': tinker.TensorData.from_torch(torch.zeros(d.model_input.length))}
                   for d in result.datums]
    prepared, _ = prepare_batch(result.datums, predictions, 'none', value_granularity='turn',
                              gamma=gamma, gae_lambda=gae_lambda)
    for datum, advantage in zip(prepared, expected, strict=True):
        assert datum.loss_fn_inputs['advantages'].to_torch()[-1].item() == pytest.approx(advantage)
