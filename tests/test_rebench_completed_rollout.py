import asyncio
import threading
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from openhands.sdk.conversation.exceptions import ConversationRunError
from openhands.sdk.conversation.impl.remote_conversation import RemoteState
from openhands.sdk.event import ActionEvent, AgentErrorEvent, MessageEvent, ObservationEvent
from openhands.sdk.event.conversation_error import ConversationErrorEvent
from openhands.sdk.llm import Message, MessageToolCall, TextContent
from openhands.sdk.tool.builtins.finish import FinishAction, FinishObservation
from platoon.envs.base import Task
from platoon.episode.context import current_trajectory_collection
from platoon.episode.trajectory import TrajectoryCollection
from platoon.openhands import env as openhands_env
from platoon.episode.loop import run_episode
from platoon.issue_resolution import env as env_module
from platoon.openhands.agent import OpenHandsAgent
from openhands.sdk.conversation import RemoteConversation
from openhands.tools.terminal import TerminalAction


def final_events():
    prompt = MessageEvent(source="user", llm_message=Message(role="user", content=[TextContent(text="fix it")]))
    actions = [
        ActionEvent(
            thought=[], action=FinishAction(message="done"), tool_name="finish", tool_call_id=f"call-{i}",
            tool_call=MessageToolCall(id=f"call-{i}", name="finish", arguments='{"message":"done"}', origin="completion"),
            llm_response_id=f"completion-{i // 2}",
        ) for i in range(3)
    ]
    observations = [
        ObservationEvent(action_id=a.id, tool_name="finish", tool_call_id=a.tool_call_id,
                         observation=FinishObservation.from_text("done")) for a in actions
    ]
    # Parallel tool results arrive in reverse order within the first completion.
    return [prompt, actions[0], actions[1], observations[1], observations[0], actions[2], observations[2]]


@pytest.fixture
def scenario(monkeypatch):
    events = final_events()
    server = SimpleNamespace(events=[], status="running", fail_events=False, pages=0)
    conversation_id = uuid.uuid4()

    def handle(request):
        if request.url.path.endswith('/events/search'):
            server.pages += 1
            if server.fail_events:
                return httpx.Response(503)
            # REST history is chronological; reconcile inserts by timestamp.
            server.events[:] = [
                event.model_copy(update={"timestamp": f"2026-09-27T00:00:00.{index:06d}"})
                for index, event in enumerate(server.events)
            ]
            offset = int(request.url.params.get('page_id', '0'))
            page = server.events[offset:offset + 2]
            return httpx.Response(200, json={
                'items': [e.model_dump(mode='json') for e in page],
                'next_page_id': str(offset + 2) if offset + 2 < len(server.events) else None,
            })
        return httpx.Response(200, json={'execution_status': server.status})

    client = httpx.Client(base_url='http://agent', transport=httpx.MockTransport(handle))
    state = RemoteState(client, str(conversation_id))
    conversation = Mock(spec=RemoteConversation,
        id=conversation_id, state=state, workspace=SimpleNamespace(client=client),
        _conversation_action_base_path='/api/conversations', send_message=Mock(),
    )
    monkeypatch.setattr(openhands_env, 'Conversation', Mock(return_value=conversation))
    monkeypatch.setattr(env_module, 'extract_patch_from_environment', Mock(return_value='patch'))
    localization = Mock(return_value=(0.5, {'file_f1': 0.5}))
    test_reward = AsyncMock(return_value=(0.4, {'binary_reward': 1.0}))
    monkeypatch.setattr(env_module, 'compute_localization_reward', localization)
    monkeypatch.setattr(env_module, 'compute_test_execution_reward', test_reward)
    for name in ('compute_tool_json_error_reward', 'compute_agent_error_reward', 'compute_str_replace_reward'):
        monkeypatch.setattr(env_module, name, Mock(return_value=(1.0, {name: 1.0})))
    env = env_module.SWERebenchEnv(
        task=Task(id='test', goal='fix it', max_steps=40, misc={'repo_path': '/testbed'}),
        agent=None, workspace='/tmp', length_penalty_threshold=10, use_binary_reward=True,
        conversation_timeout=1800, wait_for_completion=True,
    )
    env.close = AsyncMock()
    monkeypatch.delenv("PLATOON_OPENHANDS_ACT_DEBUG_DIR", raising=False)
    yield SimpleNamespace(env=env, server=server, conversation=conversation, events=events,
                          localization=localization, test_reward=test_reward)
    client.close()



async def episode(s):
    collection = TrajectoryCollection()
    token = current_trajectory_collection.set(collection)
    try:
        return await asyncio.wait_for(
            asyncio.create_task(run_episode(OpenHandsAgent(), s.env, enforce_step_budget=False)),
            timeout=3,
        )
    finally:
        current_trajectory_collection.reset(token)


@pytest.mark.parametrize('status', ['finished', 'error', 'stuck', 'running'])
@pytest.mark.parametrize('binary', [True, False])
def test_waits_freezes_and_rewards_once(scenario, status, binary):
    s = scenario
    s.env._use_binary_reward = binary
    main_thread = threading.get_ident()

    def run(timeout):
        assert threading.get_ident() != main_thread
        assert timeout == 1800
        s.server.events = s.events
        s.server.status = status
        # Only reconciliation after the thread exits makes events visible.
        assert len(s.conversation.state.events) == 0
        s.test_reward.assert_not_awaited()
        if status != 'finished':
            raise ConversationRunError(s.conversation.id, RuntimeError(status))

    s.conversation.run = run
    trajectory = asyncio.run(episode(s))
    expected = 1.0 if binary else 0.505
    assert trajectory.reward == pytest.approx(expected)
    assert [step.reward for step in trajectory.steps] == pytest.approx([0, 0, expected])
    assert [step.misc['action_misc']['completion_id'] for step in trajectory.steps[1:]] == [
        'completion-0', 'completion-1',
    ]
    assert trajectory.steps[1].action_events == s.events[1:3]
    assert trajectory.steps[1].observation_events == [s.events[4], s.events[3]]
    assert trajectory.error_message == (None if status == 'finished' else status)
    s.test_reward.assert_awaited_once()
    s.localization.assert_called_once()
    assert s.server.pages == 5  # Initial empty fetch + one reconciliation of four pages.
    snapshot = s.env._state.conversation_state.events
    assert isinstance(snapshot, tuple)
    s.conversation.state.events.add_event(final_events()[0])
    assert s.env._state.conversation_state.events == snapshot
    assert len(snapshot) == len(s.events)


@pytest.mark.parametrize('case', ['message_then_finish', 'missing_tool_result', 'partial_batch',
                                  'tool_error', 'trailing_error', 'no_actions', 'empty'])
def test_completed_history_corner_cases(scenario, case):
    s = scenario
    events = s.events
    message = MessageEvent(source='agent', llm_response_id='message-completion',
                           llm_message=Message(role='assistant', content=[TextContent(text='done')]))
    tool = ActionEvent(thought=[], action=TerminalAction(command='true'), tool_name='terminal',
                       tool_call_id='terminal-0', llm_response_id='terminal-completion',
                       tool_call=MessageToolCall(id='terminal-0', name='terminal',
                                                arguments='{"command":"true"}', origin='completion'))
    error = AgentErrorEvent(tool_name='finish', tool_call_id='call-0', error='invalid arguments')
    histories = {
        'message_then_finish': [events[0], message, events[-2]],
        'missing_tool_result': [events[0], tool],
        'partial_batch': events[:3] + [events[3]],
        'tool_error': [events[0], events[1], error],
        'trailing_error': events + [error],
        'no_actions': events[:1],
        'empty': [],
    }
    history = histories[case]

    def run(timeout):
        s.server.events = history
        s.server.status = 'error'
        raise ConversationRunError(s.conversation.id, RuntimeError('Remote conversation ended with error'))

    s.conversation.run = run
    trajectory = asyncio.run(episode(s))
    expected_actions = [e for e in history if isinstance(e, ActionEvent) or (isinstance(e, MessageEvent) and e.source == 'agent')]
    assert [e for step in trajectory.steps for e in (step.action_events or [])] == expected_actions
    assert trajectory.reward == 1.0
    assert trajectory.error_message == 'Remote conversation ended with error'
    assert sum(step.reward != 0 for step in trajectory.steps) == 1
    s.test_reward.assert_awaited_once()
    if case == 'tool_error':
        assert [e.id for e in trajectory.steps[-1].observation_events] == [error.id]


def test_sdk_timeout_is_evaluated_after_thread_exits(scenario):
    s = scenario

    def run(timeout):
        s.server.events = s.events[:3]
        raise ConversationRunError(s.conversation.id, TimeoutError('Run timed out after 1800 seconds.'))

    s.conversation.run = run
    trajectory = asyncio.run(episode(s))
    assert trajectory.error_message == 'Run timed out after 1800 seconds.'
    assert trajectory.reward == 1.0
    s.test_reward.assert_awaited_once()


@pytest.mark.parametrize('code,detail,expected', [
    ('LLMContextWindowExceedError', '64826 prompt tokens + 2048 max_tokens > 65000 context window length.',
     'LLMContextWindowExceedError: 64826 prompt tokens + 2048 max_tokens > 65000 context window length.'),
    ('MaxIterationsReached', 'Agent reached maximum iterations limit (40).',
     'MaxIterationsReached: Agent reached maximum iterations limit (40).'),
    (' ProviderError ', ' ', 'ProviderError'),
    (' ', ' Request failed ', 'Request failed'),
    (' ', ' ', 'Remote conversation ended with error'),
])
def test_error_detail_arriving_after_run_is_logged_and_saved(scenario, caplog, code, detail, expected):
    s = scenario

    def run(timeout):
        s.server.status = 'error'
        s.server.events = s.events + [
            ConversationErrorEvent(source='environment', code='EarlierError', detail='old failure'),
            ConversationErrorEvent(source='environment', code=code, detail=detail),
        ]
        assert len(s.conversation.state.events) == 0
        raise ConversationRunError(s.conversation.id, RuntimeError('Remote conversation ended with error'))

    s.conversation.run = run
    trajectory = asyncio.run(episode(s))
    assert trajectory.error_message == expected
    assert s.env._state.misc['error_message'] == expected
    warnings = [r.getMessage() for r in caplog.records
                if r.name == 'platoon.openhands.env' and r.levelname == 'WARNING']
    assert warnings == [expected]
    assert trajectory.reward == 1.0
    s.test_reward.assert_awaited_once()


def test_no_parsing_or_evaluation_until_thread_exits(scenario, monkeypatch):
    s = scenario
    started, release = threading.Event(), threading.Event()
    parser = Mock(wraps=openhands_env.get_obs_for_last_action)
    monkeypatch.setattr(openhands_env, 'get_obs_for_last_action', parser)

    def run(timeout):
        started.set()
        release.wait(timeout=3)
        s.server.events = s.events
        s.server.status = 'finished'

    s.conversation.run = run

    async def check():
        task = asyncio.create_task(episode(s))
        try:
            while not started.is_set():
                await asyncio.sleep(0.001)
            parser.assert_not_called()
            s.test_reward.assert_not_awaited()
            assert s.server.pages == 1
        finally:
            release.set()
        assert (await task).reward == 1.0

    asyncio.run(check())
