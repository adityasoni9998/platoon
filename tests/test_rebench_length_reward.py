from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / "plugins/issue-resolution-rebench/platoon/issue_resolution/env.py"


@pytest.fixture
def env_module(monkeypatch):
    class BaseEnv:
        def __init__(self, task, agent, workspace, **kwargs):
            self._task = task
            self._workspace = workspace

    modules = {
        "openhands.sdk.workspace": {"BaseWorkspace": object},
        "platoon.openhands.env": {"OpenHandsEnv": BaseEnv},
        "platoon.utils.openhands_utils": {
            "is_action": lambda event: event.kind == "action"
            or (event.kind == "message" and event.source == "agent"),
            "is_finished": lambda state: state.finished,
        },
        "platoon.issue_resolution.test_execution_reward.test_execution_reward": {
            "compute_test_execution_reward": AsyncMock(return_value=(1.0, {"binary_reward": 1.0})),
        },
    }
    for name, attrs in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("rebench_length_reward_test_env", ENV_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "extract_patch_from_environment", lambda **kwargs: "patch")
    return module


def _event(response_id, *, kind="action", source="agent"):
    return SimpleNamespace(kind=kind, source=source, llm_response_id=response_id)


def _env(module, turns=10, *, finished=True, enable_length_penalty=True):
    env = module.SWERebenchEnv(
        task=SimpleNamespace(max_steps=40, misc={"repo_path": "/testbed"}),
        agent=object(),
        workspace=object(),
        length_penalty_threshold=10,
        enable_length_penalty=enable_length_penalty,
    )
    env._state = SimpleNamespace(
        finished=finished,
        reward=0.0,
        conversation_state=SimpleNamespace(events=[_event(f"response-{i}") for i in range(turns)]),
    )
    return env


@pytest.mark.parametrize("turns,length_reward", [(1, 0.3), (10, 0.0), (25, -0.5), (40, -1.0), (43, -1.1)])
@pytest.mark.parametrize("binary_reward", [0.0, 1.0])
def test_terminal_reward_uses_exact_unclamped_formula(env_module, turns, length_reward, binary_reward):
    env_module.compute_test_execution_reward.return_value = (binary_reward, {"binary_reward": binary_reward})
    reward, info = asyncio.run(_env(env_module, turns).evaluate())

    assert reward == pytest.approx(binary_reward + length_reward)
    assert info["length_reward"] == pytest.approx(length_reward)
    assert info["binary_reward"] == binary_reward
    assert info["length_reward_weight"] == 1.0
    assert info["num_agent_turns"] == turns


def test_turn_count_deduplicates_tool_calls_and_includes_agent_messages(env_module):
    env = _env(env_module, turns=0)
    env._state.conversation_state.events = [
        _event("tools-1"),
        _event("tools-1"),
        _event("tools-1", kind="message"),
        _event("final-answer", kind="message"),
        _event("user-message", kind="message", source="user"),
        _event("tool-output", kind="observation", source="environment"),
        _event(None),
        _event(""),
    ]

    _, info = asyncio.run(env.evaluate())
    assert info["num_agent_turns"] == 2
    assert info["length_reward"] == pytest.approx(8 / 30)


def test_reward_respects_configured_threshold_and_max_turns(env_module):
    env = env_module.SWERebenchEnv(
        task=SimpleNamespace(max_steps=60, misc={"repo_path": "/testbed"}),
        agent=object(),
        workspace=object(),
        length_penalty_threshold=30,
        enable_length_penalty=True,
    )
    env._state = SimpleNamespace(
        finished=True,
        conversation_state=SimpleNamespace(events=[_event(str(i)) for i in range(15)]),
    )
    reward, info = asyncio.run(env.evaluate())
    assert reward == 1.5
    assert info["length_reward"] == 0.5
    assert info["length_penalty_threshold"] == 30


@pytest.mark.parametrize("enable_length_penalty", [True, False])
def test_unfinished_episode_returns_no_reward(env_module, enable_length_penalty):
    env = _env(env_module, turns=25, finished=False, enable_length_penalty=enable_length_penalty)
    assert asyncio.run(env.evaluate()) == (0.0, {})
    env_module.compute_test_execution_reward.assert_not_awaited()


@pytest.mark.parametrize("threshold", [-1, 40, 41, 10.5, True, None])
def test_invalid_threshold_is_rejected(env_module, threshold):
    with pytest.raises(ValueError, match="length_penalty_threshold"):
        env_module.SWERebenchEnv(
            task=SimpleNamespace(max_steps=40),
            agent=object(),
            workspace=object(),
            length_penalty_threshold=threshold,
            enable_length_penalty=True,
        )


@pytest.mark.parametrize("finished,binary_reward", [(True, 1.0), (True, 0.0), (False, 0.0)])
def test_disabled_penalty_preserves_binary_reward(env_module, finished, binary_reward):
    env = _env(env_module, turns=40, finished=finished, enable_length_penalty=False)
    env_module.compute_test_execution_reward.return_value = (binary_reward, {"binary_reward": binary_reward})

    # Disabled runs must not read SDK events or compute the length reward.
    del env._state.conversation_state

    reward, info = asyncio.run(env.evaluate())
    assert reward == binary_reward
    assert "length_reward" not in info


def test_penalty_is_disabled_by_default_and_ignores_unused_threshold(env_module):
    env = env_module.SWERebenchEnv(
        task=SimpleNamespace(max_steps=5, misc={"repo_path": "/testbed"}),
        agent=object(),
        workspace=object(),
        length_penalty_threshold=10,
    )
    env._state = SimpleNamespace(finished=True)
    reward, _ = asyncio.run(env.evaluate())
    assert reward == 1.0
