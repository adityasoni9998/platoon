"""Exercise the legacy subprocess lifecycle against CodeScout SAIL's adapter."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from platoon.codescout_sail import tinker_workflow

BASE = Path(__file__).resolve().parents[1] / "plugins/codescout_sail/platoon/codescout_sail"


@pytest.fixture
def adapter():
    return tinker_workflow


@pytest.fixture
def child_rollout(tmp_path, monkeypatch):
    """Importable test rollout executed by the real single-use worker."""
    import os

    (tmp_path / "legacy_test_rollout.py").write_text('''
import asyncio
import os
import time
from pathlib import Path

async def run_rollout(task, config):
    root = Path(config.extra["markers"])
    (root / (task.id + ".started")).write_text(str(os.getpid()))
    try:
        if task.misc.get("fail"):
            raise RuntimeError("child rollout failed")
        if task.misc.get("hang"):
            time.sleep(60)  # Blocks even the SIGTERM callback on this event loop.
        if task.misc.get("wait"):
            await asyncio.sleep(60)
        if task.misc.get("exit_hang"):
            import threading
            threading.Thread(target=time.sleep, args=(60,), daemon=False).start()
        while not (root / "release").exists():
            time.sleep(0.01)  # Parent and other children must remain responsive.
        task.misc["child_mutation"] = True
        return {"trajectories": [task.id], "pid": os.getpid(),
                "session": config.extra["llm_extra_headers"]["X-Platoon-Tinker-Session"],
                "endpoint": config.model_endpoint,
                "max_tokens": config.inference_params.max_completion_tokens,
                "max_input_tokens": config.extra["max_input_tokens"]}
    finally:
        (root / (task.id + ".cleaning")).touch()
        await asyncio.sleep(0.1)
        (root / (task.id + ".cleaned")).touch()

async def context_rollout(task, config):
    import logging
    from types import SimpleNamespace
    from platoon.episode import context
    from platoon.episode.loop import run_episode
    from platoon.episode.trajectory import TrajectoryCollection, TrajectoryStep
    from platoon.visualization.event_sinks import JsonlFileSink

    assert context.current_trajectory_collection.get(None) is None
    collection = TrajectoryCollection()
    context.current_trajectory_collection.set(collection)
    collection.register_event_handlers(JsonlFileSink(
        Path(config.extra["markers"]) / (collection.id + ".jsonl"),
        collection_id=collection.id, process_id=os.getpid(),
    ))
    class Env:
        async def reset(self):
            assert context.current_env.get() is self
            assert context.current_agent.get() is agent
            assert context.episode_step_timeout.get() == 17
            trajectory = context.current_trajectory.get()
            assert trajectory.parent_info is None
            collection.set_trajectory_task(trajectory.id, task)
            collection.add_trajectory_step(trajectory.id, TrajectoryStep(
                misc={"action_misc": {"completion_id": "response-1"}}
            ))
            trajectory.reward = 1.0
            return SimpleNamespace(finished=True)
        async def close(self):
            pass
    class Agent:
        async def close(self):
            pass
    env, agent = Env(), Agent()
    env.task = task
    await asyncio.create_task(run_episode(agent, env, timeout=17))
    logging.info("context rollout complete")
    print("child stdout intact", flush=True)
    return collection.to_dict()
''')
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([
        str(tmp_path), str(BASE.parents[3]), os.environ.get("PYTHONPATH", "")
    ]))
    return SimpleNamespace(__module__="legacy_test_rollout", __name__="run_rollout")


def child_config(tmp_path, timeout=None):
    from platoon.config_defs import InferenceParams, RolloutConfig

    config = RolloutConfig(
        model_endpoint="https://shared-proxy.example/v1",
        inference_params=InferenceParams(max_completion_tokens=2048),
        extra={"markers": str(tmp_path)},
    )
    if timeout is not None:
        config.extra["subprocess_timeout"] = timeout
    return config


async def wait_for_file(path):
    async with asyncio.timeout(10):
        while not path.exists():
            await asyncio.sleep(0.01)


def test_real_children_isolate_tasks_and_leave_parent_responsive(adapter, child_rollout, tmp_path):
    import os

    from platoon.envs.base import Task

    async def run():
        proxy = Mock(context_window_length=42048)
        proxy.pop_interactions.side_effect = lambda session: {"captured": session}
        wrapped = adapter.TinkerProxyRollout(child_rollout, proxy)
        task = Task(id="one")
        config = child_config(tmp_path)
        children = [asyncio.create_task(wrapped(task, config)) for _ in range(2)]
        try:
            await wait_for_file(tmp_path / "one.started")
            assert await asyncio.wait_for(asyncio.to_thread(lambda: "responsive"), 1) == "responsive"
        finally:
            (tmp_path / "release").touch()
        results = await asyncio.gather(*children)
        assert len({r["pid"] for r in results}) == 2
        assert all(r["pid"] != os.getpid() for r in results)
        assert len({r["session"] for r in results}) == 2
        for result in results:
            assert result["_tinker_interactions"] == {"captured": result["session"]}
            assert result["endpoint"] == config.model_endpoint
            assert result["max_tokens"] == 2048
            assert result["max_input_tokens"] == 42048
        assert task.misc == {}
        assert "llm_extra_headers" not in config.extra
        assert not config.return_dict
        assert proxy.discard_session.call_count == 2

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["cancel", "hang", "timeout", "exit_hang", "fail", "launch_cancel"])
def test_real_child_failure_and_cancellation_reap_before_session_cleanup(
    adapter, child_rollout, tmp_path, monkeypatch, mode
):
    from platoon.envs.base import Task

    monkeypatch.setattr(adapter, "CHILD_CLEANUP_TIMEOUT", 0.3)

    async def run():
        processes = []
        paths = []
        launch = asyncio.create_subprocess_exec
        launched = asyncio.Event()

        async def record_launch(*args, **kwargs):
            process = await launch(*args, **kwargs)
            processes.append(process)
            paths.append(Path(args[-2]).parent)
            launched.set()
            if mode == "launch_cancel":
                await asyncio.sleep(0.1)  # Cancellation before the parent gets its handle.
            return process

        monkeypatch.setattr(adapter.asyncio, "create_subprocess_exec", record_launch)
        proxy = Mock(context_window_length=42048)

        def discard(session):
            assert processes[0].returncode is not None
            assert not paths[0].exists()

        proxy.discard_session.side_effect = discard
        wrapped = adapter.TinkerProxyRollout(child_rollout, proxy)
        task = Task(id="one", misc={
            "fail": mode == "fail", "hang": mode in {"hang", "timeout"},
            "wait": mode in {"cancel", "launch_cancel"},
            "exit_hang": mode == "exit_hang",
        })
        if mode == "exit_hang":
            (tmp_path / "release").touch()
        pending = asyncio.create_task(wrapped(task, child_config(tmp_path, 1 if mode in {"timeout", "exit_hang"} else None)))
        if mode == "launch_cancel":
            await launched.wait()
            pending.cancel()
        elif mode in {"cancel", "hang"}:
            await wait_for_file(tmp_path / "one.started")
            pending.cancel()
            if mode == "cancel":
                await wait_for_file(tmp_path / "one.cleaning")
            else:
                await asyncio.sleep(0.02)
            pending.cancel()  # Must not interrupt child cleanup/reaping.
        error = RuntimeError if mode == "fail" else TimeoutError if mode in {"timeout", "exit_hang"} else asyncio.CancelledError
        with pytest.raises(error):
            await asyncio.wait_for(pending, 10)
        assert processes[0].returncode is not None
        if mode in {"hang", "timeout"}:
            assert processes[0].returncode == -9
        if mode in {"cancel", "exit_hang"}:
            assert (tmp_path / "one.cleaned").exists()
        proxy.discard_session.assert_called_once()
        proxy.pop_interactions.assert_not_called()

    asyncio.run(run())


def test_child_launch_failure_cleans_session_and_files(adapter, child_rollout, tmp_path, monkeypatch):
    from platoon.envs.base import Task

    paths = []

    async def fail_launch(*args, **kwargs):
        paths.append(Path(args[-2]).parent)
        raise OSError("cannot launch child")

    monkeypatch.setattr(adapter.asyncio, "create_subprocess_exec", fail_launch)
    proxy = Mock(context_window_length=42048)
    wrapped = adapter.TinkerProxyRollout(child_rollout, proxy)
    with pytest.raises(OSError, match="cannot launch child"):
        asyncio.run(wrapped(Task(id="one"), child_config(tmp_path)))
    assert not paths[0].exists()
    proxy.discard_session.assert_called_once()


def test_child_episode_context_event_logs_and_serialized_result(adapter, child_rollout, tmp_path, capfd):
    import json

    from platoon.envs.base import Task
    from platoon.episode.context import current_trajectory_collection
    from platoon.episode.trajectory import TrajectoryCollection

    child_rollout.__name__ = "context_rollout"
    parent_collection = TrajectoryCollection()
    token = current_trajectory_collection.set(parent_collection)
    try:
        result = asyncio.run(adapter.TinkerProxyRollout(child_rollout, Mock(context_window_length=42048))(
            Task(id="one", max_steps=10), child_config(tmp_path)
        ))
        assert current_trajectory_collection.get() is parent_collection
        assert not parent_collection.trajectories
    finally:
        current_trajectory_collection.reset(token)
    trajectory = next(iter(result["trajectories"].values()))
    assert trajectory["reward"] == 1.0
    assert trajectory["error_message"] is None
    assert trajectory["steps"][0]["misc"]["action_misc"]["completion_id"] == "response-1"
    records = [json.loads(line) for line in (tmp_path / (result["id"] + ".jsonl")).read_text().splitlines()]
    assert records[-1]["type"] == "trajectory_finished"
    assert all(r["collection_id"] == result["id"] for r in records)
    output = capfd.readouterr()
    assert "child stdout intact" in output.out
    assert "context rollout complete" in output.err


def test_child_llm_receives_parent_session_and_context_limit(monkeypatch):
    from platoon.codescout_sail import rollout
    from platoon.config_defs import InferenceParams, RolloutConfig

    monkeypatch.setattr(rollout, "_get_active_tinker_proxy", lambda: None)
    llm = Mock()
    monkeypatch.setattr(rollout, "LLM", llm)
    config = RolloutConfig(
        model_name="openai/platoon-tinker",
        model_endpoint="https://relay.example/v1",
        inference_params=InferenceParams(max_completion_tokens=2048),
        extra={
            "llm_extra_headers": {"X-Platoon-Tinker-Session": "child-session"},
            "max_input_tokens": 42048,
        },
    )
    rollout.prepare_llm(config)
    assert llm.call_args.kwargs["extra_headers"] == {"X-Platoon-Tinker-Session": "child-session"}
    assert llm.call_args.kwargs["max_input_tokens"] == 42048
    assert llm.call_args.kwargs["base_url"] == "https://relay.example/v1"
