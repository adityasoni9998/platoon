"""Real forked workers: isolation, failure containment, cancellation and fallback."""

import asyncio
import importlib
import os
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from platoon.config_defs import RolloutConfig
from platoon.envs.base import Task
from platoon.utils.forkserver import ForkServerError
from platoon.utils.rollout_forkserver import RolloutForkServer


@pytest.mark.parametrize(
    "plugin,package",
    [
        ("codescout_sail", "codescout_sail"),
        ("issue-resolution-rebench", "issue_resolution"),
    ],
)
def test_plugin_forked_workers_are_isolated_and_reaped(tmp_path, monkeypatch, plugin, package):
    import platoon

    plugin_root = Path(__file__).resolve().parents[1] / "plugins" / plugin
    monkeypatch.setattr(platoon, "__path__", [*platoon.__path__, str(plugin_root / "platoon")])
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(
            [
                str(tmp_path),
                str(plugin_root),
                os.environ.get("PYTHONPATH", ""),
            ]
        ),
    )
    (tmp_path / "fork_rollout_stub.py").write_text("""
import asyncio, os, signal, time
from pathlib import Path

CALLS = 0

async def run_rollout(task, config):
    global CALLS
    CALLS += 1
    root = Path(config.output_dir)
    (root / (task.id + ".pid")).write_text(str(os.getpid()))
    try:
        if task.misc.get("kill"):
            os.kill(os.getpid(), signal.SIGKILL)
        if task.misc.get("wait"):
            await asyncio.sleep(60)
        await asyncio.sleep(.05)
        return {"trajectories": [task.id], "calls": CALLS, "pid": os.getpid(),
                "session": config.extra["llm_extra_headers"]["X-Platoon-Tinker-Session"]}
    finally:
        (root / (task.id + ".cleaned")).touch()
""")
    sys.modules.pop("fork_rollout_stub", None)
    stub = importlib.import_module("fork_rollout_stub")
    adapter = importlib.import_module(f"platoon.{package}.tinker_workflow")
    monkeypatch.setattr(adapter, "CHILD_CLEANUP_TIMEOUT", 1)

    async def run():
        proxy = Mock(context_window_length=42048)
        proxy.pop_interactions.return_value = {}
        wrapped = adapter.TinkerProxyRollout(stub.run_rollout, proxy)
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        config = RolloutConfig(output_dir=str(tmp_path), extra={
            "worker_start_method": "forkserver",
            "workspace": {"type": "apptainer", "apptainer": {"scratch_dir": str(scratch)}},
        })
        launch_worker = wrapped._launch_worker

        async def check_ipc_location(request, response, worker_config):
            assert request.parent.parent == scratch
            assert response.parent == request.parent
            return await launch_worker(request, response, worker_config)

        monkeypatch.setattr(wrapped, "_launch_worker", check_ipc_location)
        server_pid = None
        try:
            results = await asyncio.gather(
                wrapped(Task(id="one"), config),
                wrapped(Task(id="killed", misc={"kill": True}), config),
                wrapped(Task(id="two"), config),
                return_exceptions=True,
            )
            assert isinstance(results[1], RuntimeError)
            assert "-9" in str(results[1])
            assert results[0]["calls"] == results[2]["calls"] == 1
            assert results[0]["pid"] != results[2]["pid"]
            assert results[0]["session"] != results[2]["session"]
            assert stub.CALLS == 0
            server_pid = wrapped._forkserver._server.pid
            # Reuse the same preloaded server after an unrelated child was killed.
            result = await wrapped(Task(id="three"), config)
            assert result["calls"] == 1
            assert wrapped._forkserver._server.pid == server_pid
            pending = asyncio.create_task(wrapped(Task(id="cancel", misc={"wait": True}), config))
            for _ in range(500):
                if (tmp_path / "cancel.pid").exists():
                    break
                await asyncio.sleep(0.01)
            assert (tmp_path / "cancel.pid").exists()
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(pending, 5)
            assert (tmp_path / "cancel.cleaned").exists()
            assert proxy.discard_session.call_count == 5
            assert proxy.pop_interactions.call_count == 3
            for path in tmp_path.glob("*.pid"):
                assert not Path("/proc", path.read_text()).exists()
        finally:
            await wrapped.aclose()
        assert not Path("/proc", str(server_pid)).exists()
        assert not list(scratch.iterdir())

    asyncio.run(run())


def test_unavailable_forkserver_falls_back_once(tmp_path, monkeypatch, caplog):
    from unittest.mock import AsyncMock

    from platoon.utils.rollout_forkserver import ForkServer

    start = AsyncMock(side_effect=ForkServerError("unsafe preload", permanent=True))
    monkeypatch.setattr(ForkServer, "start", start)

    async def run():
        server = RolloutForkServer("unused_worker", "unused_rollout")
        args = ["unused_rollout", "run", str(tmp_path / "request.pkl"), str(tmp_path / "response.pkl")]
        assert await server.spawn(args, str(tmp_path)) is None
        assert await server.spawn(args, str(tmp_path)) is None
        assert start.await_count == 1
        await server.aclose()

    asyncio.run(run())
    assert any(
        record.levelname == "ERROR" and "Forkserver did not start" in record.message
        for record in caplog.records
    )


@pytest.mark.parametrize("alive", [True, False])
def test_spawn_failure_reuses_only_a_live_forkserver(tmp_path, monkeypatch, alive):
    from unittest.mock import AsyncMock

    from platoon.utils.rollout_forkserver import ForkServer

    worker = Mock()
    upstream = Mock(alive=alive, env={}, cwd=str(tmp_path), pid=123, seconds=0.1)
    upstream.spawn = AsyncMock(side_effect=[ForkServerError("worker launch failed"), worker])
    upstream.kill = AsyncMock()
    start = AsyncMock(return_value=upstream)
    monkeypatch.setattr(ForkServer, "start", start)

    async def run():
        server = RolloutForkServer("unused_worker", "unused_rollout")
        args = ["unused_rollout", "run", str(tmp_path / "request.pkl"), str(tmp_path / "response.pkl")]
        try:
            assert await server.spawn(args, str(tmp_path)) is None
            result = await server.spawn(args, str(tmp_path))
            if alive:
                assert result.worker is worker
                assert upstream.spawn.await_count == 2
            else:
                assert result is None
                assert upstream.spawn.await_count == 1
            assert start.await_count == 1
        finally:
            await server.aclose()

    asyncio.run(run())
