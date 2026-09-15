"""Transport isolation and cleanup checks without hosted training services."""

import ast
import asyncio
import importlib.util
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
import yaml

BASE = Path(__file__).resolve().parents[1] / "plugins/issue-resolution-legacy/platoon/issue_resolution"


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, BASE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tunnel_module(monkeypatch):
    ssh = load_module("legacy_ssh", "ssh_tunnel.py")
    monkeypatch.setitem(sys.modules, "platoon.issue_resolution.ssh_tunnel", ssh)
    selector = load_module("legacy_tunnel_selector", "inference_tunnel.py")
    monkeypatch.setitem(sys.modules, "platoon.issue_resolution.inference_tunnel", selector)
    return selector


@pytest.fixture
def adapter(monkeypatch, tunnel_module):
    proxy_module = types.ModuleType("platoon.train.tinker.fastapi_litellm_proxy")
    proxy_module.SESSION_HEADER = "X-Platoon-Tinker-Session"
    proxy_module.FastAPILiteLLMTinkerHTTPProxyServer = Mock()
    monkeypatch.setitem(sys.modules, proxy_module.__name__, proxy_module)
    return load_module("legacy_tinker_adapter", "tinker_workflow.py")


@pytest.mark.parametrize("failure", [None, RuntimeError, asyncio.CancelledError])
def test_tinker_rollout_sessions_are_isolated_and_cleaned(adapter, failure, monkeypatch):
    proxy = Mock()
    proxy.pop_interactions.side_effect = lambda session: {"response": session}
    config = SimpleNamespace(extra={"llm_extra_headers": {"Existing": "value"}}, return_dict=False)
    seen = {}

    async def rollout(task, local_config):
        assert local_config.return_dict
        headers = local_config.extra["llm_extra_headers"]
        assert headers["Existing"] == "value"
        seen[task] = headers["X-Platoon-Tinker-Session"]
        await asyncio.sleep(0)
        if failure:
            raise failure()
        return {"trajectories": [task]}

    async def run():
        wrapped = adapter.TinkerProxyRollout(rollout, proxy)
        monkeypatch.setattr(wrapped, "_run_subprocess", AsyncMock(side_effect=rollout))
        return await asyncio.gather(wrapped("a", config), wrapped("b", config), return_exceptions=True)

    results = asyncio.run(run())
    assert len(set(seen.values())) == 2
    assert config.extra == {"llm_extra_headers": {"Existing": "value"}}
    assert not config.return_dict
    assert {call.args[0] for call in proxy.discard_session.call_args_list} == set(seen.values())
    if failure:
        assert all(isinstance(result, failure) for result in results)
        proxy.pop_interactions.assert_not_called()
    else:
        assert [result["_tinker_interactions"]["response"] for result in results] == [seen["a"], seen["b"]]


@pytest.mark.parametrize("failure_stage", [None, "start", "tunnel", "train"])
def test_tinker_proxy_routing_and_cleanup(adapter, monkeypatch, failure_stage):
    model = SimpleNamespace(
        model_name="tinker/model", base_url=None, api_key=None, llm=SimpleNamespace(context_window_length=32768)
    )
    proxy = Mock(base_url="http://127.0.0.1:9000/v1", model_name="openai/platoon-tinker", api_key="key")
    if failure_stage == "start":
        proxy.start.side_effect = RuntimeError("start")
    factory = Mock(return_value=proxy)
    monkeypatch.setattr(adapter, "FastAPILiteLLMTinkerHTTPProxyServer", factory)
    tunnel = Mock()
    tunnel.__enter__ = Mock(return_value={"http://127.0.0.1:9000": "http://ogma:1234"})
    tunnel.__exit__ = Mock(return_value=False)
    if failure_stage == "tunnel":
        tunnel.__enter__.side_effect = RuntimeError("tunnel")
    tunnel_factory = Mock(return_value=tunnel)
    monkeypatch.setattr(adapter, "inference_tunnels", tunnel_factory)

    def run():
        with adapter.routed_tinker_proxy(model) as active:
            assert active is proxy
            assert model.base_url == "http://ogma:1234/v1"
            assert model.model_name == "openai/platoon-tinker"
            assert model.api_key == "key"
            tunnel_factory.assert_called_once_with(["http://127.0.0.1:9000"], config=None)
            if failure_stage == "train":
                raise RuntimeError("train")

    if failure_stage:
        with pytest.raises(RuntimeError, match=failure_stage):
            run()
    else:
        run()
    assert (model.model_name, model.base_url, model.api_key) == ("tinker/model", None, None)
    proxy.stop.assert_called_once()
    if failure_stage in (None, "train"):
        tunnel.__exit__.assert_called_once()


def test_shared_llm_preserves_areal_options_and_accepts_transport_headers():
    # Execute just the real LLM builder, avoiding Modal/SDK startup dependencies.
    tree = ast.parse((BASE / "rollout.py").read_text())
    builder = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "prepare_llm")
    llm = Mock()
    namespace = {"LLM": llm, "RolloutConfig": object}
    exec(compile(ast.Module(body=[builder], type_ignores=[]), "rollout.py", "exec"), namespace)
    config = SimpleNamespace(
        model_name="model",
        model_endpoint="http://ogma:1234",
        model_api_key="session-key",
        extra={"max_input_tokens": 30720},
        inference_params=SimpleNamespace(temperature=1, max_completion_tokens=2048),
    )
    namespace["prepare_llm"](config)
    areal_kwargs = llm.call_args.kwargs
    assert "extra_headers" not in areal_kwargs
    assert areal_kwargs["api_key"] == "session-key"
    assert areal_kwargs["base_url"] == config.model_endpoint
    assert areal_kwargs["max_input_tokens"] == 30720
    config.extra["llm_extra_headers"] = {"X-Platoon-Tinker-Session": "rollout-1"}
    namespace["prepare_llm"](config)
    tinker_kwargs = dict(llm.call_args.kwargs)
    assert tinker_kwargs.pop("extra_headers") == config.extra["llm_extra_headers"]
    assert tinker_kwargs == areal_kwargs


def test_tunnel_preserves_areal_endpoint_paths(tunnel_module, monkeypatch):
    forward = Mock(side_effect=[2345, 2346])
    connection = MagicMock()
    connection.__enter__.return_value = forward
    factory = Mock(return_value=connection)
    monkeypatch.setattr(tunnel_module, "ssh_connection", factory)
    health = Mock()
    monkeypatch.setattr(tunnel_module, "wait_for_health", health)
    urls = ["http://worker:1234/", "http://worker:1234", "http://worker:1235/v1/"]
    with tunnel_module.inference_tunnels(urls) as endpoints:
        assert endpoints == {
            "http://worker:1234": "http://ogma.lti.cs.cmu.edu:2345",
            "http://worker:1235/v1": "http://ogma.lti.cs.cmu.edu:2346/v1",
        }
        assert forward.call_count == health.call_count == 2
    factory.assert_called_once_with("adityabs@ogma.lti.cs.cmu.edu")
    connection.__exit__.assert_called_once()


@pytest.mark.parametrize("urls", [[], ["https://worker:1234"], ["http://worker"]])
def test_tunnel_rejects_invalid_origins_before_startup(tunnel_module, monkeypatch, urls):
    connection = Mock()
    monkeypatch.setattr(tunnel_module, "ssh_connection", connection)
    with pytest.raises(ValueError):
        with tunnel_module.inference_tunnels(urls):
            pytest.fail("Invalid origins must not create a tunnel")
    connection.assert_not_called()


def test_recipe_preserves_legacy_rollout_policy():
    areal = yaml.safe_load((BASE / "train_issue_resolution_fft_areal_fsdp.yaml").read_text())
    tinker = yaml.safe_load((BASE / "train_issue_resolution_fft_tinker.yaml").read_text())
    aw = areal["workflow_config"]
    tw = tinker["train"]["workflow_config"]
    for key in ("group_size", "filter_zero_variance_groups"):
        assert tw[key] == aw[key]
    for key in ("max_steps", "timeout", "step_timeout", "train", "extra", "inference_params"):
        assert tw["rollout_config"][key] == aw["rollout_config"][key]
    assert tinker["train"]["num_epochs"] == areal["total_train_epochs"]
    assert tinker["train"]["batch_size"] == areal["train_dataset"]["batch_size"]
    assert tinker["train"]["num_concurrent_rollout_workflow_workers"] == areal["rollout"]["max_concurrent_rollouts"]


def test_areal_workflow_keeps_internal_session_and_routes_only_rollout(monkeypatch):
    internal = "http://worker:9000"
    public = "http://ogma:1234"

    class BaseWorkflow:
        def __init__(self, **kwargs):
            self.proxy_base_url = internal

        def to_workflow_kwargs(self):
            return {"proxy_base_url": self.proxy_base_url}

        def _require_proxy_base_url(self):
            return self.proxy_base_url

        def _build_rollout_config(self, engine, session):
            return SimpleNamespace(
                rollout_config=SimpleNamespace(model_endpoint=internal, model_api_key=session.api_key)
            )

    for name, attrs in {
        "areal.api": {"InferenceEngine": object},
        "platoon.train.areal.proxy": {"ArealProxySession": object},
        "platoon.train.areal.workflows": {"GroupRolloutWorkflow": BaseWorkflow},
    }.items():
        stub = types.ModuleType(name)
        stub.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, stub)
    module = load_module("legacy_areal_workflow", "areal_workflow.py")
    workflow = module.RoutedGroupRolloutWorkflow(proxy_endpoint_map={internal: public})
    result = workflow._build_rollout_config(object(), SimpleNamespace(api_key="session-key"))
    assert result.rollout_config.model_endpoint == public
    assert result.rollout_config.model_api_key == "session-key"
    assert workflow.to_workflow_kwargs() == {"proxy_base_url": internal, "proxy_endpoint_map": {internal: public}}
    workflow.proxy_endpoint_map.clear()
    with pytest.raises(RuntimeError, match="No tunnel registered"):
        workflow._build_rollout_config(object(), SimpleNamespace(api_key="session-key"))


def test_tunnel_yaml_defaults_and_both_config_loaders(tunnel_module):
    from omegaconf import OmegaConf

    selector = tunnel_module
    config = selector.InferenceTunnelConfig()
    assert config.backend == "ogma"
    assert (config.modal.cpu, config.modal.memory, config.modal.timeout) == (0.125, 128, 86400)
    # AReaL uses OmegaConf structured dataclasses; Tinker uses Platoon's loader.
    structured = OmegaConf.structured(selector.InferenceTunnelConfig)
    parsed = OmegaConf.to_object(OmegaConf.merge(structured, {"backend": "modal"}))
    assert parsed.backend == "modal"
    assert parsed.modal == config.modal
    utils_path = BASE.parents[3] / "platoon/utils/config.py"
    spec = importlib.util.spec_from_file_location("legacy_config_utils", utils_path)
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    parsed = utils._dataclass_from_dict(selector.InferenceTunnelConfig, {"backend": "modal"})
    assert parsed.modal == config.modal
    for filename, backend in (
        ("train_issue_resolution_fft_tinker.yaml", "modal"),
        ("train_issue_resolution_fft_areal_fsdp.yaml", "ogma"),
    ):
        data = yaml.safe_load((BASE / filename).read_text())["inference_tunnel"]
        expected = selector.InferenceTunnelConfig(backend=backend)
        assert utils._dataclass_from_dict(selector.InferenceTunnelConfig, data) == expected
    with pytest.raises(ValueError, match="Unknown inference_tunnel.backend"):
        with selector.inference_tunnels(["http://localhost:1234"], selector.InferenceTunnelConfig(backend="invalid")):
            pytest.fail("Invalid backend must not create a tunnel")


@pytest.fixture
def modal_router_module(monkeypatch, tunnel_module):
    module = load_module("legacy_modal_router", "modal_tunnel.py")
    monkeypatch.setitem(sys.modules, "platoon.issue_resolution.modal_tunnel", module)
    return module


@pytest.mark.parametrize(
    "stage", [None, "create", "allocation", "readiness", "ssh", "forward", "health", "body", "interrupt"]
)
def test_modal_router_routing_and_resource_lifetime(modal_router_module, tunnel_module, monkeypatch, stage):
    module = modal_router_module
    sdk = types.ModuleType("modal")
    for name in ("Image", "App", "Secret", "Probe", "Sandbox"):
        setattr(sdk, name, Mock())
    sandbox = sdk.Sandbox.create.return_value
    sandbox.tunnels.return_value = {
        2222: SimpleNamespace(tcp_socket=("relay.modal.host", 2345)),
        10000: SimpleNamespace(url="https://first.modal.host"),
        10001: SimpleNamespace(url="https://second.modal.host"),
    }
    sandbox.poll.return_value = None
    monkeypatch.setitem(sys.modules, "modal", sdk)
    connection = MagicMock()
    connection.__enter__.return_value.recv.side_effect = [b"", b"SSH-2.0-OpenSSH\r\n"]
    monkeypatch.setattr(module.socket, "create_connection", Mock(return_value=connection))
    monkeypatch.setattr(module.time, "sleep", Mock())
    key_paths = []

    def keygen(args, **kwargs):
        path = Path(args[-1])
        key_paths.append(path)
        path.write_text("private-key")
        path.with_suffix(".pub").write_text("ssh-ed25519 public-key\n")

    monkeypatch.setattr(module.subprocess, "run", Mock(side_effect=keygen))
    if stage == "create":
        sdk.Sandbox.create.side_effect = RuntimeError(stage)
    if stage == "allocation":
        sandbox.tunnels.side_effect = RuntimeError(stage)
    if stage == "readiness":
        monkeypatch.setattr(module, "wait_for_ssh", Mock(side_effect=RuntimeError(stage)))
    forward = Mock()
    if stage == "forward":
        forward.side_effect = RuntimeError(stage)
    calls = []

    @contextmanager
    def ssh(target, options):
        assert target == "root@relay.modal.host"
        assert "StrictHostKeyChecking=yes" in options
        assert "HostKeyAlias=platoon-modal-router" in options
        assert key_paths[0].exists()
        if stage == "ssh":
            raise RuntimeError(stage)
        try:
            yield forward
        finally:
            sandbox.terminate.assert_not_called()
            calls.append("ssh-closed")

    monkeypatch.setattr(module, "ssh_connection", ssh)
    health = Mock(side_effect=RuntimeError(stage) if stage == "health" else None)
    monkeypatch.setattr(tunnel_module, "wait_for_health", health)

    def run():
        with tunnel_module.inference_tunnels(
            ["http://127.0.0.1:9000", "http://worker:9001/v1"],
            tunnel_module.InferenceTunnelConfig(backend="modal"),
        ) as endpoints:
            assert endpoints == {
                "http://127.0.0.1:9000": "https://first.modal.host",
                "http://worker:9001/v1": "https://second.modal.host/v1",
            }
            assert [call.args for call in forward.call_args_list] == [
                ("http://127.0.0.1:9000", 10000),
                ("http://worker:9001/v1", 10001),
            ]
            assert module.socket.create_connection.call_count == 2
            if stage == "body":
                raise RuntimeError(stage)
            if stage == "interrupt":
                raise KeyboardInterrupt

    if stage:
        with pytest.raises(KeyboardInterrupt if stage == "interrupt" else RuntimeError):
            run()
    else:
        run()
    kwargs = sdk.Sandbox.create.call_args.kwargs
    assert (kwargs["cpu"], kwargs["memory"], kwargs["timeout"]) == (0.125, 128, 86400)
    assert kwargs["encrypted_ports"] == [10000, 10001]
    assert kwargs["unencrypted_ports"] == [2222]
    assert key_paths and all(not path.parent.exists() for path in key_paths)
    assert sandbox.terminate.call_count == (0 if stage == "create" else 1)
    assert calls == ([] if stage in {"create", "allocation", "readiness", "ssh"} else ["ssh-closed"])


@pytest.mark.parametrize("stage", [None, "startup", "forward", "health", "interrupt", "kill"])
def test_ssh_connection_cleans_up_on_every_exit(tunnel_module, monkeypatch, stage):
    import subprocess

    module = sys.modules["platoon.issue_resolution.ssh_tunnel"]
    process = Mock()
    process.poll.return_value = 1 if stage == "startup" else None
    if stage == "kill":
        process.wait.side_effect = [subprocess.TimeoutExpired("ssh", 10), None]
    paths = []

    def popen(args, **kwargs):
        paths.append(Path(args[args.index("-S") + 1]))
        assert paths[-1].parent.exists()
        if stage == "startup":
            kwargs["stderr"].write("connection failed")
        return process

    def command(args, **kwargs):
        if "forward" in args:
            if stage == "forward":
                raise subprocess.CalledProcessError(1, args)
            assert args[args.index("-R") + 1] == "0.0.0.0:0:[::1]:9000"
        return SimpleNamespace(stdout="12345\n")

    monkeypatch.setattr(module.subprocess, "Popen", popen)
    monkeypatch.setattr(module.subprocess, "run", command)
    health = Mock(side_effect=RuntimeError("health") if stage == "health" else None)
    monkeypatch.setattr(tunnel_module, "wait_for_health", health)

    def run():
        with tunnel_module.inference_tunnels(["http://[::1]:9000"]) as endpoints:
            assert list(endpoints.values()) == ["http://ogma.lti.cs.cmu.edu:12345"]
            if stage == "interrupt":
                raise KeyboardInterrupt

    errors = {
        "startup": RuntimeError,
        "forward": subprocess.CalledProcessError,
        "health": RuntimeError,
        "interrupt": KeyboardInterrupt,
    }
    if stage in errors:
        with pytest.raises(errors[stage]):
            run()
    else:
        run()
    assert paths and not paths[0].parent.exists()
    assert process.terminate.call_count == (0 if stage == "startup" else 1)
    assert process.kill.call_count == (1 if stage == "kill" else 0)


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
                "max_tokens": config.inference_params.max_completion_tokens}
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
        proxy = Mock()
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
        proxy = Mock()

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
    proxy = Mock()
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
        result = asyncio.run(adapter.TinkerProxyRollout(child_rollout, Mock())(
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


def test_main_reaps_rollout_before_stopping_proxy(adapter, child_rollout, tmp_path):
    from platoon.envs.base import Task

    module_path = BASE / "train_tinker.py"
    tree = ast.parse(module_path.read_text())
    main_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    proxy = Mock()

    @contextmanager
    def router(*args):
        try:
            yield proxy
        finally:
            assert (tmp_path / "one.cleaned").exists()
            proxy.discard_session.assert_called_once()

    async def train(trainer, wrapped):
        # Leave an active rollout for asyncio.run's shutdown to cancel and reap.
        asyncio.create_task(wrapped(Task(id="one", misc={"wait": True}), child_config(tmp_path)))
        await wait_for_file(tmp_path / "one.started")
        raise asyncio.CancelledError

    namespace = {
        "__file__": str(module_path), "Path": Path, "asyncio": asyncio,
        "IssueResolutionTinkerConfig": object,
        "load_config": lambda **kwargs: (SimpleNamespace(inference_tunnel=None), {}),
        "load_data": lambda: ({"one": None}, {}),
        "Dataset": SimpleNamespace(from_list=lambda rows: rows),
        "PlatoonTinkerRLTrainer": lambda **kwargs: SimpleNamespace(model_info=None),
        "routed_tinker_proxy": router, "TinkerProxyRollout": adapter.TinkerProxyRollout,
        "run_rollout": child_rollout, "train": train,
    }
    exec(compile(ast.Module(body=[main_node], type_ignores=[]), str(module_path), "exec"), namespace)
    with pytest.raises(asyncio.CancelledError):
        namespace["main"]([])
