"""CodeScout proxy routing and cleanup without a hosted training service."""

import asyncio
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient
from platoon.codescout_sail import train_tinker

from platoon.inference import inference_tunnel, modal_tunnel
from platoon.train.tinker import fastapi_litellm_proxy
from platoon.train.tinker.proxy import proxy_interactions
from platoon.utils.config import load_config


def mock_training(monkeypatch, model, train=None):
    trainer = SimpleNamespace(model_info=model, run_log_path="/tmp/test", train=AsyncMock(side_effect=train))

    class TrainerContext:
        def __getattr__(self, name):
            return getattr(trainer, name)

        async def __aenter__(self):
            return trainer

        async def __aexit__(self, *args):
            return False

    config = SimpleNamespace(
        inference_tunnel=inference_tunnel.InferenceTunnelConfig(backend="modal"),
        train=SimpleNamespace(workflow_config=Mock()),
    )
    monkeypatch.setattr(train_tinker, "load_config", Mock(return_value=(config, [])))
    monkeypatch.setattr(train_tinker, "load_data", Mock(return_value=({}, {})))
    monkeypatch.setattr(train_tinker, "PlatoonTinkerRLTrainer", Mock(return_value=TrainerContext()))
    monkeypatch.setattr(train_tinker, "GroupRolloutWorkflow", Mock())
    return trainer


@pytest.mark.parametrize("failure", [None, "start", "local_health", "tunnel", "train", "cancel"])
def test_proxy_routing_restores_model_and_cleans_up(monkeypatch, failure):
    model = SimpleNamespace(
        model_name="tinker/model",
        base_url=None,
        api_key=None,
        llm=SimpleNamespace(context_window_length=42048),
    )
    proxy = Mock(base_url="http://127.0.0.1:9000/v1", model_name="openai/platoon-tinker", api_key="key")
    if failure == "start":
        proxy.start.side_effect = RuntimeError("start")
    factory = Mock(return_value=proxy)
    monkeypatch.setattr(train_tinker, "FastAPILiteLLMTinkerHTTPProxyServer", factory)
    health = Mock(side_effect=RuntimeError("local_health") if failure == "local_health" else None)
    monkeypatch.setattr(train_tinker, "wait_for_health", health)
    closed = []

    @contextmanager
    def tunnel(urls, config):
        assert urls == ["http://127.0.0.1:9000"]
        assert config.backend == "modal"
        if failure == "tunnel":
            raise RuntimeError("tunnel")
        try:
            yield {urls[0]: "https://router.modal.host"}
        finally:
            proxy.stop.assert_not_called()
            closed.append(True)

    monkeypatch.setattr(train_tinker, "inference_tunnels", tunnel)

    async def train(**kwargs):
        wrapped = train_tinker.GroupRolloutWorkflow.call_args.kwargs["rollout_fn"]
        assert isinstance(wrapped, train_tinker.TinkerProxyRollout)
        assert wrapped.proxy is proxy
        assert wrapped.rollout_fn is train_tinker.run_rollout
        assert model.base_url == "https://router.modal.host/v1"
        assert model.api_key == "key"
        assert model.model_name == "openai/platoon-tinker"
        if failure == "train":
            raise RuntimeError("train")
        if failure == "cancel":
            raise asyncio.CancelledError

    mock_training(monkeypatch, model, train)

    def run():
        asyncio.run(train_tinker.main([]))

    if failure:
        with pytest.raises(asyncio.CancelledError if failure == "cancel" else RuntimeError):
            run()
    else:
        run()
    assert (model.model_name, model.base_url, model.api_key) == ("tinker/model", None, None)
    proxy.stop.assert_called_once()
    assert bool(closed) == (failure in {None, "train", "cancel"})
    assert factory.call_args.kwargs["context_window_length"] == 42048


@pytest.mark.parametrize("failure", [None, "readiness", "health", "body"])
def test_modal_router_preserves_legacy_transport_and_cleanup(monkeypatch, failure):
    sdk = SimpleNamespace(**{name: Mock() for name in ("Image", "App", "Secret", "Probe", "Sandbox")})
    monkeypatch.setitem(sys.modules, "modal", sdk)
    sandbox = sdk.Sandbox.create.return_value
    sandbox.tunnels.return_value = {
        2222: SimpleNamespace(tcp_socket=("relay.modal.host", 2345)),
        10000: SimpleNamespace(url="https://public.modal.host"),
    }
    key_paths = []

    def keygen(args, **kwargs):
        path = Path(args[-1])
        key_paths.append(path)
        path.write_text("private-key")
        path.with_suffix(".pub").write_text("ssh-ed25519 public-key\n")

    monkeypatch.setattr(modal_tunnel.subprocess, "run", keygen)
    monkeypatch.setattr(
        modal_tunnel, "wait_for_ssh", Mock(side_effect=RuntimeError("readiness") if failure == "readiness" else None)
    )
    forward = Mock()
    closed = []

    @contextmanager
    def ssh(target, options):
        assert target == "root@relay.modal.host"
        assert "StrictHostKeyChecking=yes" in options
        assert "HostKeyAlias=platoon-modal-router" in options
        try:
            yield forward
        finally:
            sandbox.terminate.assert_not_called()
            closed.append(True)

    monkeypatch.setattr(modal_tunnel, "ssh_connection", ssh)
    monkeypatch.setattr(
        inference_tunnel, "wait_for_health", Mock(side_effect=RuntimeError("health") if failure == "health" else None)
    )

    def run():
        with inference_tunnel.inference_tunnels(
            ["http://127.0.0.1:9000"], inference_tunnel.InferenceTunnelConfig(backend="modal")
        ) as endpoints:
            assert endpoints == {"http://127.0.0.1:9000": "https://public.modal.host"}
            forward.assert_called_once_with("http://127.0.0.1:9000", 10000)
            if failure == "body":
                raise RuntimeError("body")

    if failure:
        with pytest.raises(RuntimeError, match=failure):
            run()
    else:
        run()
    kwargs = sdk.Sandbox.create.call_args.kwargs
    assert (kwargs["cpu"], kwargs["memory"], kwargs["timeout"]) == (0.125, 128, 86400)
    assert kwargs["encrypted_ports"] == [10000]
    assert kwargs["unencrypted_ports"] == [2222]
    sandbox.terminate.assert_called_once()
    assert all(not path.parent.exists() for path in key_paths)
    assert bool(closed) == (failure != "readiness")


def test_proxy_rollout_session_isolation(monkeypatch):
    async def completion(**kwargs):
        marker = kwargs["messages"][0]["content"]
        proxy_interactions.get()["response"] = marker
        return {"id": marker, "choices": []}

    monkeypatch.setattr(fastapi_litellm_proxy.litellm, "acompletion", completion)
    proxy = fastapi_litellm_proxy.FastAPILiteLLMTinkerHTTPProxyServer("test-model")
    with TestClient(proxy.app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/v1/models").status_code == 200
        for session in ["a", "b"]:
            response = client.post(
                "/v1/chat/completions",
                headers={"X-Platoon-Tinker-Session": session},
                json={"messages": [{"role": "user", "content": session}]},
            )
            assert response.status_code == 200
            assert response.json()["id"] == session
    assert proxy.pop_interactions("a") == {"response": "a"}
    assert proxy.pop_interactions("b") == {"response": "b"}
    assert proxy.api_key == "sk-xxx"


def test_codescout_config_loads_modal_defaults_and_ogma_override():
    config_path = Path(train_tinker.__file__).with_name("train_codescout_fft_tinker.yaml")
    config, _ = load_config([], train_tinker.CodeScoutTinkerConfig, str(config_path))
    assert config.inference_tunnel.backend == "modal"
    assert config.inference_tunnel.modal.cpu == 0.125
    assert config.train.context_window_length == 42048
    assert config.train.workflow_config.rollout_config.inference_params.max_completion_tokens == 2048
    config, _ = load_config(
        ["--inference_tunnel.backend", "ogma"], train_tinker.CodeScoutTinkerConfig, str(config_path)
    )
    assert config.inference_tunnel.backend == "ogma"
