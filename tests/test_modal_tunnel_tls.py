"""Modal proxy transport selection, TLS verification, and cleanup."""

import shlex
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from platoon.inference import modal_tunnel
from platoon.inference.inference_tunnel import ModalRouterConfig


@pytest.mark.parametrize("tls", [False, True])
@pytest.mark.parametrize("fail_forward", [False, True])
def test_transport_and_cleanup(monkeypatch, tls, fail_forward):
    sdk = SimpleNamespace(**{name: Mock() for name in ("Image", "App", "Secret", "Probe", "Sandbox")})
    monkeypatch.setitem(sys.modules, "modal", sdk)
    monkeypatch.setattr(modal_tunnel.shutil, "which", lambda name: "/usr/bin/openssl")
    sandbox = sdk.Sandbox.create.return_value
    sandbox.tunnels.return_value = {
        2222: SimpleNamespace(host="tls.modal.host", port=443, tcp_socket=("tcp.modal.host", 2345)),
        10000: SimpleNamespace(url="https://proxy.modal.host"),
    }

    def keygen(args, **kwargs):
        path = Path(args[-1])
        path.write_text("private-key")
        path.with_suffix(".pub").write_text("ssh-ed25519 public-key\n")

    monkeypatch.setattr(modal_tunnel.subprocess, "run", keygen)
    readiness = Mock()
    monkeypatch.setattr(modal_tunnel, "wait_for_ssh", readiness)
    forwarded = []
    closed = []

    @contextmanager
    def ssh(target, options):
        assert target == ("root@tls.modal.host" if tls else "root@tcp.modal.host")
        assert "StrictHostKeyChecking=yes" in options
        assert "HostKeyAlias=platoon-modal-router" in options
        proxies = [option for option in options if option.startswith("ProxyCommand=")]
        if tls:
            command = shlex.split(proxies[0].split("=", 1)[1])
            assert command[:2] == ["openssl", "s_client"]
            assert command[command.index("-connect") + 1] == "tls.modal.host:443"
            assert command[command.index("-servername") + 1] == "tls.modal.host"
            assert command[command.index("-verify_hostname") + 1] == "tls.modal.host"
            assert "-verify_return_error" in command
            assert "-nocommands" in command
            readiness.assert_called_once_with(sandbox, "tls.modal.host", 443, tls=True)
        else:
            assert not proxies
            readiness.assert_called_once_with(sandbox, "tcp.modal.host", 2345)

        def forward(url, port):
            if fail_forward:
                raise RuntimeError("forward failed")
            forwarded.append((url, port))

        try:
            yield forward
        finally:
            sandbox.terminate.assert_not_called()
            closed.append(True)

    monkeypatch.setattr(modal_tunnel, "ssh_connection", ssh)

    def run():
        with modal_tunnel.modal_tunnels(["http://127.0.0.1:9000/v1"], ModalRouterConfig(ssh_over_tls=tls)) as endpoints:
            assert endpoints == {"http://127.0.0.1:9000/v1": "https://proxy.modal.host/v1"}

    if fail_forward:
        with pytest.raises(RuntimeError, match="forward failed"):
            run()
    else:
        run()
        assert forwarded == [("http://127.0.0.1:9000/v1", 10000)]
    assert closed == [True]
    sandbox.terminate.assert_called_once()
    options = sdk.Sandbox.create.call_args.kwargs
    assert options["encrypted_ports"] == ([10000, 2222] if tls else [10000])
    assert options["unencrypted_ports"] == ([] if tls else [2222])


def test_missing_openssl_fails_before_creating_sandbox(monkeypatch):
    sdk = SimpleNamespace(Sandbox=Mock())
    monkeypatch.setitem(sys.modules, "modal", sdk)
    monkeypatch.setattr(modal_tunnel.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="requires openssl"):
        with modal_tunnel.modal_tunnels(["http://127.0.0.1:9000"], ModalRouterConfig(ssh_over_tls=True)):
            pytest.fail("The tunnel must not start without TLS support")
    sdk.Sandbox.create.assert_not_called()
