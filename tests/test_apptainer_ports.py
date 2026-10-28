"""Real socket checks for workspace port claims, without launching Apptainer."""

import socket
import subprocess
import sys
from unittest.mock import Mock

import pytest

from platoon.openhands import apptainer as apptainer_module
from platoon.openhands.apptainer import DirectoryOverlayWorkspace


@pytest.fixture
def workspace_factory(tmp_path, monkeypatch):
    monkeypatch.setattr(apptainer_module.ApptainerWorkspace, "model_post_init", lambda self, context: None)
    workspaces = []

    def create(**kwargs):
        workspace = DirectoryOverlayWorkspace.model_construct(scratch_dir=str(tmp_path), **kwargs)
        workspaces.append(workspace)
        return workspace

    yield create
    for workspace in workspaces:
        workspace.cleanup()


def test_claims_are_exclusive_across_processes_before_server_start(workspace_factory):
    workspace = workspace_factory()
    port = workspace.host_port
    # There is no TCP server yet, but another rollout process cannot claim this
    # port. No shared Python state or inherited descriptors are used.
    result = subprocess.run(
        [sys.executable, "-c", r"""
import errno
import socket
import sys
port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
    probe.bind(('0.0.0.0', port))
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as claim:
    try:
        claim.bind(f'\0platoon-apptainer-port-{port}')
    except OSError as exc:
        assert exc.errno == errno.EADDRINUSE
    else:
        raise AssertionError('Claim was not exclusive')
""", str(port)],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    with pytest.raises(OSError):
        workspace_factory(host_port=port)
    workspace.cleanup()
    assert workspace._port_claim is None
    replacement = workspace_factory(host_port=port)
    assert replacement.host_port == port


def test_selection_skips_claims_and_tcp_listeners(workspace_factory, monkeypatch):
    first = workspace_factory()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        occupied = listener.getsockname()[1]
        with pytest.raises(OSError):
            workspace_factory(host_port=occupied)
        rng = apptainer_module.random.SystemRandom()
        candidates = iter([first.host_port, occupied])
        choose = Mock(side_effect=lambda low, high: next(candidates, None) or rng.randint(low, high))
        monkeypatch.setattr(apptainer_module.random, "SystemRandom", lambda: Mock(randint=choose))
        second = workspace_factory()
        assert second.host_port not in (first.host_port, occupied)
        assert choose.call_count >= 3


def test_failed_start_releases_claim(workspace_factory, monkeypatch):
    ports = []

    def fail(self, context):
        ports.append(self.host_port)
        raise RuntimeError("startup failed")

    monkeypatch.setattr(apptainer_module.ApptainerWorkspace, "model_post_init", fail)
    with pytest.raises(RuntimeError, match="startup failed"):
        workspace_factory()
    monkeypatch.setattr(apptainer_module.ApptainerWorkspace, "model_post_init", lambda self, context: None)
    workspace = workspace_factory(host_port=ports[0])
    assert workspace.host_port == ports[0]


def test_extra_ports_are_rejected(workspace_factory):
    with pytest.raises(ValueError, match="only the agent-server port"):
        workspace_factory(extra_ports=True)


@pytest.mark.parametrize("requested_port", [None, 12345])
@pytest.mark.parametrize("error", [PermissionError("denied"), RuntimeError("failed"), KeyboardInterrupt()])
def test_retry_policy_and_failed_claim_release(workspace_factory, monkeypatch, requested_port, error):
    real_socket = socket.socket
    failed_claim = Mock()
    failed_claim.bind.side_effect = error
    calls = 0

    def create_socket(*args, **kwargs):
        nonlocal calls
        calls += 1
        return failed_claim if calls == 1 else real_socket(*args, **kwargs)

    monkeypatch.setattr(apptainer_module.socket, "socket", create_socket)
    if requested_port is None and isinstance(error, Exception):
        workspace_factory()
        assert calls >= 3  # Failed claim, successful claim, TCP probe.
    else:
        with pytest.raises(type(error)):
            workspace_factory(host_port=requested_port)
        assert calls == 1
    failed_claim.close.assert_called_once()
