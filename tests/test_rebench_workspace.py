"""Exercise local workspace setup without pulling images or starting a training job."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from platoon.issue_resolution import rollout, tinker_workflow
from platoon.issue_resolution.workspace import ApptainerConfig, WorkspaceConfig
from platoon.openhands import apptainer as apptainer_module
from platoon.openhands.apptainer import DirectoryOverlayWorkspace
from pydantic import ValidationError

from platoon.inference import inference_tunnel


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    """Stand in for the runtime, exercising real process startup/health/cleanup."""
    binary = tmp_path / "apptainer"
    binary.write_text(f"#!{sys.executable}\n" + '''
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

if sys.argv[1] == "version":
    print("test-runtime")
    sys.exit(0)
Path(os.environ["TEST_APPTAINER_RECORD"]).write_text(json.dumps({
    "args": sys.argv[1:], "pid": os.getpid(),
    "env": {key: os.environ.get(key) for key in ("APPTAINER_CACHEDIR", "APPTAINER_TMPDIR", "TMPDIR")},
}))
if os.environ.get("TEST_APPTAINER_FAIL"):
    sys.exit(1)
class Health(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
    def log_message(self, *args):
        pass
HTTPServer(("127.0.0.1", int(sys.argv[sys.argv.index("--port") + 1])), Health).serve_forever()
''')
    binary.chmod(0o700)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    record = tmp_path / "launch.json"
    monkeypatch.setenv("TEST_APPTAINER_RECORD", str(record))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    shared_tmp = tmp_path / "shared-runtime-tmp"
    shared_tmp.mkdir()
    (shared_tmp / "other-workspace-data").write_text("keep")
    host_tmp = tmp_path / "host-tmp"
    host_tmp.mkdir()
    monkeypatch.setenv("APPTAINER_TMPDIR", str(shared_tmp))
    monkeypatch.setenv("TMPDIR", str(host_tmp))
    sif = tmp_path / "instance.sif"
    sif.touch()
    return SimpleNamespace(
        record=record, scratch=scratch, shared_tmp=shared_tmp, host_tmp=host_tmp,
        kwargs=dict(sif_file=str(sif), cache_dir=str(tmp_path / "cache"),
                    scratch_dir=str(scratch), detach_logs=False, health_check_timeout=5),
    )


def test_directory_overlay_is_private_disk_backed_and_cleaned(runtime):
    original_tmp = os.environ.get("APPTAINER_TMPDIR")
    with DirectoryOverlayWorkspace(**runtime.kwargs) as first:
        launch = json.loads(runtime.record.read_text())
        args = launch["args"]
        overlay = Path(args[args.index("--overlay") + 1])
        workdir = Path(args[args.index("--workdir") + 1])
        assert overlay.parent.parent == runtime.scratch
        assert overlay.parent.stat().st_mode & 0o777 == 0o700
        assert (overlay / "upper").is_dir() and (overlay / "work").is_dir()
        assert workdir.is_dir() and workdir.parent == overlay.parent
        assert "--compat" not in args and "--writable-tmpfs" not in args
        assert "--net" not in args
        assert args[args.index("--bind") + 1] == "/etc/resolv.conf:/etc/resolv.conf:ro"
        assert launch["env"]["APPTAINER_CACHEDIR"] == runtime.kwargs["cache_dir"]
        assert Path(launch["env"]["APPTAINER_TMPDIR"]).parent == overlay.parent
        assert launch["env"]["TMPDIR"] == launch["env"]["APPTAINER_TMPDIR"]
        assert os.environ.get("APPTAINER_TMPDIR") == original_tmp
        assert os.environ["TMPDIR"] == str(runtime.host_tmp)
        with DirectoryOverlayWorkspace(**runtime.kwargs) as second:
            assert first._runtime_dir != second._runtime_dir
            assert first._process.poll() is None and second._process.poll() is None
        process = first._process
    assert process.poll() is not None
    assert not list(runtime.scratch.iterdir())
    first.cleanup()  # Rollout/environment teardown may both request cleanup.
    assert (runtime.shared_tmp / "other-workspace-data").read_text() == "keep"


def test_failed_constructor_reaps_process_and_removes_overlay(runtime, monkeypatch):
    monkeypatch.setenv("TEST_APPTAINER_FAIL", "1")
    with pytest.raises(RuntimeError, match="Container process stopped"):
        DirectoryOverlayWorkspace(**runtime.kwargs)
    pid = json.loads(runtime.record.read_text())["pid"]
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert not list(runtime.scratch.iterdir())
    assert (runtime.shared_tmp / "other-workspace-data").read_text() == "keep"


@pytest.mark.parametrize("error", [
    PermissionError("protected file"), OSError("busy filesystem"), RuntimeError("removal failed"),
])
def test_cleanup_ignores_removal_errors_without_retrying(runtime, monkeypatch, error):
    workspace = DirectoryOverlayWorkspace(**runtime.kwargs)
    directory = workspace._runtime_dir
    process = workspace._process

    def remove(path):
        assert path == directory
        assert process.poll() is not None
        workspace.cleanup()  # A second caller must return while cleanup is in progress.
        raise error

    with monkeypatch.context() as patch:
        remove_mock = Mock(side_effect=remove)
        run = Mock(side_effect=AssertionError("No fallback container should start"))
        patch.setattr(apptainer_module.shutil, "rmtree", remove_mock)
        patch.setattr(apptainer_module.subprocess, "run", run)
        workspace.cleanup()
        workspace.cleanup()
        remove_mock.assert_called_once_with(directory)
        run.assert_not_called()
    assert workspace.cleaned_up and directory.exists()
    assert (runtime.shared_tmp / "other-workspace-data").read_text() == "keep"


def test_cleanup_attempts_removal_even_if_process_has_not_exited(tmp_path, monkeypatch):
    monkeypatch.setattr(apptainer_module.ApptainerWorkspace, "model_post_init", lambda self, context: None)
    workspace = DirectoryOverlayWorkspace.model_construct(scratch_dir=str(tmp_path))
    directory = tmp_path / "runtime"
    directory.mkdir()
    workspace._runtime_dir = directory
    workspace._process = Mock()
    workspace._process.poll.return_value = None
    stop = Mock()
    monkeypatch.setattr(apptainer_module.ApptainerWorkspace, "cleanup", stop)
    workspace.cleanup()
    workspace.cleanup()
    stop.assert_called_once()
    workspace._process.wait.assert_not_called()
    assert workspace.cleaned_up and not directory.exists()


def test_apptainer_paths_and_missing_configuration(runtime):
    config = ApptainerConfig(
        sif_dir=str(runtime.record.parent),
        cache_dir=runtime.kwargs["cache_dir"], scratch_dir=str(runtime.scratch),
    )
    with config.create_workspace({"instance_id": "instance"}) as workspace:
        assert workspace.sif_file == runtime.kwargs["sif_file"]
        assert workspace.working_dir == "/testbed"
        assert workspace.health_check_timeout == 600
    with pytest.raises(FileNotFoundError, match="Prebuilt Apptainer SIF"):
        config.create_workspace({"instance_id": "missing"})
    for field in ("sif_dir", "cache_dir", "scratch_dir"):
        values = config.model_dump()
        values[field] = None
        with pytest.raises(ValidationError, match=field):
            ApptainerConfig.model_validate(values)
        del values[field]
        with pytest.raises(ValidationError, match=field):
            ApptainerConfig.model_validate(values)
    assert WorkspaceConfig().type == "modal"
    for raw in ({"type": "typo"}, {"type": "apptainer"}, {"typo": "modal"}):
        with pytest.raises(ValidationError):
            WorkspaceConfig.model_validate(raw)


def test_apptainer_selection_uses_existing_setup_retries(monkeypatch):
    workspace = Mock()
    workspace.execute_command.return_value = SimpleNamespace(exit_code=0)
    create = Mock(side_effect=[RuntimeError("startup failed"), workspace])
    monkeypatch.setattr(ApptainerConfig, "create_workspace", create)
    modal = Mock(side_effect=AssertionError("Modal agent workspace should not start"))
    monkeypatch.setattr(rollout, "_modal_workspace", modal)
    result, hook = rollout.prepare_workspace(
        {"instance_id": "instance"}, workspace_config={
            "type": "apptainer",
            "apptainer": {"sif_dir": "unused", "cache_dir": "unused", "scratch_dir": "unused"},
        },
    )
    assert result is workspace and hook is None and create.call_count == 2
    assert "git reset --hard HEAD && git clean -fd" in workspace.execute_command.call_args.args[0]


def test_tinker_no_relay_keeps_local_proxy_and_restores_model(monkeypatch):
    model = SimpleNamespace(model_name="original", base_url=None, api_key=None,
                            llm=SimpleNamespace(context_window_length=4096))
    proxy = Mock(base_url="http://127.0.0.1:9000/v1", model_name="proxy", api_key="key")
    monkeypatch.setattr(tinker_workflow, "FastAPILiteLLMTinkerHTTPProxyServer", Mock(return_value=proxy))
    ssh = Mock(side_effect=AssertionError("No relay should start"))
    monkeypatch.setattr(inference_tunnel, "ssh_connection", ssh)
    with tinker_workflow.routed_tinker_proxy(
        model, inference_tunnel.InferenceTunnelConfig(backend="none"),
    ):
        assert model.base_url == proxy.base_url
        assert model.model_name == "proxy"
    ssh.assert_not_called()
    proxy.start.assert_called_once()
    proxy.stop.assert_called_once()
    assert (model.model_name, model.base_url, model.api_key) == ("original", None, None)
