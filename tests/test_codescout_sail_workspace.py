"""CodeScout workspace selection and Apptainer command construction."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from platoon.codescout_sail import rollout
from platoon.codescout_sail import workspace as workspace_module
from platoon.codescout_sail.workspace import ApptainerConfig, WorkspaceConfig
from platoon.openhands import apptainer as apptainer_module
from platoon.openhands.apptainer import DirectoryOverlayWorkspace
from pydantic import ValidationError


def test_single_sif_config_requires_paths_and_sets_server_command(tmp_path, monkeypatch):
    sif = tmp_path / "codescout.sif"
    sif.touch()
    factory = Mock()
    monkeypatch.setattr(workspace_module, "DirectoryOverlayWorkspace", factory)
    config = ApptainerConfig(
        sif_path=str(sif),
        cache_dir=str(tmp_path / "cache"), scratch_dir=str(tmp_path),
    )
    config.create_workspace("/workspace/repo_task")
    kwargs = factory.call_args.kwargs
    assert kwargs["sif_file"] == str(sif)
    assert kwargs["working_dir"] == "/workspace/repo_task"
    assert kwargs["server_command"] == [
        "/agent-server/.venv/bin/python", "-m", "openhands.agent_server",
    ]
    assert kwargs["detach_logs"] is False and kwargs["health_check_timeout"] == 600
    for field in ("sif_path", "cache_dir", "scratch_dir"):
        values = config.model_dump()
        values[field] = None
        with pytest.raises(ValidationError, match=field):
            ApptainerConfig.model_validate(values)
        del values[field]
        with pytest.raises(ValidationError, match=field):
            ApptainerConfig.model_validate(values)
    config.sif_path = str(tmp_path / "missing.sif")
    with pytest.raises(FileNotFoundError):
        config.create_workspace("/workspace/repo_task")
    assert WorkspaceConfig().type == "sail"
    with pytest.raises(ValidationError):
        WorkspaceConfig(type="apptainer")


@pytest.mark.parametrize("backend", ["sail", "apptainer"])
def test_workspace_selection_preserves_clone_and_retries(monkeypatch, backend):
    failed, ready = Mock(), Mock()
    failed.execute_command.return_value = SimpleNamespace(exit_code=1, stderr="clone failed")
    ready.execute_command.return_value = SimpleNamespace(exit_code=0)
    create = Mock(side_effect=[failed, ready])
    unused = Mock(side_effect=AssertionError("Wrong workspace backend"))
    monkeypatch.setattr(rollout, "_sail_workspace", create if backend == "sail" else unused)
    monkeypatch.setattr(
        ApptainerConfig, "create_workspace", create if backend == "apptainer" else unused,
    )
    config = None if backend == "sail" else {
        "type": "apptainer",
        "apptainer": {"sif_path": "unused", "cache_dir": "unused", "scratch_dir": "unused"},
    }
    path, workspace = rollout.prepare_workspace(
        {"repo": "owner/repo", "instance_id": "task-1"}, workspace_config=config,
    )
    assert path == Path("/workspace/owner_repo_task-1") and workspace is ready
    assert create.call_count == 2
    failed.cleanup.assert_called_once()
    unused.assert_not_called()
    ready.execute_command.assert_called_once_with(
        "git clone --depth 1 --single-branch --branch task-1 "
        "https://github.com/owner/repo.git /workspace/owner_repo_task-1",
        cwd="/workspace", timeout=300,
    )


@pytest.mark.parametrize("server_command", [[], ["/agent-server/.venv/bin/python", "-m", "openhands.agent_server"]])
def test_server_command_uses_exec_and_default_uses_runscript(tmp_path, monkeypatch, server_command):
    monkeypatch.setattr(apptainer_module.ApptainerWorkspace, "model_post_init", lambda self, context: None)
    workspace = DirectoryOverlayWorkspace.model_construct(
        scratch_dir=str(tmp_path), cache_dir=str(tmp_path / "cache"),
        server_command=server_command, host_port=12345, detach_logs=False,
    )
    workspace._sif_path = "/image.sif"
    popen = Mock()
    monkeypatch.setattr(apptainer_module.subprocess, "Popen", popen)
    monkeypatch.setattr(apptainer_module.ApptainerWorkspace, "cleanup", Mock())
    try:
        workspace._start_container()
        command = popen.call_args.args[0]
        assert command[:2] == ["apptainer", "exec" if server_command else "run"]
        assert command[command.index("/image.sif") + 1:] == [
            *server_command, "--host", "0.0.0.0", "--port", "12345",
        ]
        assert "--overlay" in command and "--workdir" in command
        assert command[command.index("--bind") + 1] == "/etc/resolv.conf:/etc/resolv.conf:ro"
    finally:
        workspace.cleanup()
