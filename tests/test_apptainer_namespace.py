"""Mapping lifetime across exec'd workers and workspace launch selection."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from platoon.openhands import apptainer as workspace_module
from platoon.openhands import apptainer_namespace as namespace_module


def test_mapping_is_owned_once_until_workers_exit(tmp_path, monkeypatch):
    monkeypatch.delenv(namespace_module.NAMESPACE_ENV, raising=False)
    marker = tmp_path / "namespace"
    marker.write_text("mapping")
    descriptor = os.open(marker, os.O_RDONLY)
    create = Mock(return_value=descriptor)
    monkeypatch.setattr(namespace_module, "_create_namespace", create)
    reference = None
    with pytest.raises(RuntimeError, match="training failed"):
        with namespace_module.shared_apptainer_namespace(enabled=True):
            reference = os.environ[namespace_module.NAMESPACE_ENV]
            # A nested owner and separately exec'd workers reuse the same handle.
            with namespace_module.shared_apptainer_namespace(enabled=True):
                workers = [
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-c",
                            "import os; from pathlib import Path; "
                            f"print(Path(os.environ[{namespace_module.NAMESPACE_ENV!r}]).read_text())",
                        ],
                        stdout=subprocess.PIPE,
                        text=True,
                    )
                    for _ in range(2)
                ]
                for worker in workers:
                    output, _ = worker.communicate(timeout=10)
                    assert worker.returncode == 0 and output.strip() == "mapping"
            assert Path(reference).exists()
            create.assert_called_once()
            raise RuntimeError("training failed")
    assert namespace_module.NAMESPACE_ENV not in os.environ
    assert not Path(reference).exists()


def test_other_backends_do_not_map_ids(monkeypatch):
    create = Mock(side_effect=AssertionError("Unexpected namespace setup"))
    monkeypatch.setattr(namespace_module, "_create_namespace", create)
    with namespace_module.shared_apptainer_namespace(enabled=False):
        pass
    create.assert_not_called()


@pytest.mark.parametrize("cached,use_fakeroot", [(False, True), (True, True), (True, False)])
def test_workspace_uses_cached_mapping_only_for_fakeroot(tmp_path, monkeypatch, cached, use_fakeroot):
    namespace = "/proc/123/fd/8"
    if cached:
        monkeypatch.setenv(namespace_module.NAMESPACE_ENV, namespace)
    else:
        monkeypatch.delenv(namespace_module.NAMESPACE_ENV, raising=False)
    monkeypatch.setattr(workspace_module.ApptainerWorkspace, "model_post_init", lambda self, context: None)
    workspace = workspace_module.DirectoryOverlayWorkspace.model_construct(
        scratch_dir=str(tmp_path),
        cache_dir=str(tmp_path / "cache"),
        host_port=12345,
        detach_logs=False,
        use_fakeroot=use_fakeroot,
    )
    workspace._sif_path = "/image.sif"
    popen = Mock()
    monkeypatch.setattr(workspace_module.subprocess, "Popen", popen)
    monkeypatch.setattr(workspace_module.ApptainerWorkspace, "cleanup", Mock())
    try:
        workspace._start_container()
        command = popen.call_args.args[0]
        if cached and use_fakeroot:
            assert command[:6] == namespace_module.namespace_command(namespace, [])
            assert command[6:8] == ["apptainer", "run"]
            assert "--fakeroot" not in command
        else:
            assert command[:2] == ["apptainer", "run"]
            assert ("--fakeroot" in command) == use_fakeroot
        assert "--overlay" in command and "--containall" in command
        assert "--net" not in command
    finally:
        workspace.cleanup()
