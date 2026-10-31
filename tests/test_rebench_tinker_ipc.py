"""Rollout IPC files are read and deleted off the event loop, and always cleaned up."""

import asyncio
import pickle
import threading
from types import SimpleNamespace

import pytest

from platoon.issue_resolution import tinker_workflow


def _config(scratch):
    return SimpleNamespace(timeout=1800, extra={"workspace": {"type": "apptainer", "apptainer": {"scratch_dir": str(scratch)}}})


def _run(scratch, launch, monkeypatch):
    threads = {}
    real_load, real_rmtree = tinker_workflow._load_response, tinker_workflow.shutil.rmtree

    def load(path):
        threads["load"] = threading.get_ident()
        return real_load(path)

    def rmtree(path):
        threads["rmtree"] = threading.get_ident()
        real_rmtree(path)

    monkeypatch.setattr(tinker_workflow, "_load_response", load)
    monkeypatch.setattr(tinker_workflow.shutil, "rmtree", rmtree)
    rollout = tinker_workflow.TinkerProxyRollout(lambda: None, None)
    monkeypatch.setattr(rollout, "_launch_worker", launch)

    async def main():
        threads["loop"] = threading.get_ident()
        return await rollout._run_subprocess(SimpleNamespace(id="t"), _config(scratch))

    return asyncio.run(main()), threads


def _child(returncode, result=None):
    async def launch(request, response, config):
        assert pickle.loads(request.read_bytes())[0].id == "t"
        if result is not None:
            response.write_bytes(pickle.dumps(result))

        async def wait():
            return returncode

        return SimpleNamespace(returncode=returncode, wait=wait)

    return launch


def test_result_is_loaded_and_directory_removed_off_the_event_loop(tmp_path, monkeypatch):
    result, threads = _run(tmp_path, _child(0, {"trajectories": {"a": {}}}), monkeypatch)
    assert result == {"trajectories": {"a": {}}}
    assert threads["load"] != threads["loop"] and threads["rmtree"] != threads["loop"]
    assert list(tmp_path.iterdir()) == []


def test_failed_child_still_removes_the_directory(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="exited with code 1"):
        _run(tmp_path, _child(1), monkeypatch)
    assert list(tmp_path.iterdir()) == []


def test_launch_failure_still_removes_the_directory(tmp_path, monkeypatch):
    async def launch(request, response, config):
        raise OSError("no fork server")

    with pytest.raises(OSError, match="no fork server"):
        _run(tmp_path, launch, monkeypatch)
    assert list(tmp_path.iterdir()) == []
