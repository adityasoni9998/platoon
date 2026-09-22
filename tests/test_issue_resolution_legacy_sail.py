"""Check Sail workspace boundaries without creating hosted resources."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins/issue-resolution-legacy-sail"
PACKAGE = PLUGIN / "platoon/issue_resolution_sail"
LEGACY = ROOT / "plugins/issue-resolution-legacy/platoon/issue_resolution"


def test_training_and_rewards_preserve_legacy():
    current = yaml.safe_load((PACKAGE / "train_issue_resolution_fft_tinker.yaml").read_text())
    original = yaml.safe_load((LEGACY / "train_issue_resolution_fft_tinker.yaml").read_text())
    for key in ("train", "eval", "checkpoint", "inference_tunnel"):
        assert current[key] == original[key]
    for name in (
        "easy_instances.py",
        "prompts/default.j2",
        "env.py",
        "tinker_workflow.py",
        "subprocess_worker.py",
        "test_execution_reward/test_execution_reward.py",
    ):
        expected = (LEGACY / name).read_text().replace("platoon.issue_resolution", "platoon.issue_resolution_sail")
        assert (PACKAGE / name).read_text() == expected


def test_sail_workspace_uses_registry_image(monkeypatch):
    from platoon.issue_resolution_sail import rollout, tasks

    factory = Mock()
    monkeypatch.setattr(rollout, "SailWorkspace", factory)
    monkeypatch.setenv("SAILBOX_MEMORY_GIB", "4")
    monkeypatch.delenv("SAIL_EXPECTED_SERVER_GIT_SHA", raising=False)
    instance = {"image_name": "swebench/swesmith.x86_64.example.abc123"}
    rollout._sail_workspace(instance)
    kwargs = factory.call_args.kwargs
    assert kwargs["server_image"] == tasks.agent_server_image_for_instance(instance)
    assert kwargs["server_image"].startswith("docker.io/adityasoni8/eval-agent-server:d3fd1c4-")
    assert kwargs["expected_server_git_sha"] == tasks.SDK_GIT_SHA
    assert kwargs["target_type"] == "source"
    assert kwargs["memory_limit_gib"] == 4
    assert "named_server_image" not in kwargs


@pytest.mark.parametrize("fail_all", [False, True])
def test_workspace_setup_cleans_failed_attempts(monkeypatch, fail_all):
    from platoon.issue_resolution_sail import rollout

    bad = Mock()
    bad.execute_command.return_value = SimpleNamespace(exit_code=1, stderr="copy failed")
    good = Mock()
    good.execute_command.return_value = SimpleNamespace(exit_code=0, stderr="")
    factory = Mock(side_effect=[bad, bad, bad] if fail_all else [bad, good])
    monkeypatch.setattr(rollout, "_sail_workspace", factory)
    instance = {"instance_id": "example.123", "repo": "owner/example"}
    if fail_all:
        with pytest.raises(RuntimeError, match="Error preparing Sail workspace"):
            rollout.prepare_workspace(instance)
        assert bad.cleanup.call_count == 3
    else:
        assert rollout.prepare_workspace(instance) is good
        bad.cleanup.assert_called_once()
        good.cleanup.assert_not_called()
        assert instance["repo_path"] == "/workspace/example/"
        assert [call.kwargs["timeout"] for call in good.execute_command.call_args_list] == [900, 300, 300]


def test_builder_deduplicates_subset_and_matches_runtime_tags(monkeypatch):
    from platoon.issue_resolution_sail.tasks import agent_server_image_for_instance

    spec = importlib.util.spec_from_file_location("sail_image_builder", PLUGIN / "scripts/build_agent_images.py")
    module = importlib.util.module_from_spec(spec)
    import sys

    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    row = {"image_name": "swebench/swesmith.x86_64.example.abc123"}
    task = SimpleNamespace(misc=row)
    monkeypatch.setattr(module, "load_data", lambda: ({"a": task, "b": task}, {}))
    images = module.easy_base_images()
    assert images == ["docker.io/" + row["image_name"]]
    requests = module.build_requests(images, push=True)
    assert len(requests) == 1
    request = requests[0]
    assert request.push
    actual = f"{request.destination_image}:{module.SDK_SHORT_SHA}-{request.custom_tag}-source-minimal"
    assert actual == agent_server_image_for_instance(row)
