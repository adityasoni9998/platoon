"""Install the capture hook, download JSONL, and associate captures with completions."""

import hashlib
import json
import shlex
from pathlib import Path
from tempfile import TemporaryDirectory

from openhands.sdk.event.hook_execution import HookExecutionEvent
from openhands.sdk.hooks import HookConfig, HookDefinition, HookMatcher
from platoon.utils.openhands_utils import group_actions

HOOK_DIR = "/tmp/hooks"
EMPTY_PATCH_HASH = hashlib.sha256(b"").hexdigest()


def capture_command(instance):
    return shlex.join([
        "python3", f"{HOOK_DIR}/capture_patch.py", "--repo", instance["repo_path"],
        "--base-commit", instance["base_commit"], "--output-dir", f"{HOOK_DIR}/captures",
    ])


def install_capture_hook(workspace, instance) -> HookConfig:
    result = workspace.execute_command(f"mkdir -p {HOOK_DIR}", timeout=180)
    if result.exit_code != 0 or result.timeout_occurred:
        raise RuntimeError(f"Cannot prepare capture directory: {result.stderr}")
    script = Path(__file__).parent / "hooks" / "capture_patch.py"
    result = workspace.file_upload(script, f"{HOOK_DIR}/capture_patch.py")
    if not result.success:
        raise RuntimeError(f"Cannot upload capture hook: {result.error}")
    command = capture_command(instance)
    result = workspace.execute_command(command, timeout=180)
    if result.exit_code != 0 or result.timeout_occurred:
        raise RuntimeError(f"Cannot initialize capture: {result.stderr}")
    return HookConfig(post_tool_use=[HookMatcher(
        matcher="*", hooks=[HookDefinition(command=command, timeout=180, async_=False)],
    )])


def download_snapshots(workspace, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir, prefix="turn-snapshots-", ignore_cleanup_errors=True) as temp_dir:
        destination = Path(temp_dir) / "snapshots.jsonl"
        result = workspace.file_download(f"{HOOK_DIR}/captures/snapshots.jsonl", destination)
        if not result.success:
            raise RuntimeError(f"Cannot download turn snapshots: {result.error}")
        with destination.open() as stream:
            return {record["patch_hash"]: record for record in map(json.loads, stream)}


def completion_snapshots(events, command: str) -> dict[str, str]:
    _, action_to_completion, _, _ = group_actions(events)
    captures = {}
    for event in events:
        if not isinstance(event, HookExecutionEvent) or event.hook_command != command:
            continue
        if event.hook_event_type != "PostToolUse":
            continue
        if not event.success or event.exit_code != 0:
            raise ValueError(f"Turn capture failed: {event.error or event.stderr}")
        completion = action_to_completion[event.action_id]
        # The SDK waits for the tool batch before emitting observations/hooks.
        captures[completion] = json.loads(event.stdout)["patch_hash"]
    # Turns without a hook reuse the previous snapshot during evaluation.
    return captures
