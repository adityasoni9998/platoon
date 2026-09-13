"""Hydrate CodeScout-only assets into an existing SWE-rebench image."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class _Workspace(Protocol):
    def file_upload(self, source_path: str | Path, destination_path: str | Path): ...

    def execute_command(
        self,
        command: str,
        cwd: str | Path | None = None,
        timeout: float = 30.0,
    ): ...


CUSTOM_TOOL_FILES = ("__init__.py", "localization_finish.py")
CUSTOM_TOOL_IMPORT = "custom_tools.localization_finish"


def install_ripgrep(workspace: _Workspace) -> None:
    """Install rg in every disposable sandbox without rebuilding its image."""
    result = workspace.execute_command(
        "apt-get update -qq && "
        "DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ripgrep "
        "&& rg --version",
        timeout=300,
    )
    if result.exit_code != 0:
        raise RuntimeError(f"Runtime ripgrep installation failed: {result.stderr}")


def hydrate_custom_tool(workspace: _Workspace, local_package: Path) -> None:
    """Upload and import-check the finish tool without changing the Modal image."""
    # Upload into a directory already on the running server's sys.path. Its
    # working directory need not be /agent-server (the image can use /testbed).
    discovery = workspace.execute_command(
        "/agent-server/.venv/bin/python -I -c "
        "'import sysconfig; print(sysconfig.get_path(\"purelib\"))'",
        cwd="/agent-server",
        timeout=30,
    )
    if discovery.exit_code != 0:
        raise RuntimeError(f"Could not locate server site-packages: {discovery.stderr}")
    site_packages = Path(discovery.stdout.strip())
    if not site_packages.is_absolute() or "site-packages" not in site_packages.parts:
        raise RuntimeError(f"Unexpected server site-packages: {site_packages}")
    remote_package = site_packages / "custom_tools"
    for filename in CUSTOM_TOOL_FILES:
        source = local_package / filename
        if not source.is_file():
            raise FileNotFoundError(f"CodeScout custom tool asset not found: {source}")
        destination = remote_package / filename
        result = workspace.file_upload(source, destination)
        if not result.success:
            raise RuntimeError(f"Failed to upload {source} to {destination}: {result.error}")

    # Exclude the checkout and PYTHONPATH from module resolution. For example,
    # /testbed/httpx must not shadow the server's installed HTTPX dependency.
    check = workspace.execute_command(
        "/agent-server/.venv/bin/python -I -c 'import custom_tools.localization_finish'",
        cwd="/agent-server",
        timeout=30,
    )
    if check.exit_code != 0:
        raise RuntimeError(f"Remote CodeScout custom tool import failed: {check.stderr}")


def read_system_prompt(prompt_path: Path) -> str:
    """Read the host-side prompt that is serialized inline with the Agent."""
    if not prompt_path.is_file():
        raise FileNotFoundError(f"CodeScout system prompt not found: {prompt_path}")
    prompt = prompt_path.read_text()
    if not prompt.strip():
        raise ValueError(f"CodeScout system prompt is empty: {prompt_path}")
    return prompt
