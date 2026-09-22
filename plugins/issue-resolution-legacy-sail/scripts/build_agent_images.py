#!/usr/bin/env python3
"""Build and push only the legacy easy-subset agent images, using Modal VMs.

Adapted from benchmarks/swesmith/modal_build_images.py (swe_smith_modal).
No named Modal images are published; reward evaluation uses its existing images.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import modal
from modal.container_process import ContainerProcess

from platoon.issue_resolution_sail.tasks import (
    EVAL_AGENT_SERVER_IMAGE,
    SDK_GIT_SHA,
    load_data,
)


APP_NAME = "swesmith-sail-image-builder"
BENCHMARKS_REPOSITORY = "https://github.com/adityasoni9998/benchmarks.git"
BENCHMARKS_SHA = "81de32bc6107cbcfe4f1c8f5b2efba8255c71b03"
SDK_SHA = SDK_GIT_SHA
SDK_SHORT_SHA = SDK_SHA[:7]
BUILDKIT_IMAGE = "moby/buildkit:v0.32.2"
DOCKERHUB_SECRET_NAME = "dockerhub-adityasoni8"
LOCAL_TEST_IMAGE = "local/swesmith-modal-build-test"
REGISTRY_IMAGE = EVAL_AGENT_SERVER_IMAGE
BUILD_TIMEOUT_SECONDS = 45 * 60
SANDBOX_TIMEOUT_SECONDS = 60 * 60


@dataclass(frozen=True)
class BuildRequest:
    base_image: str
    custom_tag: str
    destination_image: str
    push: bool = False


@dataclass
class BuildResult:
    method: Literal["vm-sandbox"]
    request: BuildRequest
    started_at: str
    duration_seconds: float
    success: bool
    tags: list[str]
    error: str | None = None
    docker_info: str | None = None
    sandbox_id: str | None = None


@dataclass(frozen=True)
class SandboxBuildConfig:
    cpu: float = 2.0
    memory: int = 8192
    build_timeout_seconds: int = BUILD_TIMEOUT_SECONDS
    sandbox_timeout_seconds: int = SANDBOX_TIMEOUT_SECONDS


def custom_tag_for_base_image(base_image: str) -> str:
    return base_image.rsplit("/", 1)[-1].split(":", 1)[0]


def build_requests(
    base_images: Iterable[str],
    *,
    push: bool = False,
    destination_image: str | None = None,
) -> list[BuildRequest]:
    destination = destination_image or (REGISTRY_IMAGE if push else LOCAL_TEST_IMAGE)
    return [
        BuildRequest(
            base_image=base_image,
            custom_tag=custom_tag_for_base_image(base_image),
            destination_image=destination,
            push=push,
        )
        for base_image in base_images
    ]


def easy_base_images() -> list[str]:
    """Use the exact training subset, deduplicating shared repository images."""
    train_data, _ = load_data()
    images = {str(task.misc["image_name"]).lower().strip() for task in train_data.values()}
    return sorted(name if name.startswith("docker.io/") else f"docker.io/{name}" for name in images)


app = modal.App(APP_NAME)
registry_secret = modal.Secret.from_name(DOCKERHUB_SECRET_NAME)
builder_image = (
    modal.Image.from_registry("ubuntu:24.04", add_python="3.12")
    .entrypoint([])
    .env(
        {
            "DEBIAN_FRONTEND": "noninteractive",
            "OPENHANDS_SUPPRESS_BANNER": "1",
            "UV_LINK_MODE": "copy",
        }
    )
    .apt_install(
        "build-essential",
        "ca-certificates",
        "curl",
        "docker-buildx",
        "docker.io",
        "git",
        "make",
    )
    .run_commands(
        "curl -LsSf https://astral.sh/uv/0.11.8/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh",
        (
            f"git clone {BENCHMARKS_REPOSITORY} /opt/benchmarks "
            f"&& cd /opt/benchmarks && git checkout {BENCHMARKS_SHA} "
            "&& git submodule update --init --recursive "
            f'&& test "$(git -C vendor/software-agent-sdk rev-parse HEAD)" = {SDK_SHA} '
            "&& uv sync --dev --frozen"
        ),
    )
    .workdir("/opt/benchmarks")
)


WORKER_CODE = """
import json
import os
import sys

from benchmarks.utils.build_utils import build_image

request = json.loads(sys.argv[1])
os.environ["BUILDKIT_PROGRESS"] = "plain"
os.environ["BUILDKIT_RESET_ON_FAILURE"] = "1"
os.environ["EXTENSIONS_REF"] = "main"
os.environ["OPENHANDS_BUILDKIT_CACHE_MODE"] = "off"
if request["push"]:
    os.environ["OPENHANDS_IMAGE_COMPRESSION"] = "estargz"

output = build_image(
    base_image=request["base_image"],
    target_image=request["destination_image"],
    custom_tag=request["custom_tag"],
    target="source-minimal",
    push=request["push"],
    force_build=True,
)
print(json.dumps({"tags": output.tags, "error": output.error}, sort_keys=True))
raise SystemExit(1 if output.error or not output.tags else 0)
"""


def _stream_process(
    process: ContainerProcess[str],
) -> tuple[str, str]:
    output: dict[str, list[str]] = {"stdout": [], "stderr": []}

    def consume(name: str) -> None:
        stream = process.stdout if name == "stdout" else process.stderr
        for line in stream:
            output[name].append(line)
            print(line, end="", file=sys.stdout if name == "stdout" else sys.stderr)

    stdout_thread = threading.Thread(target=consume, args=("stdout",))
    stderr_thread = threading.Thread(target=consume, args=("stderr",))
    stdout_thread.start()
    stderr_thread.start()
    process.wait()
    stdout_thread.join()
    stderr_thread.join()
    return "".join(output["stdout"]), "".join(output["stderr"])


def _build_in_vm_sandbox(request: BuildRequest, config: SandboxBuildConfig) -> dict[str, Any]:
    started_at = datetime.now(UTC).isoformat()
    started = time.monotonic()
    sandbox: modal.Sandbox | None = None
    sandbox_id: str | None = None
    tags: list[str] = []
    error: str | None = None
    docker_info: str | None = None
    try:
        sandbox = modal.Sandbox.create(
            "/usr/bin/dockerd",
            "--host=unix:///var/run/docker.sock",
            "--log-level=error",
            timeout=config.sandbox_timeout_seconds,
            app=app,
            image=builder_image,
            secrets=[registry_secret],
            cpu=config.cpu,
            memory=config.memory,
            experimental_options={"vm_runtime": True},
            tags={"method": "vm-sandbox", "sdk": SDK_SHORT_SHA},
        )
        sandbox_id = sandbox.object_id
        setup = sandbox.exec(
            "bash",
            "-lc",
            (
                "set -euo pipefail; "
                "for i in $(seq 1 120); do "
                "docker info >/dev/null 2>&1 && break; sleep 1; done; "
                "docker info --format 'storage_driver={{.Driver}}'; "
                "printf '%s' \"$REGISTRY_PASSWORD\" | docker login "
                '--username "$REGISTRY_USERNAME" --password-stdin >/dev/null; '
                "docker buildx create --name swesmith-modal-sdk "
                "--driver docker-container "
                f"--driver-opt image={BUILDKIT_IMAGE} --use; "
                "docker buildx inspect --bootstrap"
            ),
            timeout=180,
        )
        setup_stdout, setup_stderr = _stream_process(setup)
        if setup.returncode != 0:
            raise RuntimeError(
                f"Docker setup failed: {setup_stderr[-4000:] or setup_stdout[-4000:]}"
            )
        docker_info = setup_stdout

        process = sandbox.exec(
            "/opt/benchmarks/.venv/bin/python",
            "-c",
            WORKER_CODE,
            json.dumps(asdict(request)),
            workdir="/opt/benchmarks",
            timeout=config.build_timeout_seconds,
            env={
                "OPENHANDS_SUPPRESS_BANNER": "1",
                "UV_LINK_MODE": "copy",
            },
        )
        stdout, stderr = _stream_process(process)
        result_line = next(
            (line for line in reversed(stdout.splitlines()) if line.startswith("{")),
            None,
        )
        if result_line is not None:
            payload = json.loads(result_line)
            tags = list(payload.get("tags", []))
            error = payload.get("error")
        if process.returncode != 0 and error is None:
            error = stderr[-4000:] or stdout[-4000:] or "Build process failed"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if sandbox is not None:
            try:
                sandbox.terminate(wait=True)
            finally:
                sandbox.detach()

    result = BuildResult(
        method="vm-sandbox",
        request=request,
        started_at=started_at,
        duration_seconds=round(time.monotonic() - started, 3),
        success=error is None and bool(tags),
        tags=tags,
        error=error,
        docker_info=docker_info,
        sandbox_id=sandbox_id,
    )
    return asdict(result)


def _write_results(method: str, results: list[dict[str, Any]]) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(".agent_tmp") / f"modal-build-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{method}.json"
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {method} results to {output_path}")
    return output_path


def run_vm_builds(
    requests: list[BuildRequest],
    *,
    max_workers: int | None = None,
    config: SandboxBuildConfig | None = None,
) -> list[dict[str, Any]]:
    if not requests:
        return []
    if max_workers is not None and max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    worker_count = min(max_workers or len(requests), len(requests))
    sandbox_config = config or SandboxBuildConfig()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_build_in_vm_sandbox, request, sandbox_config): request
            for request in requests
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"VM build completed success={result['success']} base={futures[future].base_image}"
            )
    results.sort(key=lambda result: str(result["request"]))
    _write_results("vm-sandbox", results)
    return results


def _print_summary(results: list[dict[str, Any]]) -> None:
    successful = sum(bool(result["success"]) for result in results)
    durations = [float(result["duration_seconds"]) for result in results]
    print(
        json.dumps(
            {
                "total": len(results),
                "successful": successful,
                "failed": len(results) - successful,
                "wall_clock_proxy_seconds": max(durations, default=0),
                "cumulative_seconds": round(sum(durations), 3),
            },
            indent=2,
        )
    )


@app.local_entrypoint()
def main(max_workers: int = 8) -> None:
    results = run_vm_builds(build_requests(easy_base_images(), push=True), max_workers=max_workers)
    _print_summary(results)
    if any(not result["success"] for result in results):
        raise RuntimeError("Some easy-subset images failed; see the saved build results")


if __name__ == "__main__":
    raise SystemExit("Run with `modal run scripts/build_agent_images.py`.")
