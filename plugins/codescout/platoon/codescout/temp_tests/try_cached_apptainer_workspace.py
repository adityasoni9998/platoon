#!/usr/bin/env python3
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import platform
import shutil
import tempfile
from pathlib import Path

from openhands.workspace import ApptainerWorkspace


EVAL_AGENT_SERVER_IMAGE = "docker.io/adityasoni8/eval-agent-server:5f106d0-custom-base-image_tag_latest-source"
DEFAULT_CACHE_DIR = "/data/user_data/adityabs/apptainer_cache"
DEFAULT_CMD = "pwd"
DEFAULT_COUNT = 256


def detect_platform() -> str:
    machine = platform.machine().lower()
    if "arm" in machine or "aarch64" in machine:
        return "linux/arm64"
    return "linux/amd64"


def make_workspace(cache_dir: Path, working_dir: Path) -> ApptainerWorkspace:
    working_dir.mkdir(parents=True, exist_ok=True)
    return ApptainerWorkspace(
        server_image=EVAL_AGENT_SERVER_IMAGE,
        working_dir=str(working_dir),
        platform=detect_platform(),
        cache_dir=str(cache_dir),
        detach_logs=True,
    )


def start_workspace(index: int, cache_dir: Path, root_dir: Path, cmd: str, timeout: float):
    workdir = root_dir / f"workspace-{index:04d}"
    workspace = make_workspace(cache_dir, workdir)
    try:
        result = workspace.execute_command(cmd, timeout=timeout)
        if result.exit_code != 0:
            raise RuntimeError(result.stderr)
    except Exception:
        workspace.cleanup()
        raise
    return index, workspace


def cleanup_one(workspace: ApptainerWorkspace) -> None:
    try:
        workspace.cleanup()
    except Exception as exc:
        print(f"cleanup failed: {type(exc).__name__}: {exc}", flush=True)


def cleanup(workspaces: list[ApptainerWorkspace], root_dir: Path | None) -> None:
    if workspaces:
        with ThreadPoolExecutor(max_workers=len(workspaces)) as executor:
            list(executor.map(cleanup_one, workspaces))
    if root_dir is not None:
        shutil.rmtree(root_dir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("APPTAINER_CACHEDIR", DEFAULT_CACHE_DIR),
    )
    parser.add_argument("-n", "--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--workdir-root", help="Root workspace directory; a temp dir is used if omitted")
    parser.add_argument("--cmd", default=DEFAULT_CMD)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    if args.count < 1:
        raise ValueError("--count must be at least 1")

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cleanup_root = args.workdir_root is None
    root_dir = Path(args.workdir_root).expanduser().resolve() if args.workdir_root else Path(
        tempfile.mkdtemp(prefix="codescout-apptainer-batch-")
    )
    root_dir.mkdir(parents=True, exist_ok=True)

    print(f"cache_dir={cache_dir}")
    print(f"server_image={EVAL_AGENT_SERVER_IMAGE}")
    print(f"workdir_root={root_dir}")
    print(f"count={args.count}")

    workspaces: list[ApptainerWorkspace] = []
    errors: list[str] = []
    try:
        with ThreadPoolExecutor(max_workers=args.count) as executor:
            futures = [
                executor.submit(start_workspace, i, cache_dir, root_dir, args.cmd, args.timeout)
                for i in range(args.count)
            ]
            for future in as_completed(futures):
                try:
                    index, workspace = future.result()
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")
                else:
                    workspaces.append(workspace)
                    print(f"started={index} alive={len(workspaces)}/{args.count}", flush=True)

        if errors:
            for error in errors:
                print(error, flush=True)
            return 1

        print(f"all_alive={len(workspaces)}")
        return 0
    finally:
        cleanup(workspaces, root_dir if cleanup_root else None)


if __name__ == "__main__":
    raise SystemExit(main())
