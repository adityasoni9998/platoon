"""Subprocess rollouts using the trainer-owned Tinker proxy."""

import asyncio
import pickle
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

from platoon.train.tinker.fastapi_litellm_proxy import SESSION_HEADER
from platoon.utils.rollout_forkserver import ForkedProcess, RolloutForkServer


class TinkerProxyRollout:
    """Run each rollout in a fresh process; keep the shared proxy in the parent."""

    def __init__(self, rollout_fn, proxy):
        self.rollout_fn = rollout_fn
        self.proxy = proxy
        self._forkserver = RolloutForkServer(
            f"{__package__}.subprocess_worker", rollout_fn.__module__,
        )

    async def __call__(self, task, config):
        # Serialization gives the child its own task, without reloading datasets.
        config = deepcopy(config)
        session_id = str(uuid4())
        config.extra["llm_extra_headers"] = {
            **config.extra.get("llm_extra_headers", {}),
            SESSION_HEADER: session_id,
        }
        config.extra["max_input_tokens"] = self.proxy.context_window_length
        config.return_dict = True
        try:
            result = await self._run_subprocess(task, config)
            result["_tinker_interactions"] = self.proxy.pop_interactions(session_id)
            return result
        finally:
            self.proxy.discard_session(session_id)

    async def _run_subprocess(self, task, config):
        # These pickle files are private, local parent/child IPC. Logs keep using
        # inherited stdout/stderr, so SDK output cannot corrupt the result.
        workspace = config.extra.get("workspace", {})
        scratch_dir = None
        if workspace.get("type") == "apptainer":
            scratch_dir = workspace.get("apptainer", {}).get("scratch_dir")
            if not scratch_dir:
                raise ValueError("Apptainer scratch_dir is required for rollout IPC files")
        with tempfile.TemporaryDirectory(prefix="platoon-rollout-", dir=scratch_dir) as directory:
            request = Path(directory) / "request.pkl"
            response = Path(directory) / "response.pkl"
            with request.open("wb") as stream:
                pickle.dump((task, config), stream)
            startup = asyncio.create_task(self._launch_worker(request, response, config))
            try:
                # Shield launch so cancellation cannot lose an already-created child.
                process = await asyncio.shield(startup)
                # Bound the whole child lifetime, including interpreter shutdown.
                # Workspace setup can retry three times before the episode starts;
                # allow two extra hours for imports, those retries, and cleanup.
                timeout = config.extra.get("subprocess_timeout", (config.timeout or 1230) + 7200)
                try:
                    await asyncio.wait_for(process.wait(), timeout=timeout)
                except asyncio.TimeoutError as error:
                    raise TimeoutError(
                        f"Rollout subprocess for task {task.id} exceeded {timeout}s"
                    ) from error
                if process.returncode:
                    raise RuntimeError(f"Rollout subprocess exited with code {process.returncode}")
                with response.open("rb") as stream:
                    result = pickle.load(stream)
                if not isinstance(result, dict):
                    raise TypeError("Rollout subprocess must return a trajectory dictionary")
                return result
            finally:
                # Repeated cancellation must not release the proxy/session or
                # delete IPC files while the child is still using them.
                cleanup = asyncio.create_task(_stop_child(startup))
                cancelled = False
                while not cleanup.done():
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        cancelled = True
                cleanup.result()
                if cancelled:
                    raise asyncio.CancelledError

    async def _launch_worker(self, request: Path, response: Path, config):
        method = config.extra.get("worker_start_method", "exec")
        if method not in {"exec", "forkserver"}:
            raise ValueError(f"Unknown worker_start_method: {method!r}")
        args = [self.rollout_fn.__module__, self.rollout_fn.__name__, str(request), str(response)]
        if method == "forkserver":
            process = await self._forkserver.spawn(args, config.output_dir)
            if process is not None:
                return process
        return await asyncio.create_subprocess_exec(
            sys.executable,
            str(Path(__file__).with_name("subprocess_worker.py")),
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )

    async def aclose(self):
        await self._forkserver.aclose()


CHILD_CLEANUP_TIMEOUT = 180


async def _stop_child(startup):
    process = await startup
    try:
        if process.returncode is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), timeout=CHILD_CLEANUP_TIMEOUT)
            except asyncio.TimeoutError:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()
    finally:
        if isinstance(process, ForkedProcess):
            process.close()
