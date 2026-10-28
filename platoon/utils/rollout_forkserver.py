"""Small forkserver adapter for the plugins' existing single-use rollout workers."""

from __future__ import annotations

import asyncio
import ctypes
import logging
import os
import signal
import sys
from pathlib import Path

from platoon.utils.forkserver import ForkedWorker, ForkServer, ForkServerError

logger = logging.getLogger(__name__)


def initialize_child() -> None:
    """Request ordinary worker cleanup if its forkserver disappears."""
    parent = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), "Could not set worker parent-death signal")
    if parent == 1 or os.getppid() != parent:
        os.kill(os.getpid(), signal.SIGTERM)


class ForkedProcess:
    """The process operations used by the plugins' existing timeout/cleanup code."""

    def __init__(self, worker: ForkedWorker):
        self.worker = worker

    @property
    def pid(self) -> int:
        return self.worker.pid

    @property
    def returncode(self) -> int | None:
        return self.worker.returncode

    async def wait(self) -> int:
        code = await self.worker.wait()
        if code is None:
            raise RuntimeError("Rollout worker exited without a status after its forkserver disappeared")
        return code

    def terminate(self) -> None:
        signal.pidfd_send_signal(self.worker.pidfd, signal.SIGTERM)

    def kill(self) -> None:
        signal.pidfd_send_signal(self.worker.pidfd, signal.SIGKILL)

    def close(self) -> None:
        self.worker.close()


class RolloutForkServer:
    """One lazily initialized server per plugin adapter, with exec fallback."""

    def __init__(self, worker_module: str, rollout_module: str):
        self.worker_module = worker_module
        self.rollout_module = rollout_module
        self._server: ForkServer | None = None
        self._lock = asyncio.Lock()
        self._disabled = False

    async def spawn(self, args: list[str], output_dir: str) -> ForkedProcess | None:
        """Return None only when the caller should use its existing exec launcher."""
        async with self._lock:
            if self._disabled:
                return None
            if self._server is None:
                log_dir = Path(output_dir) / "workers"
                log_dir.mkdir(parents=True, exist_ok=True)
                env = os.environ.copy()
                # These apply to rollout workers only; sampling stays in the trainer.
                env.update(
                    CUDA_VISIBLE_DEVICES="",
                    PYTORCH_NVML_BASED_CUDA_CHECK="1",
                    TOKENIZERS_PARALLELISM="false",
                    TINKER_TELEMETRY="0",
                )
                for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                    env[name] = "1"
                env.pop("TINKER_SUBPROCESS_SAMPLING", None)
                try:
                    self._server = await ForkServer.start(
                        python=sys.executable,
                        module=self.worker_module,
                        preload=[self.rollout_module, "platoon.train.tinker.fastapi_litellm_proxy"],
                        child_init=["platoon.utils.rollout_forkserver:initialize_child"],
                        env=env,
                        cwd=os.getcwd(),
                        log_path=str(log_dir / "forkserver.log"),
                        timeout_s=600,
                    )
                except (ForkServerError, OSError) as exc:
                    self._disabled = True
                    logger.error("Forkserver did not start; using exec for this run: %s", exc)
                    return None
                logger.info("Rollout forkserver %s ready in %.1fs", self._server.pid, self._server.seconds)

        server = self._server
        log_dir = Path(output_dir) / "workers"
        log_dir.mkdir(parents=True, exist_ok=True)
        # request.pkl lives in a unique per-rollout temporary directory.
        log_path = log_dir / f"{Path(args[-2]).parent.name}.log"
        try:
            worker = await server.spawn(args, env=server.env, cwd=server.cwd, log_path=str(log_path))
        except ForkServerError as exc:
            if server.alive:
                logger.error("Forkserver could not start worker; using exec for this rollout only: %s", exc)
            else:
                self._disabled = True
                logger.error("Forkserver exited; using exec for the rest of this run: %s", exc)
            return None
        return ForkedProcess(worker)

    async def aclose(self) -> None:
        """Called after the training task group has finished cleaning up its workers."""
        if self._server is not None:
            await self._server.kill()
            self._server = None
