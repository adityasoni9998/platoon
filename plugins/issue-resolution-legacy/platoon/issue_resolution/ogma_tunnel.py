"""Reverse SSH tunnels from public Ogma ports to AReaL proxy workers."""

from __future__ import annotations

import atexit
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from urllib.request import urlopen

logger = logging.getLogger(__name__)

OGMA_SSH_TARGET = "adityabs@ogma.lti.cs.cmu.edu"
OGMA_PUBLIC_HOST = "ogma.lti.cs.cmu.edu"
OGMA_REMOTE_BIND_ADDRESS = "0.0.0.0"


class OgmaReverseTunnelManager:
    """Own one SSH connection carrying AReaL reverse proxy forwards."""

    def __init__(self, proxy_urls: list[str]) -> None:
        if not proxy_urls:
            raise ValueError("Expected at least one AReaL proxy URL")
        self.proxy_urls = list(proxy_urls)
        self.endpoint_map: dict[str, str] = {}
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._control_socket: Path | None = None
        self._master: subprocess.Popen | None = None
        self._log_handle = None
        self._closed = False

    @staticmethod
    def _target_host_port(proxy_url: str) -> tuple[str, int]:
        parsed = urlsplit(proxy_url)
        if parsed.scheme != "http" or parsed.hostname is None or parsed.port is None:
            raise ValueError(f"Invalid AReaL HTTP proxy URL: {proxy_url!r}")
        host = parsed.hostname
        if ":" in host:
            host = f"[{host}]"
        return host, parsed.port

    @staticmethod
    def _public_url(proxy_url: str, remote_port: int) -> str:
        parsed = urlsplit(proxy_url)
        return urlunsplit(
            (
                "http",
                f"{OGMA_PUBLIC_HOST}:{remote_port}",
                parsed.path,
                "",
                "",
            )
        ).rstrip("/")

    def _control_command(self, *args: str) -> list[str]:
        if self._control_socket is None:
            raise RuntimeError("SSH control socket is not initialized")
        return [
            "ssh",
            "-S",
            str(self._control_socket),
            *args,
            OGMA_SSH_TARGET,
        ]

    def _start_master(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory(prefix="issue-resolution-ogma-")
        temp_path = Path(self._temp_dir.name)
        self._control_socket = temp_path / "control.sock"
        self._log_handle = (temp_path / "ssh.log").open("w+")
        command = [
            "ssh",
            "-M",
            "-S",
            str(self._control_socket),
            "-N",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "ConnectionAttempts=1",
            "-o",
            "ControlPersist=no",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            OGMA_SSH_TARGET,
        ]
        self._master = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=self._log_handle,
            text=True,
        )

        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if self._master.poll() is not None:
                self._log_handle.seek(0)
                raise RuntimeError(
                    "Ogma SSH ControlMaster exited during startup: "
                    f"{self._log_handle.read().strip()}"
                )
            check = subprocess.run(
                self._control_command("-O", "check"),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
                check=False,
            )
            if check.returncode == 0:
                return
            time.sleep(0.1)
        raise TimeoutError("Timed out waiting for the Ogma SSH ControlMaster")

    def _add_forward(self, proxy_url: str) -> str:
        target_host, target_port = self._target_host_port(proxy_url)
        remote_forward = f"{OGMA_REMOTE_BIND_ADDRESS}:0:{target_host}:{target_port}"
        result = subprocess.run(
            self._control_command("-O", "forward", "-R", remote_forward),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        allocated = result.stdout.strip()
        if not allocated.isdigit():
            raise RuntimeError(
                "SSH did not report an allocated Ogma port for "
                f"{proxy_url}: stdout={result.stdout!r}, stderr={result.stderr!r}"
            )
        public_url = self._public_url(proxy_url, int(allocated))
        logger.info("Forwarded AReaL proxy %s through %s", proxy_url, public_url)
        return public_url

    @staticmethod
    def _wait_for_health(public_url: str) -> None:
        health_url = f"{public_url.rstrip('/')}/health"
        deadline = time.monotonic() + 30
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                with urlopen(health_url, timeout=3) as response:
                    if 200 <= response.status < 300:
                        return
            except Exception as error:
                last_error = error
            time.sleep(0.5)
        raise RuntimeError(f"Ogma tunnel health check failed for {health_url}: {last_error}")

    def start(self) -> dict[str, str]:
        if self._master is not None:
            return dict(self.endpoint_map)
        try:
            self._start_master()
            atexit.register(self.close)
            for proxy_url in self.proxy_urls:
                key = proxy_url.rstrip("/")
                public_url = self._add_forward(proxy_url)
                self._wait_for_health(public_url)
                self.endpoint_map[key] = public_url
            return dict(self.endpoint_map)
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        master = self._master
        if master is not None and master.poll() is None:
            try:
                subprocess.run(
                    self._control_command("-O", "exit"),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=10,
                    check=False,
                )
                master.wait(timeout=10)
            except (subprocess.TimeoutExpired, OSError):
                master.terminate()
                try:
                    master.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    master.kill()
                    master.wait(timeout=5)
        self._master = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None
        try:
            atexit.unregister(self.close)
        except Exception:
            pass

    def __enter__(self) -> "OgmaReverseTunnelManager":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
