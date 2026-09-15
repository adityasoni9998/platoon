"""One owned SSH connection, with reverse forwards added through its control socket."""

import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit


def target_host_port(proxy_url: str) -> tuple[str, int]:
    parsed = urlsplit(proxy_url)
    if parsed.scheme != "http" or parsed.hostname is None or parsed.port is None:
        raise ValueError(f"Invalid HTTP proxy URL: {proxy_url!r}")
    host = parsed.hostname
    return (f"[{host}]" if ":" in host else host), parsed.port


@contextmanager
def ssh_connection(target: str, options: tuple[str, ...] = ()):
    """Yield forward(url, port=0); port zero asks SSH to allocate a public port.

    The process, log, and control socket have the same lifetime. This scope also
    cleans up if startup, forwarding, or the caller is interrupted.
    """
    with tempfile.TemporaryDirectory(prefix="platoon-ssh-") as directory:
        control = ["ssh", "-S", str(Path(directory) / "control.sock"), *options]

        def command(*args):
            return subprocess.run(
                [*control, *args, target],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=15,
                check=True,
            )

        def forward(proxy_url: str, port: int = 0) -> int:
            host, local_port = target_host_port(proxy_url)
            result = command("-O", "forward", "-R", f"0.0.0.0:{port}:{host}:{local_port}")
            if port:
                return port
            allocated = result.stdout.strip()
            if not allocated.isdigit():
                raise RuntimeError(f"SSH did not report an allocated port: {result.stdout!r}")
            return int(allocated)

        with tempfile.TemporaryFile(mode="w+") as log:
            settings = (
                "BatchMode=yes",
                "ConnectTimeout=15",
                "ConnectionAttempts=1",
                "ControlPersist=no",
                "ExitOnForwardFailure=yes",
                "ChannelTimeout=forwarded-tcpip=6m",
                "ServerAliveInterval=30",
                "ServerAliveCountMax=3",
            )
            master = subprocess.Popen(
                [
                    *control,
                    "-M",
                    "-N",
                    "-T",
                    *(arg for setting in settings for arg in ("-o", setting)),
                    target,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=log,
            )
            try:
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline:
                    if master.poll() is not None:
                        log.seek(0)
                        raise RuntimeError(f"SSH to {target} exited during startup: {log.read()}")
                    try:
                        command("-O", "check")
                        break
                    except subprocess.CalledProcessError:
                        time.sleep(0.1)
                else:
                    raise TimeoutError(f"Timed out waiting for SSH to {target}")
                yield forward
            finally:
                if master.poll() is None:
                    master.terminate()
                    try:
                        master.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        master.kill()
                        master.wait()
