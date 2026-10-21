"""A small Modal sandbox relaying public HTTPS traffic over reverse SSH."""

from __future__ import annotations

import logging
import shlex
import shutil
import socket
import ssl
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

from platoon.inference.inference_tunnel import ModalRouterConfig
from platoon.inference.ssh_tunnel import ssh_connection

logger = logging.getLogger(__name__)
SSH_PORT = 2222
FIRST_PROXY_PORT = 10000

# Keys are injected at sandbox creation, never baked into the shared image.
ROUTER_BOOTSTRAP = r'''
import os
from pathlib import Path

Path("/run/sshd").mkdir(parents=True, exist_ok=True)
Path("/root/.ssh").mkdir(mode=0o700, exist_ok=True)
key = Path("/etc/ssh/router_host_key")
key.write_text(os.environ.pop("ROUTER_HOST_KEY"))
key.chmod(0o600)
authorized = Path("/root/.ssh/authorized_keys")
authorized.write_text(os.environ.pop("ROUTER_CLIENT_KEY") + "\n")
authorized.chmod(0o600)
config = Path("/etc/ssh/router_sshd_config")
config.write_text("""
Port 2222
ListenAddress 0.0.0.0
HostKey /etc/ssh/router_host_key
PidFile /run/sshd/router.pid
AuthorizedKeysFile /root/.ssh/authorized_keys
PermitRootLogin prohibit-password
PasswordAuthentication no
KbdInteractiveAuthentication no
UsePAM no
AllowTcpForwarding remote
GatewayPorts clientspecified
AllowAgentForwarding no
X11Forwarding no
PermitTTY no
MaxSessions 0
ClientAliveInterval 30
ClientAliveCountMax 3
""")
os.execv("/usr/sbin/sshd", ["/usr/sbin/sshd", "-D", "-e", "-f", str(config)])
'''


def wait_for_ssh(sandbox, host: str, port: int, *, tls: bool = False) -> None:
    # Modal tunnel allocation can precede sshd readiness.
    deadline = time.monotonic() + 120
    context = ssl.create_default_context() if tls else None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=3) as connection:
                if context is None:
                    banner = connection.recv(256)
                else:
                    with context.wrap_socket(connection, server_hostname=host) as secured:
                        banner = secured.recv(256)
                if banner.startswith(b"SSH-"):
                    return
        except OSError:
            pass
        if sandbox.poll() is not None:
            raise RuntimeError("Modal router exited before SSH became ready")
        time.sleep(0.5)
    raise TimeoutError("Timed out waiting for Modal router SSH readiness")


@contextmanager
def modal_tunnels(proxy_urls: list[str], config: ModalRouterConfig):
    """Own a Modal router, its temporary keys, and one reverse SSH connection."""
    import modal

    if config.ssh_over_tls and shutil.which("openssl") is None:
        raise RuntimeError("Modal SSH-over-TLS requires openssl on the client node")

    with tempfile.TemporaryDirectory(prefix="platoon-modal-router-") as directory:
        key_dir = Path(directory)
        for name in ("client", "host"):
            subprocess.run(
                ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key_dir / name)],
                check=True,
                capture_output=True,
                timeout=15,
            )
        known_hosts = key_dir / "known_hosts"
        known_hosts.write_text("platoon-modal-router " + (key_dir / "host.pub").read_text())
        image = modal.Image.debian_slim(python_version="3.12").apt_install("openssh-server")
        app = modal.App.lookup(config.app_name, create_if_missing=True)
        ports = [FIRST_PROXY_PORT + i for i in range(len(proxy_urls))]
        sandbox = modal.Sandbox.create(
            "python",
            "-c",
            ROUTER_BOOTSTRAP,
            app=app,
            image=image,
            cpu=config.cpu,
            memory=config.memory,
            timeout=config.timeout,
            encrypted_ports=ports + ([SSH_PORT] if config.ssh_over_tls else []),
            unencrypted_ports=[] if config.ssh_over_tls else [SSH_PORT],
            secrets=[
                modal.Secret.from_dict(
                    {
                        "ROUTER_HOST_KEY": (key_dir / "host").read_text(),
                        "ROUTER_CLIENT_KEY": (key_dir / "client.pub").read_text(),
                    }
                )
            ],
            readiness_probe=modal.Probe.with_tcp(SSH_PORT),
            tags={"purpose": "platoon-inference-router"},
        )
        try:
            logger.info("Created Modal inference router %s", sandbox.object_id)
            tunnels = sandbox.tunnels(timeout=120)
            ssh_tunnel = tunnels[SSH_PORT]
            if config.ssh_over_tls:
                host, port = ssh_tunnel.host, ssh_tunnel.port
                wait_for_ssh(sandbox, host, port, tls=True)
            else:
                host, port = ssh_tunnel.tcp_socket
                wait_for_ssh(sandbox, host, port)
            options = (
                "-p",
                str(port),
                "-i",
                str(key_dir / "client"),
                "-o",
                "IdentitiesOnly=yes",
                "-o",
                "StrictHostKeyChecking=yes",
                "-o",
                f"UserKnownHostsFile={known_hosts}",
                "-o",
                "HostKeyAlias=platoon-modal-router",
            )
            if config.ssh_over_tls:
                proxy = shlex.join(
                    [
                        "openssl", "s_client", "-quiet", "-no_ign_eof",
                        # Never interpret SSH ciphertext as interactive s_client commands.
                        "-nocommands", "-connect", f"{host}:{port}",
                        "-servername", host, "-verify_return_error", "-verify_hostname", host,
                    ]
                )
                options += ("-o", f"ProxyCommand={proxy}")
            with ssh_connection(f"root@{host}", options) as forward:
                endpoints = {}
                for url, port in zip(proxy_urls, ports, strict=True):
                    forward(url, port)
                    endpoints[url] = tunnels[port].url.rstrip("/") + urlsplit(url).path
                yield endpoints
        finally:
            sandbox.terminate()
