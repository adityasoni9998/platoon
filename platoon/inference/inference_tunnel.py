"""Select or disable the inference relay independently of the training backend."""

import logging
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from urllib.parse import urlsplit
from urllib.request import urlopen

from platoon.inference.ssh_tunnel import ssh_connection, target_host_port

logger = logging.getLogger(__name__)
OGMA_HOST = "ogma.lti.cs.cmu.edu"


@dataclass
class ModalRouterConfig:
    app_name: str = "platoon-inference-router"
    cpu: float = 0.125
    memory: int = 128
    timeout: int = 86400


@dataclass
class InferenceTunnelConfig:
    backend: str = "ogma"
    modal: ModalRouterConfig = field(default_factory=ModalRouterConfig)


def wait_for_health(public_url: str) -> None:
    health_url = f"{public_url.rstrip('/')}/health"
    deadline = time.monotonic() + 30
    last_error = None
    while time.monotonic() < deadline:
        try:
            with urlopen(health_url, timeout=3) as response:
                if 200 <= response.status < 300:
                    return
        except OSError as error:
            last_error = error
        time.sleep(0.5)
    raise RuntimeError(f"Tunnel health check failed for {health_url}: {last_error}")


@contextmanager
def inference_tunnels(proxy_urls: list[str], config: InferenceTunnelConfig | None = None):
    """Yield internal-to-public URLs, keeping the relay alive for the caller's scope."""
    config = config or InferenceTunnelConfig()
    urls = list(dict.fromkeys(url.rstrip("/") for url in proxy_urls))
    if not urls:
        raise ValueError("Expected at least one inference proxy URL")
    for url in urls:
        target_host_port(url)
    if config.backend == "none":
        yield {url: url for url in urls}
        return
    with ExitStack() as stack:
        if config.backend == "ogma":
            forward = stack.enter_context(ssh_connection(f"adityabs@{OGMA_HOST}"))
            endpoints = {url: f"http://{OGMA_HOST}:{forward(url)}{urlsplit(url).path}" for url in urls}
        elif config.backend == "modal":
            from platoon.inference.modal_tunnel import modal_tunnels

            endpoints = stack.enter_context(modal_tunnels(urls, config.modal))
        else:
            raise ValueError(
                f"Unknown inference_tunnel.backend {config.backend!r}; "
                "expected 'none', 'ogma' or 'modal'"
            )
        for internal, public in endpoints.items():
            wait_for_health(public)
            logger.info("Forwarded inference proxy %s through %s", internal, public)
        yield endpoints
