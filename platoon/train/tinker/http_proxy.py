"""In-process HTTP proxy for routing Apptainer LLM calls to TinkerLLM."""

from __future__ import annotations

import json
import logging
import socket
import threading
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from platoon.train.tinker.proxy import TinkerLLM, TinkerLLMInteraction, TinkerLLMProxySession

logger = logging.getLogger(__name__)

SESSION_HEADER = "X-Platoon-Tinker-Session"

_active_proxy: TinkerHTTPProxyServer | None = None


def get_active_tinker_http_proxy() -> TinkerHTTPProxyServer | None:
    return _active_proxy


def set_active_tinker_http_proxy(proxy: TinkerHTTPProxyServer | None) -> None:
    global _active_proxy
    _active_proxy = proxy


class TinkerProxyState:
    def __init__(self, tinker_llm: TinkerLLM):
        self.tinker_llm = tinker_llm
        self._lock = threading.RLock()
        self._interactions_by_session: defaultdict[str, dict[str, TinkerLLMInteraction]] = defaultdict(dict)

    def chat_completion(self, payload: dict[str, Any], session_id: str) -> dict[str, Any]:
        messages = payload.get("messages")
        if messages is None:
            raise ValueError("Missing required field: messages")

        optional_params = {k: v for k, v in payload.items() if k not in {"model", "messages"}}
        optional_params.pop("stream", None)

        with TinkerLLMProxySession() as proxy_session:
            response = self.tinker_llm.completion(messages=messages, optional_params=optional_params)

        with self._lock:
            self._interactions_by_session[session_id].update(proxy_session.interactions)

        if hasattr(response, "model_dump"):
            response_data = response.model_dump(mode="json", exclude_none=True)
        else:
            response_data = dict(response)
        response_data.setdefault("object", "chat.completion")
        return response_data

    def pop_interactions(self, session_id: str) -> dict[str, TinkerLLMInteraction]:
        with self._lock:
            return self._interactions_by_session.pop(session_id, {})

    def discard_session(self, session_id: str) -> None:
        with self._lock:
            self._interactions_by_session.pop(session_id, None)


class _TinkerProxyHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], state: TinkerProxyState):
        super().__init__(server_address, _TinkerProxyRequestHandler)
        self.state = state


class _TinkerProxyRequestHandler(BaseHTTPRequestHandler):
    server: _TinkerProxyHTTPServer

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug("Tinker HTTP proxy: " + format, *args)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        if self.path in {"/v1/models", "/models"}:
            self._send_json(200, {"object": "list", "data": [{"id": "platoon-tinker", "object": "model"}]})
            return
        self._send_json(404, {"error": {"message": f"Unknown path: {self.path}"}})

    def do_POST(self) -> None:
        if self.path not in {"/v1/chat/completions", "/chat/completions"}:
            self._send_json(404, {"error": {"message": f"Unknown path: {self.path}"}})
            return

        session_id = self.headers.get(SESSION_HEADER)
        if not session_id:
            self._send_json(400, {"error": {"message": f"Missing required header: {SESSION_HEADER}"}})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            response = self.server.state.chat_completion(payload, session_id)
        except Exception as exc:
            logger.exception("Tinker HTTP proxy request failed")
            self._send_json(500, {"error": {"message": str(exc), "type": type(exc).__name__}})
            return

        self._send_json(200, response)


def _find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


class TinkerHTTPProxyServer:
    def __init__(self, tinker_llm: TinkerLLM, host: str = "127.0.0.1", port: int | None = None):
        self.host = host
        self.port = port if port is not None else _find_free_port(host)
        self.state = TinkerProxyState(tinker_llm)
        self._server: _TinkerProxyHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @property
    def model_name(self) -> str:
        return "openai/platoon-tinker"

    @property
    def api_key(self) -> str:
        return "sk-xxx"

    @property
    def context_window_length(self) -> int | None:
        return self.state.tinker_llm.context_window_length

    def start(self) -> None:
        if self._server is not None:
            return
        self._server = _TinkerProxyHTTPServer((self.host, self.port), self.state)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        set_active_tinker_http_proxy(self)
        logger.info("Started Tinker HTTP proxy at %s", self.base_url)

    def stop(self) -> None:
        if self._server is None:
            return
        try:
            self._server.shutdown()
            self._server.server_close()
        finally:
            if self._thread is not None:
                self._thread.join(timeout=5.0)
            self._server = None
            self._thread = None
            if get_active_tinker_http_proxy() is self:
                set_active_tinker_http_proxy(None)
            logger.info("Stopped Tinker HTTP proxy")

    def pop_interactions(self, session_id: str) -> dict[str, TinkerLLMInteraction]:
        return self.state.pop_interactions(session_id)

    def discard_session(self, session_id: str) -> None:
        self.state.discard_session(session_id)
