"""FastAPI implementation of the in-process Tinker HTTP proxy.

This is an optional alternative to ``http_proxy.py`` when FastAPI/uvicorn are
available. It keeps the same interaction-storage model: the trainer process owns
the proxy, Apptainer calls it over HTTP, and rollout code pops interactions from
shared Python state.
"""

from __future__ import annotations

import logging
import socket
import threading
from collections import defaultdict
from typing import Any

import uvicorn
from fastapi import FastAPI, Header, HTTPException

from platoon.train.tinker.proxy import TinkerLLM, TinkerLLMInteraction, TinkerLLMProxySession

logger = logging.getLogger(__name__)

SESSION_HEADER = "X-Platoon-Tinker-Session"


class FastAPITinkerProxyState:
    def __init__(self, tinker_llm: TinkerLLM):
        self.tinker_llm = tinker_llm
        self._lock = threading.RLock()
        self._interactions_by_session: defaultdict[str, dict[str, TinkerLLMInteraction]] = defaultdict(dict)

    async def chat_completion(self, payload: dict[str, Any], session_id: str) -> dict[str, Any]:
        messages = payload.get("messages")
        if messages is None:
            raise HTTPException(status_code=400, detail="Missing required field: messages")
        if payload.get("stream"):
            raise HTTPException(status_code=400, detail="Streaming is not supported by this proxy")

        optional_params = {k: v for k, v in payload.items() if k not in {"model", "messages", "stream"}}

        async with TinkerLLMProxySession() as proxy_session:
            response = await self.tinker_llm.acompletion(messages=messages, optional_params=optional_params)

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


def create_tinker_proxy_app(state: FastAPITinkerProxyState) -> FastAPI:
    app = FastAPI(title="Platoon Tinker Proxy")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    @app.get("/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": "platoon-tinker", "object": "model"}]}

    @app.post("/v1/chat/completions")
    @app.post("/chat/completions")
    async def chat_completions(
        payload: dict[str, Any],
        x_platoon_tinker_session: str | None = Header(default=None, alias=SESSION_HEADER),
    ) -> dict[str, Any]:
        if not x_platoon_tinker_session:
            raise HTTPException(status_code=400, detail=f"Missing required header: {SESSION_HEADER}")
        return await state.chat_completion(payload, x_platoon_tinker_session)

    return app


def _find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


class FastAPITinkerHTTPProxyServer:
    def __init__(self, tinker_llm: TinkerLLM, host: str = "127.0.0.1", port: int | None = None):
        self.host = host
        self.port = port if port is not None else _find_free_port(host)
        self.state = FastAPITinkerProxyState(tinker_llm)
        self.app = create_tinker_proxy_app(self.state)
        self._server: uvicorn.Server | None = None
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
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="warning", workers=1)
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        logger.info("Started FastAPI Tinker proxy at %s", self.base_url)

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._server = None
        self._thread = None
        logger.info("Stopped FastAPI Tinker proxy")

    def pop_interactions(self, session_id: str) -> dict[str, TinkerLLMInteraction]:
        return self.state.pop_interactions(session_id)

    def discard_session(self, session_id: str) -> None:
        self.state.discard_session(session_id)

