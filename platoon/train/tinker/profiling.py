"""Opt-in inference traces. Producers never wait for the writer or serialize JSON.

Enable in SWE-rebench with --profiling.enabled true and a fresh --stats.trial_name.
Files go to <run_log_path>/profiling/trace-<timestamp>-<pid>.jsonl. Use
scripts/profile_inference.py to summarize them or inspect one trace ID. Each
HTTP response also exposes X-Platoon-Trace-ID for matching client-side probes.

The writer owns JSON encoding and disk I/O. Producers use a bounded deque with
nonblocking lock acquisition and drop rather than wait. A 100 ms heartbeat
measures each loop separately. This has finite CPU/allocation overhead; it does
not promise zero interference. No request bodies, credentials, or token arrays
are recorded. All timing differences must be computed within a process.

SDK hooks are scoped to this session and restored on exit; installed packages
are not edited. Subprocess sampling is explicitly unsupported by these hooks.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import itertools
import json
import logging
import os
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)
_trace = contextvars.ContextVar("platoon_profile_trace", default=None)
_backend_request = contextvars.ContextVar("platoon_profile_backend_request", default=None)
_sample_submitted = contextvars.ContextVar("platoon_profile_sample_submitted", default=None)
_sdk_active = contextvars.ContextVar("platoon_profile_sdk_active", default=False)
_active: Profiler | None = None


@dataclass
class ProfilingConfig:
    enabled: bool = False
    buffer_events: int = 50_000
    heartbeat_interval_s: float = 0.1


class Profiler:
    def __init__(self, path: Path, config: ProfilingConfig):
        if config.buffer_events < 1 or config.heartbeat_interval_s < 0.01:
            raise ValueError("Profiling needs buffer_events >= 1 and heartbeat_interval_s >= 0.01")
        self.path = path
        self.config = config
        self.pending = deque()
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.closed = False
        self.dropped = 0
        self.written = 0
        self.writer_error = None
        self.ids = itertools.count()
        self.watches = {}
        self.patches = []
        self.active_requests = 0  # Updated only on the proxy event loop.
        self.writer = threading.Thread(target=self._write, name="platoon-profile-writer", daemon=True)
        self.writer.start()

    def emit(self, event, **fields):
        if self.closed:
            return
        record = {
            "event": event,
            "mono_s": time.perf_counter(),
            "wall_time_s": time.time(),
            "thread_id": threading.get_ident(),
            "trace_id": _trace.get(),
            "tinker_request_id": _backend_request.get(),
            **fields,
        }
        # Unlike Queue.put_nowait, this does not wait even for the queue's mutex.
        if not self.lock.acquire(blocking=False):
            self.dropped += 1
            return
        try:
            if len(self.pending) >= self.config.buffer_events:
                self.dropped += 1
            else:
                self.pending.append(record)
        finally:
            self.lock.release()

    def _write_batch(self, stream, batch):
        for record in batch:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.written += len(batch)

    def _write(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("x", buffering=256 * 1024) as stream:
                last_flush = time.monotonic()
                while True:
                    with self.lock:
                        batch = [self.pending.popleft() for _ in range(min(256, len(self.pending)))]
                    self._write_batch(stream, batch)
                    now = time.monotonic()
                    if now - last_flush >= 1:
                        stream.write(
                            json.dumps(
                                {
                                    "event": "writer_status",
                                    "wall_time_s": time.time(),
                                    "written": self.written,
                                    "dropped": self.dropped,
                                }
                            )
                            + "\n"
                        )
                        stream.flush()
                        last_flush = now
                    if self.stop.is_set() and not self.pending:
                        stream.write(
                            json.dumps({"event": "writer_closed", "written": self.written, "dropped": self.dropped})
                            + "\n"
                        )
                        break
                    if not batch:
                        self.stop.wait(0.05)
                    else:
                        # Yield the GIL between small batches, even with a full buffer.
                        time.sleep(0.001)
        except Exception as error:
            self.writer_error = type(error).__name__
            self.closed = True
            logger.exception("Profiling writer failed; training continues without trace output")

    @contextmanager
    def span(self, name, **fields):
        span_id = next(self.ids)
        start = time.perf_counter()
        self.emit("span_start", span=name, span_id=span_id, **fields)
        error_type = None
        try:
            yield span_id
        except BaseException as error:
            error_type = type(error).__name__
            raise
        finally:
            self.emit(
                "span_end", span=name, span_id=span_id, duration_s=time.perf_counter() - start, error_type=error_type
            )

    def watch_loop(self, label):
        loop = asyncio.get_running_loop()
        if loop in self.watches or self.closed:
            return
        interval = self.config.heartbeat_interval_s

        def schedule():
            expected = time.perf_counter() + interval
            self.watches[loop] = loop.call_later(interval, tick, expected, context=contextvars.Context())

        def tick(expected):
            if self.closed:
                return
            self.emit("loop_lag", loop=label, lag_s=max(0.0, time.perf_counter() - expected), trace_id=None)
            schedule()

        schedule()

    def patch(self, obj, name, wrapper):
        original = getattr(obj, name)
        replacement = wrapper(original)
        setattr(obj, name, replacement)
        self.patches.append((obj, name, original, replacement))

    def close(self):
        # Called after asyncio.run exits, never from a request handler.
        self.closed = True
        for loop, handle in list(self.watches.items()):
            if not loop.is_closed():
                loop.call_soon_threadsafe(handle.cancel)
        for obj, name, original, replacement in reversed(self.patches):
            if getattr(obj, name) is replacement:
                setattr(obj, name, original)
        self.stop.set()
        self.writer.join(timeout=5)
        if self.writer.is_alive():
            logger.warning("Profiling writer has not drained; shutdown is proceeding")
        if self.dropped:
            logger.warning("Profiling dropped %s events; see writer_status records", self.dropped)


def current():
    return _active


@contextmanager
def span(name, **fields):
    if (profiler := current()) is None:
        yield
    else:
        with profiler.span(name, **fields):
            yield


@contextmanager
def request_trace(trace_id=None):
    if current() is None:
        yield None
        return
    trace_id = trace_id or uuid.uuid4().hex
    token = _trace.set(trace_id)
    try:
        yield trace_id
    finally:
        _trace.reset(token)


async def to_thread(name, func, *args, **kwargs):
    profiler = current()
    if profiler is None:
        return await asyncio.to_thread(func, *args, **kwargs)
    submitted = time.perf_counter()

    def execute():
        profiler.emit("thread_queue", stage=name, wait_s=time.perf_counter() - submitted)
        with profiler.span(name + ".execute"):
            return func(*args, **kwargs)

    with profiler.span(name + ".await"):
        return await asyncio.to_thread(execute)


class ProxyTraceMiddleware:
    """Pure ASGI middleware: includes body receive, response encoding, and send."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        profiler = current()
        if profiler is None or scope["type"] != "http" or not scope["path"].endswith("chat/completions"):
            return await self.app(scope, receive, send)
        profiler.watch_loop("proxy")
        headers = dict(scope.get("headers", []))
        session_id = headers.get(b"x-platoon-tinker-session", b"").decode(errors="replace")[:128]
        with request_trace() as trace_id:
            profiler.active_requests += 1
            received = 0
            sent = 0
            profiler.emit("proxy_request", session_id=session_id, active_requests=profiler.active_requests)

            async def traced_receive():
                nonlocal received
                with profiler.span("proxy.receive_body"):
                    message = await receive()
                if message["type"] == "http.request":
                    received += len(message.get("body", b""))
                    if not message.get("more_body", False):
                        profiler.emit("proxy_body_received", bytes=received)
                return message

            async def traced_send(message):
                nonlocal sent
                if message["type"] == "http.response.start":
                    message = dict(message)
                    message["headers"] = [*message.get("headers", []), (b"x-platoon-trace-id", trace_id.encode())]
                    profiler.emit("proxy_response_headers", status=message["status"])
                elif message["type"] == "http.response.body":
                    sent += len(message.get("body", b""))
                with profiler.span("proxy.send"):
                    await send(message)

            try:
                with profiler.span("proxy.total"):
                    return await self.app(scope, traced_receive, traced_send)
            finally:
                profiler.active_requests -= 1
                profiler.emit(
                    "proxy_complete", received_bytes=received, sent_bytes=sent, active_requests=profiler.active_requests
                )


def _install_sdk_hooks(profiler):
    import httpx
    import tinker
    from tinker.lib.api_future_impl import _APIFuture
    from tinker.lib.internal_client_holder import BytesSemaphore, InternalClientHolder
    from tinker.lib.public_interfaces.sampling_client import SamplingClient

    if os.environ.get("TINKER_SUBPROCESS_SAMPLING", "").lower() in {"1", "true", "yes"}:
        raise ValueError(
            "Profiling currently requires TINKER_SUBPROCESS_SAMPLING unset/0; sidecar hooks are not installed"
        )

    def sample_wrapper(original):
        @functools.wraps(original)
        async def wrapped(self, *args, **kwargs):
            with request_trace(_trace.get()):
                submitted = _sample_submitted.set(time.perf_counter())
                try:
                    with profiler.span("sdk.sample"):
                        return await original(self, *args, **kwargs)
                finally:
                    _sample_submitted.reset(submitted)

        return wrapped

    def impl_wrapper(original):
        @functools.wraps(original)
        async def wrapped(self, *args, **kwargs):
            profiler.watch_loop("sdk")
            submitted = _sample_submitted.get()
            profiler.emit(
                "sdk_scheduled",
                wait_s=time.perf_counter() - submitted if submitted else None,
                sampling_session_id=self._sampling_session_id,
            )
            token = _sdk_active.set(True)
            try:
                with profiler.span("sdk.sample_impl"):
                    return await original(self, *args, **kwargs)
            finally:
                _sdk_active.reset(token)

        return wrapped

    def submit_wrapper(original):
        @functools.wraps(original)
        async def wrapped(self, *args, **kwargs):
            with profiler.span(
                "sdk.submit", sampling_session_id=self._sampling_session_id, seq_id=self._request_id_counter
            ):
                result = await original(self, *args, **kwargs)
                profiler.emit(
                    "sdk_ack", tinker_request_id=getattr(result, "request_id", None), backpressure=result is None
                )
                return result

        return wrapped

    def limit_wrapper(name):
        def wrap(original):
            @functools.wraps(original)
            @asynccontextmanager
            async def wrapped(self, *args, **kwargs):
                if not _sdk_active.get():
                    async with original(self, *args, **kwargs):
                        yield
                    return
                # End the wait span before executing the body of the acquired context.
                with contextmanager_wait(profiler, name) as acquired:
                    async with original(self, *args, **kwargs):
                        acquired()
                        yield

            return wrapped

        return wrap

    def future_wrapper(original):
        @functools.wraps(original)
        async def wrapped(self, *args, **kwargs):
            profiler.watch_loop("sdk")
            with request_trace(_trace.get()):
                request_token = _backend_request.set(self.request_id)
                sdk_token = _sdk_active.set(True)
                try:
                    with profiler.span("sdk.future", request_type=self.request_type):
                        return await original(self, *args, **kwargs)
                finally:
                    _sdk_active.reset(sdk_token)
                    _backend_request.reset(request_token)

        return wrapped

    def http_wrapper(original):
        @functools.wraps(original)
        async def wrapped(self, request, *args, **kwargs):
            if not _sdk_active.get():
                return await original(self, request, *args, **kwargs)
            # Do not parse/copy HTTP bodies or log credentials, headers, prompts, or tokens.
            old_trace = request.extensions.get("trace")
            with profiler.span("sdk.http", path=request.url.path, method=request.method) as http_span:

                async def trace(event, info):
                    profiler.emit("http_transport", http_span_id=http_span, transport_event=event)
                    if old_trace is not None:
                        await old_trace(event, info)

                request.extensions["trace"] = trace
                try:
                    response = await original(self, request, *args, **kwargs)
                    profiler.emit("http_response", http_span_id=http_span, status=response.status_code)
                    return response
                finally:
                    if old_trace is None:
                        request.extensions.pop("trace", None)
                    else:
                        request.extensions["trace"] = old_trace

        return wrapped

    profiler.patch(SamplingClient, "sample_async", sample_wrapper)
    profiler.patch(SamplingClient, "_sample_async_impl", impl_wrapper)
    profiler.patch(SamplingClient, "_send_asample_request", submit_wrapper)
    profiler.patch(InternalClientHolder, "sample_dispatch_rate_limit", limit_wrapper("sdk.dispatch_wait"))
    profiler.patch(BytesSemaphore, "acquire", limit_wrapper("sdk.bytes_wait"))
    profiler.patch(_APIFuture, "_result_async", future_wrapper)
    profiler.patch(httpx.AsyncClient, "send", http_wrapper)
    profiler.emit("sdk_hooks_installed", sdk_version=tinker.__version__)


@contextmanager
def contextmanager_wait(profiler, name):
    start = time.perf_counter()
    ended = False

    def acquired():
        nonlocal ended
        ended = True
        profiler.emit("sdk_limit_wait", stage=name, wait_s=time.perf_counter() - start, acquired=True)

    try:
        yield acquired
    finally:
        if not ended:
            profiler.emit("sdk_limit_wait", stage=name, wait_s=time.perf_counter() - start, acquired=False)


def _install_training_hooks(profiler):
    from platoon.train.tinker import ppo
    from platoon.train.tinker.rl import PlatoonTinkerRLTrainer

    def wait_wrapper(original):
        @functools.wraps(original)
        async def wrapped(self, coro, name, *args, **kwargs):
            with profiler.span("training.wait", phase=name):
                return await original(self, coro, name, *args, **kwargs)

        return wrapped

    def update_wrapper(original):
        @functools.wraps(original)
        async def wrapped(trainer, state, *args, **kwargs):
            with request_trace():
                with profiler.span("training.ppo_update", step=state.train_step):
                    return await original(trainer, state, *args, **kwargs)

        return wrapped

    def prepare_wrapper(original):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            with profiler.span("training.prepare_batch"):
                return original(*args, **kwargs)

        return wrapped

    profiler.patch(PlatoonTinkerRLTrainer, "_await_with_heartbeat", wait_wrapper)
    profiler.patch(ppo, "update_batch", update_wrapper)
    profiler.patch(ppo, "prepare_batch", prepare_wrapper)


@contextmanager
def profiling_session(config: ProfilingConfig, run_log_path):
    global _active
    if not config.enabled:
        yield None
        return
    if _active is not None:
        raise RuntimeError("A profiling session is already active")
    path = Path(run_log_path) / "profiling" / f"trace-{time.time_ns()}-{os.getpid()}.jsonl"
    profiler = Profiler(path, config)
    _active = profiler
    try:
        _install_sdk_hooks(profiler)
        _install_training_hooks(profiler)
        profiler.emit(
            "profile_started",
            pid=os.getpid(),
            heartbeat_interval_s=config.heartbeat_interval_s,
            buffer_events=config.buffer_events,
        )
        logger.info("Inference profiling enabled: %s", path)
        yield profiler
    finally:
        profiler.close()
        _active = None
