import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
import tinker

from platoon.train.tinker import profiling


def read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    profiler = profiling.Profiler(tmp_path / "trace.jsonl", profiling.ProfilingConfig(heartbeat_interval_s=0.01))
    monkeypatch.setattr(profiling, "_active", profiler)
    yield profiler
    profiler.close()


@pytest.mark.asyncio
async def test_slow_writer_and_full_buffer_never_wait_in_producer(tmp_path, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    loop_thread = threading.get_ident()
    original = profiling.Profiler._write_batch

    def slow_write(self, stream, batch):
        assert threading.get_ident() != loop_thread
        if batch:
            entered.set()
            release.wait(2)
        original(self, stream, batch)

    monkeypatch.setattr(profiling.Profiler, "_write_batch", slow_write)
    profiler = profiling.Profiler(tmp_path / "slow.jsonl", profiling.ProfilingConfig(buffer_events=8))
    try:
        profiler.emit("first")
        async with asyncio.timeout(2):
            while not entered.is_set():
                await asyncio.sleep(0.001)
        start = time.perf_counter()
        for _ in range(1000):
            profiler.emit("while_disk_stalled")
        await asyncio.sleep(0)
        assert time.perf_counter() - start < 0.5
        assert profiler.dropped >= 992
        assert len(profiler.pending) <= 8
        # Even a busy buffer mutex must cause a drop, not a blocking acquisition.
        profiler.lock.acquire()
        try:
            start = time.perf_counter()
            profiler.emit("contended_mutex")
            assert time.perf_counter() - start < 0.1
        finally:
            profiler.lock.release()
    finally:
        release.set()
        profiler.close()
    assert read_events(profiler.path)[-1]["dropped"] == profiler.dropped


@pytest.mark.asyncio
async def test_thread_queue_timing_and_concurrent_trace_isolation(recorder):
    gate = threading.Event()
    entered = threading.Event()
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=1))

    def work(block):
        if block:
            entered.set()
            gate.wait(2)
        return profiling._trace.get()

    async def request(label, block):
        with profiling.request_trace(label):
            return await profiling.to_thread("tokenize", work, block)

    a = asyncio.create_task(request("first", True))
    async with asyncio.timeout(2):
        while not entered.is_set():
            await asyncio.sleep(0.001)
    b = asyncio.create_task(request("second", False))
    await asyncio.sleep(0.04)
    gate.set()
    assert await asyncio.gather(a, b) == ["first", "second"]
    recorder.close()
    events = read_events(recorder.path)
    queued = {e["trace_id"]: e["wait_s"] for e in events if e["event"] == "thread_queue"}
    assert queued["second"] >= 0.025
    executions = [e for e in events if e.get("span") == "tokenize.execute" and e["event"] == "span_start"]
    assert {e["trace_id"] for e in executions} == {"first", "second"}


@pytest.mark.asyncio
async def test_loop_monitor_detects_stall_without_inheriting_request_trace(recorder):
    with profiling.request_trace("not-a-heartbeat"):
        recorder.watch_loop("test")
        await asyncio.sleep(0.02)
        time.sleep(0.05)  # Deliberately stall this test loop.
        await asyncio.sleep(0.02)
    recorder.close()
    events = [e for e in read_events(recorder.path) if e["event"] == "loop_lag"]
    assert max(e["lag_s"] for e in events) > 0.025
    assert all(e["trace_id"] is None for e in events)


@pytest.mark.asyncio
async def test_proxy_middleware_preserves_responses_and_separates_requests(recorder):
    async def app(scope, receive, send):
        await receive()
        await asyncio.sleep(0.001)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})

    middleware = profiling.ProxyTraceMiddleware(app)

    async def request(i):
        messages = []

        async def receive():
            return {"type": "http.request", "body": b'{"messages":[]}', "more_body": False}

        async def send(message):
            messages.append(message)

        await middleware(
            {
                "type": "http",
                "path": "/v1/chat/completions",
                "headers": [(b"x-platoon-tinker-session", str(i).encode())],
            },
            receive,
            send,
        )
        assert messages[-1]["body"] == b"{}"
        return dict(messages[0]["headers"])[b"x-platoon-trace-id"].decode()

    traces = await asyncio.gather(*(request(i) for i in range(30)))
    assert len(set(traces)) == 30
    recorder.close()
    events = read_events(recorder.path)
    assert len([e for e in events if e["event"] == "proxy_complete"]) == 30
    assert recorder.active_requests == 0


@pytest.mark.asyncio
async def test_sdk_hooks_with_real_sdk_mock_server_concurrency_and_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("TINKER_TELEMETRY", "0")
    monkeypatch.delenv("TINKER_SUBPROCESS_SAMPLING", raising=False)
    submitted = {}
    retrieves = {}

    async def server(request):
        body = json.loads(request.content)
        route = request.url.path.rsplit("/", 1)[-1]
        if route == "create_session":
            return httpx.Response(200, json={"session_id": "test-session"})
        if route == "create_sampling_session":
            return httpx.Response(200, json={"sampling_session_id": "test-sampling"})
        if route == "asample":
            request_id = f"req-{body['seq_id']}"
            submitted[request_id] = body["prompt"]["chunks"][0]["tokens"][0]
            await asyncio.sleep(0.005)
            return httpx.Response(200, json={"request_id": request_id})
        if route == "retrieve_future":
            rid = body["request_id"]
            retrieves[rid] = retrieves.get(rid, 0) + 1
            await asyncio.sleep(0.005)
            if retrieves[rid] == 1:
                return httpx.Response(408, json={"queue_state": "active"})
            return httpx.Response(
                200,
                json={
                    "type": "sample",
                    "sequences": [{"tokens": [submitted[rid]], "logprobs": [-0.1], "stop_reason": "stop"}],
                },
            )
        if route == "session_heartbeat":
            return httpx.Response(200, json={"type": "session_heartbeat"})
        raise AssertionError(route)

    original_send = httpx.AsyncClient.send
    with profiling.profiling_session(profiling.ProfilingConfig(enabled=True, heartbeat_interval_s=0.01), tmp_path) as p:
        client = httpx.AsyncClient(transport=httpx.MockTransport(server))
        service = await asyncio.to_thread(
            tinker.ServiceClient, base_url="https://test.invalid", api_key="tml-secret-test-key", http_client=client
        )
        sampling = await service.create_sampling_client_async(base_model="test")

        async def call(i):
            with profiling.request_trace(f"trace-{i}"):
                return await sampling.sample_async(
                    prompt=tinker.ModelInput.from_ints([i]),
                    num_samples=1,
                    sampling_params=tinker.SamplingParams(max_tokens=1),
                )

        results = await asyncio.gather(*(call(i) for i in range(20)))
        assert [r.sequences[0].tokens for r in results] == [[i] for i in range(20)]
        service.holder._loop.call_soon_threadsafe(service.holder._session_heartbeat_task.cancel)
        await client.aclose()
    assert httpx.AsyncClient.send is original_send
    events = read_events(p.path)
    acks = [e for e in events if e["event"] == "sdk_ack"]
    assert len(acks) == 20
    mapping = {e["tinker_request_id"]: e["trace_id"] for e in acks}
    assert len(set(mapping.values())) == 20
    futures = [e for e in events if e.get("span") == "sdk.future" and e["event"] == "span_start"]
    assert len(futures) == 20
    for event in futures:
        assert mapping[event["tinker_request_id"]] == event["trace_id"]
    assert len([e for e in events if e.get("status") == 408]) == 20
    assert len([e for e in events if e["event"] == "sdk_scheduled"]) == 20
    text = p.path.read_text()
    assert "secret-test-key" not in text
    assert '"tokens"' not in text


def test_disabled_session_does_not_patch_or_create_files(tmp_path):
    original = httpx.AsyncClient.send
    with profiling.profiling_session(profiling.ProfilingConfig(), tmp_path) as recorder:
        assert recorder is None
        assert httpx.AsyncClient.send is original
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_cancellation_closes_span_and_clears_trace(recorder):
    entered = asyncio.Event()

    async def request():
        with profiling.request_trace("cancelled-request"):
            with profiling.span("cancelled_stage"):
                entered.set()
                await asyncio.sleep(100)

    task = asyncio.create_task(request())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert profiling._trace.get() is None
    recorder.close()
    ends = [e for e in read_events(recorder.path) if e["event"] == "span_end"]
    assert ends[0]["error_type"] == "CancelledError"
    assert ends[0]["trace_id"] == "cancelled-request"


def test_rebench_cli_enables_profiling_with_new_trial():
    from pathlib import Path

    from platoon.issue_resolution.train_tinker import IssueResolutionTinkerConfig

    from platoon.utils.config import load_config

    root = Path(__file__).resolve().parents[1]
    config, _ = load_config(
        args=[
            "--config",
            str(
                root
                / "plugins/issue-resolution-rebench/platoon/issue_resolution/train_issue_resolution_ppo_tinker.yaml"
            ),
            "--profiling.enabled",
            "true",
            "--stats.trial_name",
            "test-profile-trial",
        ],
        config_class=IssueResolutionTinkerConfig,
    )
    assert isinstance(config.profiling, profiling.ProfilingConfig)
    assert config.profiling.enabled is True
    assert config.stats.trial_name == "test-profile-trial"
    assert config.train.num_concurrent_rollout_workflow_workers == 64
    assert config.train.workflow_config.group_size == 8
