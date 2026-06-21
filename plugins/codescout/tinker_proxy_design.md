# CodeScout Tinker Proxy Design Notes

## Problem

The existing `TinkerLLMProxySession` works when LiteLLM calls happen in the same Python process as the training loop. CodeScout is different: OpenHands runs the agent loop inside an `ApptainerWorkspace` agent-server. OpenHands serializes the `Agent`/`LLM` config to that agent-server, so LiteLLM calls happen in the Apptainer process, not in the host training loop.

Therefore the host-side `TinkerLLMProxySession` around `rollout_fn(...)` cannot capture those interactions.

## Agreed Architecture

Run one host-side HTTP proxy server inside the trainer process. All Apptainer agent-servers call this single endpoint for LLM inference.

```text
host trainer process
  - PlatoonTinkerRLTrainer
  - TinkerLLM / sampling_client
  - one HTTP proxy server
  - shared interactions_by_session dict
  - many async rollout tasks

apptainer agent-server processes
  - OpenHands agent loop
  - OpenHands LLM calls host proxy over HTTP
```

The proxy exposes an OpenAI-compatible chat endpoint:

```text
POST /v1/chat/completions
GET /health
```

Use `openai/platoon-tinker` in OpenHands, not `litellm_proxy/...`, unless we choose to run a full LiteLLM proxy. With `openai/...`, OpenHands does not require `/v1/model/info`.

## Interaction Capture

Each rollout gets a unique `session_id`.

OpenHands `LLM` must send it on every request:

```python
LLM(
    model="openai/platoon-tinker",
    base_url="http://HOST:PORT/v1",
    api_key="sk-xxx",
    max_input_tokens=context_window_length,
    max_output_tokens=config.inference_params.max_completion_tokens,
    extra_headers={"X-Platoon-Tinker-Session": session_id},
)
```

The HTTP request handler creates a local `TinkerLLMProxySession` for each request:

```python
async with TinkerLLMProxySession() as proxy_session:
    response = await tinker_llm.acompletion(**payload)

state.add_interactions(session_id, proxy_session.interactions)
```

This works because `TinkerLLM._record_interaction()` writes into the `proxy_interactions` `ContextVar` in the same request task.

## Where Interactions Are Stored

Interactions should be stored in a shared Python object in the host trainer process:

```python
interactions_by_session: dict[str, dict[str, TinkerLLMInteraction]]
```

The HTTP server writes to it. `run_rollout()` reads/pops from it after the rollout finishes.

This does not need an HTTP endpoint for returning interactions if the proxy server is started in-process by the trainer.

If the proxy runs as a separate process, interaction retrieval must instead use HTTP, Redis, SQLite, or another IPC mechanism.

## Rollout Return Value

To keep `GroupRolloutWorkflow` simple, `run_rollout()` can attach interactions to its existing returned trajectory dict:

```python
result = current_trajectory_collection.get().to_dict()
result["_tinker_interactions"] = tinker_proxy_state.pop_interactions(session_id)
return result
```

Then `GroupRolloutWorkflow.arun_episode_single` should use:

```python
results = await self.rollout_fn(task, rollout_config)
interactions = results.pop("_tinker_interactions", {})
```

and pass `interactions` into `get_train_data_for_trajectory_collection(...)`.

## Failure Handling

`run_rollout()` does not always return gracefully. Therefore cleanup should happen in `finally` or exception paths:

```python
try:
    ...
    result["_tinker_interactions"] = proxy_state.pop_interactions(session_id)
    return result
except Exception:
    proxy_state.discard_session(session_id)
    raise
finally:
    cleanup_resources(...)
```

The proxy state should also support TTL cleanup for abandoned sessions.

## Threading / Parallelism

The simplest design is one server hosted in a background thread with its own asyncio event loop. The shared store must use a `threading.Lock` or `threading.RLock`, because the server thread writes interactions while the trainer event loop reads them.

A single server thread is not necessarily a single concurrent request. Async HTTP servers can handle many concurrent requests on one event loop, as long as the handler awaits async Tinker calls.

Multiple server worker processes should be avoided unless interaction storage is moved out of memory, because each process would have its own `interactions_by_session` dict.

Multiple threads can be used only if access to the shared store and any shared Tinker client state is safe. Start with one server thread/event loop; add concurrency only after verifying `tinker_llm.acompletion` and `sampling_client` are safe under concurrent calls.

## Token Limits

`openai/...` does not compromise output length if `max_output_tokens` is set explicitly. OpenHands maps it to `max_completion_tokens`.

`max_input_tokens` in OpenHands is mostly metadata/telemetry/condenser state and is not the primary enforcement mechanism. The real check should be in `TinkerLLM._check_context_window_length()`:

```python
prompt_length + max_completion_tokens <= context_window_length
```

CodeScout currently does not set `train.context_window_length`, so local Tinker context-window prevalidation is disabled. Set it explicitly in the Tinker config and pass the same value to OpenHands `LLM(max_input_tokens=...)`.

