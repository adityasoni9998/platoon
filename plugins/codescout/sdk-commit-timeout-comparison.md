# SDK Commit Timeout Comparison

This note compares the relevant behavior of `software-agent-sdk` between:

```text
old: 737eae2b2384f04fe5853639294931284ab1f283
new: 43376f1868ffd702746080714a59c16d3f69ec12
```

It explains why CodeScout's current OpenHands adapter is more likely to time out with the newer SDK.

## Executive Summary

The timeout is not explained by parallel tool calling. In the inspected rollout JSONL files:

```text
new run:
  max action_events in a Platoon step: 1
  files with >1 tool call in one assistant message: 0
  tool_concurrency_limit in OpenHands Agent state: 1

old run:
  max action_events in a Platoon step: 1
  files with >1 tool call in one assistant message: 0
  tool_concurrency_limit in OpenHands Agent state: 1
```

The timeout is also not explained merely by this log line:

```text
Clearing new_actions due to action event that has not been observed in a future observation: ... ActionEvent
```

That line appears in both old and new runs. It is normal transient polling: Platoon can poll after OpenHands emitted an `ActionEvent` but before the matching observation is visible.

The real regression is this:

```text
old run: 1 timeout-like rollout error / 13004 rollout files
new run: 107 timeout-like rollout errors / 279 rollout files
```

In the new run, the timeout always happens while Platoon is waiting for `OpenHandsAgent.act(obs)`:

```python
# platoon/episode/loop.py
action = await asyncio.wait_for(agent.act(obs), timeout=timeout)
```

That means Platoon is waiting for OpenHands to expose the next agent action/message, but no acceptable action/message becomes visible before the 300s timeout.

## How CodeScout Currently Drives OpenHands

CodeScout does not await OpenHands directly in the same control flow. It starts OpenHands in a background daemon thread:

```python
# plugins/openhands/platoon/openhands/env.py
self._conversation.send_message(self._task.goal)
self._run_thread = threading.Thread(
    target=self._conversation.run,
    kwargs={"timeout": 420},
    daemon=True,
)
self._run_thread.start()
```

Then Platoon's adapter polls the remote conversation state:

```python
# plugins/openhands/platoon/openhands/agent.py
step_actions = get_actions_for_last_obs(obs, require_same_llm_call_id=True)
while not step_actions and not is_finished(obs):
    await asyncio.sleep(0.2)
    step_actions = get_actions_for_last_obs(obs, require_same_llm_call_id=True)
```

So the adapter depends on two things being true:

1. The background `conversation.run()` thread keeps running and does not fail silently.
2. `conversation.state.events` and `conversation.state.execution_status` become up to date while the adapter polls them.

The newer SDK made both assumptions weaker.

## Observed Timeout Modes In The New Run

From the new run:

```text
/data/user_data/adityabs/platoon_skyrl_tinker/logs/codescout-platoon-tinker/skyrl-amd-codescout-lora-cispo-20260630-134553
```

The 107 timeout files split into two important groups:

```text
67 timeout files:
  no tinker-sampling-* completion id appears in the trajectory
  no first OpenHands agent action/message reached Platoon

40 timeout files:
  at least one tinker-sampling-* completion/action appears
  Platoon later waits for another action boundary and times out
```

These two groups should not be collapsed into one explanation.

### Group A: No First Action

For 67 timeout files, the trajectory contains only the initial observation events, such as:

```text
ConversationStateUpdateEvent(full_state)
SystemPromptEvent
MessageEvent(user)
ConversationStateUpdateEvent(last_user_message_id)
ConversationStateUpdateEvent(execution_status=running)
```

Then `OpenHandsAgent.act()` waits for a first action until Platoon's 300s timeout.

This is not caused by `get_actions_for_last_obs()` clearing an action. There is no action yet.

Possible causes:

- the SDK/server never called the LLM,
- the Tinker sampling request was still running,
- the SDK/server run thread failed before emitting an action,
- the remote event cache did not receive/reconcile the action event,
- the agent server was alive but stuck before the first agent step.

The current logs are not sufficient to distinguish those cases because `env.py` does not capture the exception/result of the background `conversation.run()` thread.

### Group B: Action Exists But Next Boundary Never Arrives

For 40 timeout files, at least one `tinker-sampling-*` completion id/action appears. These are the cases where `openhands_utils.py` is more directly relevant.

The adapter tries to expose only complete action/observation boundaries to Platoon. If it sees an action but no future observation, it clears that action and keeps polling. If the remote cache never catches up, this becomes an infinite wait until the episode timeout.

## Concrete SDK Behavior Changes

### Change 1: RemoteConversation completion now waits for post-run full-state snapshots

Old SDK behavior, simplified:

```python
ws_status = self._terminal_status_queue.get(timeout=poll_interval)
self._handle_conversation_status(ws_status)
logger.info("Run completed via WebSocket notification ...")
return
```

Old REST fallback:

```python
if REST status is terminal for 3 consecutive polls:
    self._state.events.reconcile()
    return
```

New SDK behavior, simplified:

```python
self._run_armed.set()
self._wait_for_run_completion(...)
self._run_armed.clear()
```

And `_wait_for_run_completion()` now prefers a post-run full-state WebSocket snapshot:

```python
ws_status = self._terminal_status_queue.get(timeout=poll_interval)
self._handle_conversation_status(ws_status)
logger.info("Run completed via post-run WebSocket state update ...")
self._state.events.reconcile()
return
```

REST `FINISHED` is no longer immediately authoritative:

```python
# New SDK idea:
# FINISHED from REST is advisory because stop hooks can still revert it.
# Wait for post-run WS full-state, or accept REST terminal only after a hard fallback.
```

Why this matters for CodeScout:

- CodeScout polls `conversation.state.events` while `conversation.run()` is active in another thread.
- The newer SDK's strongest event reconciliation happens after the run is considered complete.
- CodeScout does not wait on `conversation.run()`; it polls the state concurrently.
- If WebSocket delivery, full-state delivery, or reconciliation lags, Platoon can observe stale/incomplete events for longer.

This does not by itself prove every timeout, but it directly weakens CodeScout's polling assumption.

### Change 2: RemoteConversation.run() can fail in the background thread and CodeScout loses the exception

The newer run logs include `ConversationRunError` and `MaxIterationsReached` entries in `output.log`.

Because CodeScout starts `conversation.run()` as:

```python
threading.Thread(target=self._conversation.run, ...)
```

an exception raised inside `RemoteConversation.run()` only terminates that daemon thread. It does not automatically update Platoon's async loop.

CodeScout currently does not store:

- whether the run thread ended,
- whether the run thread raised,
- the exception object,
- the last remote execution status after the thread ended.

So the adapter can keep polling for actions even after OpenHands has already failed or stopped.

This is a concrete adapter bug that became more important with the newer SDK because remote run completion/error behavior is stricter and more exception-driven.

### Change 3: Agent action execution was refactored into batches

Old SDK action execution, simplified:

```python
for action_event in action_events:
    self._execute_action_event(conversation, action_event, on_event=on_event)
```

New SDK action execution, simplified:

```python
batch = _ActionBatch.prepare(...)
batch.emit(on_event)
batch.finalize(...)
```

For true parallel tool calls, this would widen the time window where one or more action events exist before all corresponding observations are emitted.

However, in the inspected CodeScout logs there were no recorded parallel tool calls. So this change is probably not the primary cause of the observed timeout regression.

It still matters as a compatibility note: `openhands_utils.py` assumes action/observation pairing is easy to infer by scanning the event list. Batched execution makes that assumption less robust in general.

### Change 4: Agent response handling changed for empty/reasoning-only outputs

Old SDK behavior for no tool call:

- emit an agent `MessageEvent`,
- finish only if there is text content,
- continue if the response is reasoning-only/empty.

New SDK behavior:

- classify response as `TOOL_CALLS`, `CONTENT`, `REASONING_ONLY`, or `EMPTY`,
- for `REASONING_ONLY`/`EMPTY`, emit the agent message and add a corrective user nudge:

```text
Your last response did not include a function call or a message. Please use a tool to proceed with the task.
```

This can change iteration dynamics and prompt history. It is not the main explanation for the 67 "no first action" timeouts because those files do not show an emitted agent message/action at all. But it can affect cases where the model produces malformed or empty output repeatedly.

## Why The Existing Adapter Is Fragile

The current `is_finished()` requires two conditions:

```python
oh_conversation_finished = execution_status in {FINISHED, STUCK, ERROR}
platoon_episode_caught_up = last_event_id in {
    last_step_action_id,
    last_step_observation_id,
    last_event_seen,
}
return oh_conversation_finished and platoon_episode_caught_up
```

This is too strict for a remote conversation being updated by another thread.

It can be false when:

- the server is terminal but the local cached `execution_status` has not refreshed,
- an error event exists but the adapter has not advanced `last_step_*`,
- the run thread already raised but Platoon did not capture it,
- the final event has not reconciled into the local event cache,
- Platoon sees an action but not the matching observation.

The adapter also has no independent way to know that the background run thread has died.

## What The New SDK Changed That Makes This Show Up More

The most important concrete change is not "parallel tool calling." It is the remote conversation lifecycle:

```text
old SDK:
  terminal execution_status WS event usually completed the client run quickly
  REST terminal fallback completed after a few polls
  adapter's stale-state window was smaller

new SDK:
  completion prefers post-run full-state WS snapshot
  REST FINISHED is advisory for longer
  reconciliation is tied to completed run paths
  run errors can raise in the background thread
  adapter's stale-state / lost-exception window is larger
```

CodeScout's adapter was written as if polling `conversation.state.events` is enough to drive an episode. With the newer SDK, the adapter needs to explicitly coordinate with the run thread and remote reconciliation.

## What Is Proven Versus Still A Hypothesis

Proven from logs:

- The new run has far more `agent.act()` timeouts than the old run.
- The timeout is not limited to parallel tool calling.
- Many new timeouts happen before any first `tinker-sampling-*` action appears.
- Some new timeouts happen after at least one action appears.
- `Clearing new_actions...` exists in old and new runs, so it is not sufficient as a root cause.
- The new SDK uses "post-run WebSocket state update" completion logs; the old run uses "WebSocket notification" completion logs.

Likely but not fully proven without more instrumentation:

- Some timeouts are caused by `conversation.run()` failing or ending in the background thread while Platoon keeps polling.
- Some timeouts are caused by remote state/events not being reconciled before `get_actions_for_last_obs()` decides no valid action exists.
- Some "no first action" timeouts may be slow/stuck Tinker sampling rather than SDK event handling.

## Concrete Fixes

### Fix 1: Capture background run thread outcome

Do not start `conversation.run()` directly as the thread target. Wrap it:

```python
def _run_conversation(self) -> None:
    try:
        self._conversation.run(timeout=420)
    except BaseException as exc:
        self._run_exception = exc
    finally:
        self._run_finished = True
```

Then polling loops should check:

```python
if self._run_finished and self._run_exception is not None:
    refresh/reconcile remote state
    raise or mark the Platoon trajectory error immediately
```

This directly addresses lost `ConversationRunError`.

### Fix 2: Add explicit remote refresh/reconcile before declaring "no action"

When `get_actions_for_last_obs()` returns no action for a while, or when it is about to clear actions due to missing observations, force a sync if the state supports it:

```python
events = observation.conversation_state.events
if callable(getattr(events, "reconcile", None)):
    events.reconcile()

if callable(getattr(observation.conversation_state, "refresh_from_server", None)):
    observation.conversation_state.refresh_from_server()
```

This should be rate-limited or done only on suspicious states, not necessarily every 0.2s.

### Fix 3: Make terminal detection event-aware

Do not rely only on cached `conversation_state.execution_status`.

Also scan recent events for:

- `ConversationErrorEvent`,
- `ConversationStateUpdateEvent(key="execution_status", value in terminal statuses)`,
- `ConversationStateUpdateEvent(key="full_state", value.execution_status in terminal statuses)`.

If these are visible, unblock Platoon even if `last_step_action_id` / `last_step_observation_id` has not caught up perfectly.

### Fix 4: Add a diagnostic timeout inside the adapter

Do not let every failure collapse into:

```text
Error in episode loop at step ...
TimeoutError
```

If `OpenHandsAgent.act()` waits too long, emit a specific error containing:

- run thread alive/dead,
- stored run thread exception,
- current remote execution status,
- last 10 event ids/kinds,
- whether any `ActionEvent` exists after `last_step_observation_id`,
- whether any matching observation exists,
- whether any `ConversationErrorEvent` exists.

This would make the next failure obvious.

## Minimal Mental Model

The old SDK made this polling loop mostly work:

```text
start remote run in thread
poll remote event list until action appears
poll remote event list until observation appears
terminal status arrives quickly
```

The new SDK needs this instead:

```text
start remote run in thread
capture thread exceptions
periodically reconcile remote event list
periodically refresh remote state
treat terminal/error events as terminal even if Platoon cursors lag
fail with adapter diagnostics instead of waiting for generic 300s timeout
```

