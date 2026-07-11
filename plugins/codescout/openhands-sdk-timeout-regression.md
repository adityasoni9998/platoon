# OpenHands SDK Timeout Regression Notes

## Summary

The frequent rollout errors that look like:

```text
Error in episode loop at step ...
TimeoutError
```

are caused by a synchronization mismatch between Platoon's OpenHands adapter and newer `software-agent-sdk` behavior around remote conversation completion and event reconciliation.

This is not primarily caused by the Apptainer `tini` warning:

```text
Tini is not running as PID 1 and isn't registered as a child subreaper.
Zombie processes will not be re-parented to Tini, so zombie reaping won't work.
```

That warning is noisy and may matter for long-running cleanup health, but the observed timeout stack is in Platoon waiting for OpenHands actions/state, not in process reaping.

## Evidence From The Run

New run:

```text
/data/user_data/adityabs/platoon_skyrl_tinker/logs/codescout-platoon-tinker/skyrl-amd-codescout-lora-cispo-20260630-134553
```

Old run:

```text
/data/user_data/adityabs/platoon_skyrl_tinker/logs/codescout-platoon-tinker/skyrl-amd-codescout-fft-cispo-20260625-141230
```

Approximate counts from the rollout JSONL files:

```text
new: 107 timeout-like episode-loop errors / 279 rollout files
old:   1 timeout-like episode-loop error  / 13004 rollout files
```

The timeout stack points to `agent.act(obs)`:

```text
platoon/episode/loop.py
  action = await asyncio.wait_for(agent.act(obs), timeout=timeout)

plugins/openhands/platoon/openhands/agent.py
  while not step_actions and not is_finished(obs):
      await asyncio.sleep(0.2)
      step_actions = get_actions_for_last_obs(...)
```

Both the old and new runs contain many instances of:

```text
Clearing new_actions due to action event that has not been observed in a future observation: ... ActionEvent
```

That message comes from `platoon/utils/openhands_utils.py`. By itself it is not the regression. It is a normal transient polling artifact: Platoon may poll after OpenHands has emitted an `ActionEvent` but before the corresponding observation is visible. In the old run this often recovered quickly. The regression is the much higher rate of rollouts where the poll never recovers before Platoon's 300s episode timeout.

In other words, the bug is not "this log line appears"; the bug is "this log line, or no action at all, is followed by a long wait in `OpenHandsAgent.act()` until `run_episode()` times out."

## What Changed In software-agent-sdk

The relevant SDK commits are:

```text
old: 737eae2b2384f04fe5853639294931284ab1f283
new: 43376f1868ffd702746080714a59c16d3f69ec12
```

### 1. Remote run completion became stricter and more WebSocket/full-state dependent

In the older SDK, `RemoteConversation._wait_for_run_completion()` returned as soon as it received a terminal WebSocket execution-status update. As fallback, it also returned after several consecutive REST polls showed a terminal status.

In the newer SDK, REST `FINISHED` is treated as advisory. The client prefers a post-run full-state WebSocket snapshot before returning, and REST terminal status is accepted only after a longer hard fallback. `ERROR` and `STUCK` can still raise `ConversationRunError` from inside `RemoteConversation.run()`.

This matters because CodeScout starts the SDK run in a daemon thread:

```python
self._run_thread = threading.Thread(
    target=self._conversation.run,
    kwargs={"timeout": 420},
    daemon=True,
)
self._run_thread.start()
```

If `RemoteConversation.run()` raises in that thread, the exception is not propagated into Platoon's async loop. Platoon continues polling `conversation_state.events` and `conversation_state.execution_status`, which may be stale or unreconciled.

### 2. Action execution changed from immediate per-action execution to batch preparation/emission

Older SDK behavior was effectively:

```python
for action_event in action_events:
    self._execute_action_event(conversation, action_event, on_event=on_event)
```

Newer SDK behavior goes through `_ActionBatch`:

```python
batch = _ActionBatch.prepare(...)
batch.emit(on_event)
batch.finalize(...)
```

This makes the event stream more likely to contain transient states where an `ActionEvent` is visible before its corresponding observation or terminal update is visible to the remote client cache.

`openhands_utils.py` is fragile to exactly that state. It assumes an action without a future observation should be hidden/cleared unless the conversation is already known to be terminal. With the newer SDK, the local cached view can be missing the observation or terminal update even though the remote run has already progressed or errored.

### 3. The SDK now relies more on cached/incremental views and explicit reconciliation

The newer SDK introduced more explicit event/cache reconciliation behavior in remote conversations. The SDK's own `run()` path reconciles events after authoritative run completion. Platoon's adapter, however, reads `conversation_state.events` concurrently while the background run is still active, and does not explicitly reconcile before deciding whether actions are valid.

This breaks the adapter's implicit assumption:

```text
If an ActionEvent exists, then the matching observation or terminal state will be visible in conversation_state.events/execution_status without extra coordination.
```

That assumption held often enough in the older SDK, but not in the newer SDK.

## Why openhands_utils.py Breaks

The fragile logic is in `get_actions_for_last_obs()`:

```python
if not is_finished(observation, last_event_seen=last_event_seen) and not at_least_one_future_error_event_seen:
    for action in new_actions:
        if action.id not in seen_action_ids:
            print(
                "Clearing new_actions due to action event that has not been observed "
                f"in a future observation: {action.id} {action.kind}",
                flush=True,
            )
            new_actions.clear()
            break
```

This is intended to avoid exposing an action to Platoon before OpenHands has executed it and produced an observation. That is reasonable, but it requires the adapter to reliably see the matching observation or terminal state.

In the new SDK, this sequence can happen:

1. OpenHands emits an `ActionEvent`.
2. Platoon's `OpenHandsAgent.act()` polls `get_actions_for_last_obs()`.
3. The matching observation, final full-state snapshot, or `ConversationErrorEvent` is not yet present in the local remote event cache.
4. `get_actions_for_last_obs()` clears the action.
5. The remote conversation has already ended, errored, or the run thread has raised.
6. Platoon never observes a valid new action and `is_finished()` remains false because local state is stale.
7. `OpenHandsAgent.act()` spins until `run_episode()` times out.

This explains the subset of timeouts where an LLM response/action was already emitted but Platoon never sees the next valid action boundary. Another large subset of the new timeouts happens before any `tinker-sampling-*` completion id appears, which means OpenHands had not emitted even the first agent action before Platoon's 300s `agent.act()` timeout. For that subset, the adapter should still fail with a clearer "LLM/run thread did not produce an action" error, but the root cause may be slow or stuck sampling rather than action/observation pairing.

## Proposed Fixes

### Fix 1: Capture background run-thread exceptions

Wrap `self._conversation.run()` in a small target function that stores exceptions on the environment object.

Example shape:

```python
def _run_conversation(self) -> None:
    try:
        self._conversation.run(timeout=420)
    except BaseException as exc:
        self._run_exception = exc
```

Then in the polling loops in `reset()`, `step()`, and/or `OpenHandsAgent.act()`, check:

```python
if self._run_thread is not None and not self._run_thread.is_alive() and self._run_exception is not None:
    # refresh/reconcile state if possible, then terminate or raise a controlled error
```

Without this, `ConversationRunError` can kill the daemon thread while Platoon keeps polling forever.

### Fix 2: Explicitly reconcile remote events before clearing actions

Before `get_actions_for_last_obs()` clears an action due to a missing future observation, reconcile the remote event cache if available.

The exact location can be either:

- in the adapter polling loop before calling `get_actions_for_last_obs()`, or
- in a helper used by `get_actions_for_last_obs()` / `get_obs_for_last_action()`.

The helper should be defensive because local and remote conversations expose different state implementations:

```python
events = observation.conversation_state.events
reconcile = getattr(events, "reconcile", None)
if callable(reconcile):
    reconcile()

refresh = getattr(observation.conversation_state, "refresh_from_server", None)
if callable(refresh):
    refresh()
```

Do not call this on every 0.2s poll unconditionally if it becomes too expensive. A practical approach is to reconcile only when:

- an action would otherwise be cleared,
- no action/observation has appeared after a short grace period,
- the run thread is no longer alive,
- or the cached state says terminal/error.

### Fix 3: Treat terminal/error events as terminal even if cached execution_status lags

`is_finished()` currently depends on `conversation_state.execution_status` plus a caught-up event id check. That is too strict for the newer remote SDK.

Make terminal detection inspect recent events as well:

- `ConversationErrorEvent`
- `ConversationStateUpdateEvent(key="execution_status", value in {"finished", "error", "stuck"})`
- `ConversationStateUpdateEvent(key="full_state", value.execution_status in {"finished", "error", "stuck"})`

If any of these appears after the last Platoon-observed event, unblock the loop. For `ERROR` and `STUCK`, set Platoon's error state rather than continuing to wait for a matched action observation.

### Fix 4: Do not clear terminal actions only because their observation is missing

For terminal custom tools such as `LocalizationFinishTool`, it is valid for the conversation to become terminal at about the same time as the action is emitted. The adapter should accept a terminal action when either:

- the action itself is a finish action,
- the action's tool is a known terminal custom tool,
- a terminal state update appears after the action,
- or the remote run thread has ended and state refresh shows terminal status.

The current code already partially handles `FinishAction` and `_is_terminal_status()`, but `_is_terminal_status()` can be stale for remote conversations. That is the part that needs strengthening.

### Fix 5: Avoid infinite adapter polling

Even after the above fixes, the adapter should not spin forever inside `OpenHandsAgent.act()` or `OpenHandsEnv.step()` without reporting what it is waiting for.

Add an adapter-level timeout or diagnostic path that includes:

- last known `execution_status`,
- whether the run thread is alive,
- stored run-thread exception if any,
- last few event kinds/ids,
- last action id and whether a matching observation exists.

This makes future SDK synchronization regressions fail with an actionable error instead of a generic episode timeout.

## Separate Issue: Tinker Prefix Mismatch

The prefix mismatch debug files show another real issue:

```text
previous sampled action:
<tool_call>
{
  "name": "terminal",
  "arguments": {
    ...
  }
}
</tool_call><|im_end|>

next prompt rerender:
<tool_call>
{"name": "terminal", "arguments": {...}}
</tool_call><|im_end|>
```

This comes from the local Qwen3 renderer compacting JSON when rendering historical tool calls:

```python
json.dumps(payload, ensure_ascii=False)
```

That should be changed to render the same format as the sampled assistant action, likely pretty JSON with stable formatting. This affects sequence merging and training data quality, but it is not the direct cause of the 300s `agent.act()` timeout.

## Recommended Implementation Order

1. Capture and surface background `conversation.run()` exceptions.
2. Add remote `events.reconcile()` / `refresh_from_server()` before deciding no action/observation exists.
3. Strengthen terminal detection using terminal state/error events, not only cached `execution_status`.
4. Adjust `get_actions_for_last_obs()` so terminal/error paths do not clear actions forever.
5. Fix Qwen3 tool-call JSON rendering for Tinker prefix consistency.
