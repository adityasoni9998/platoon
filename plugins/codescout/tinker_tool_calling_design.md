# Tinker Tool Calling for CodeScout

## Summary

CodeScout can support tool calling with Tinker without changing the SkyRL/Tinker backend. The path used by `tinker-cookbook` is text-in/text-out:

1. Convert OpenAI/LiteLLM `tools` into `tinker_cookbook.renderers.ToolSpec`.
2. Ask the selected renderer to inject those tool declarations into the chat prompt with `renderer.create_conversation_prefix_with_tools(...)`.
3. Call `sampling_client.sample_async(...)` exactly as before, with only `prompt`, `sampling_params`, and `num_samples`.
4. Parse generated tool-call syntax from output tokens with `renderer.parse_response(...)`.
5. Return parsed `tool_calls` in the OpenAI-compatible assistant message so OpenHands can execute tools and send `role="tool"` results in the next request.

This does not depend on vLLM/SkyRL tool parser configuration. The backend only sees token IDs. Tool declaration and parsing are handled by `tinker-cookbook` renderers in the Platoon `TinkerLLM` process.

## What tinker-cookbook Does

The relevant cookbook paths are:

- `/tmp/tinker-cookbook/tinker_cookbook/third_party/litellm/provider.py`
- `/tmp/tinker-cookbook/tinker_cookbook/recipes/code_rl/code_env.py`
- `/tmp/tinker-cookbook/tinker_cookbook/tool_use/agent_tool_message_env.py`
- `/tmp/tinker-cookbook/tinker_cookbook/renderers/qwen3.py`
- `/tmp/tinker-cookbook/tinker_cookbook/renderers/kimi_k2.py`
- `/tmp/tinker-cookbook/tinker_cookbook/renderers/qwen3_5.py`

The installed CodeScout environment also has the cookbook LiteLLM provider at:

- `plugins/codescout/.venv/lib/python3.12/site-packages/tinker_cookbook/third_party/litellm/provider.py`

That provider has these key helpers:

- `_convert_openai_messages(...)`: converts OpenAI/LiteLLM messages into cookbook `Message` objects.
- `_convert_openai_tools(...)`: converts OpenAI `{"type": "function", "function": ...}` tools into `ToolSpec`.
- `_prepare_messages_with_tools(...)`: extracts an existing first system message, passes it plus tool specs to `renderer.create_conversation_prefix_with_tools(...)`, and prepends the resulting prefix to the remaining messages.
- `_sample_chat_completion(...)`: renders the final message list, calls `sampling_client.sample_async(...)`, then calls `renderer.parse_response(...)`.

The important point is that `_sample_chat_completion(...)` does not pass `tools` into Tinker `sample_async`. It consumes tools before sampling by changing the rendered prompt.

## Renderer Responsibility

Tool support is renderer-specific.

For `qwen3` / `qwen3_instruct`, `create_conversation_prefix_with_tools(...)` creates a system message containing a `# Tools` section, `<tools>...</tools>` JSON tool declarations, and instructions to emit `<tool_call>...</tool_call>` JSON blocks. `parse_response(...)` then parses generated `<tool_call>` blocks into cookbook `ToolCall` objects.

For `qwen3_5`, parsing builds on the Qwen3 parser but postprocesses XML-style tool calls such as:

```text
<tool_call>
<function=search>
<parameter=query>
...
</parameter>
</function>
</tool_call>
```

For Kimi renderers, tool declarations and tool call syntax use Kimi-specific `tool_declare` and `<|tool_calls_section_begin|>...` formats.

This is why we should not configure a generic vLLM `tool_call_parser` for this path. The parser that matters is `renderer.parse_response(...)`.

## Current Platoon Path

The CodeScout rollout creates an OpenHands `LLM` with model `openai/platoon-tinker`, base URL pointing at the FastAPI proxy, and a per-rollout session header:

- `plugins/codescout/platoon/codescout/rollout.py`

The FastAPI proxy forwards all request fields other than `model`, `messages`, and `stream` into LiteLLM:

- `platoon/train/tinker/fastapi_litellm_proxy.py`

So `tools` and `tool_choice` reach LiteLLM/TinkerLLM as `optional_params`.

The current `TinkerLLM` in:

- `platoon/train/tinker/proxy.py`

already does response-side tool parsing:

```python
parsed_response, parse_success = self.renderer.parse_response(seq.tokens)
tool_calls = parsed_response.get("tool_calls", None)
```

But it does not do request-side tool declaration injection. `_prepare_model_input(...)` currently only canonicalizes messages and calls:

```python
return self.renderer.build_generation_prompt(canonical_messages)
```

That means OpenHands can send tool schemas, and the proxy can forward them, but the sampled model never sees those schemas.

## Implementation Plan

Implement the cookbook pattern directly in `platoon/train/tinker/proxy.py`.

## Can We Import tinker-cookbook Directly?

Yes, but we should be careful about what we import.

`tinker-cookbook` provides a public LiteLLM integration:

```python
from tinker_cookbook.third_party.litellm import register_litellm_provider
```

That registers `provider="tinker"` and returns a `TinkerLiteLLMProvider`. The provider can inject a custom sampling client with `provider.set_client(sampling_client)`, and its implementation already handles OpenAI `tools` by converting them to renderer tool declarations before calling `sampling_client.sample_async(...)`.

However, using that provider wholesale is not a drop-in replacement for Platoon's `TinkerLLM` because Platoon needs extra behavior:

- `TinkerLLMProxySession` interaction capture for RL training data.
- `TinkerLLMInteraction(obs, action)` recording with token logprobs.
- `context_window_length` checks before sampling.
- `renderer_name` and `renderer_kwargs` from our trainer config, rather than always using the cookbook recommended renderer.
- `update_sampling_client(...)`, version tracking, and custom LiteLLM provider naming (`platoon-tinker/...`).
- Current FastAPI proxy behavior and session interaction storage for CodeScout Apptainer rollouts.

So the lowest-risk approach is not to replace `platoon.train.tinker.proxy.TinkerLLM` with `TinkerLiteLLMProvider`.

The reusable cookbook implementation details are:

- `tinker_cookbook.third_party.litellm.provider._convert_openai_messages`
- `tinker_cookbook.third_party.litellm.provider._convert_openai_tools`
- `tinker_cookbook.third_party.litellm.provider._prepare_messages_with_tools`
- `tinker_cookbook.third_party.litellm.provider._sampling_result_to_chat_completion_dict`

These are private underscore helpers, so importing them directly would couple Platoon to non-public cookbook internals. Because `pyproject.toml` pins `tinker-cookbook` to a specific git revision, direct imports are workable in the short term, but they are still brittle.

Preferred compromise:

1. Import public cookbook types and renderer methods directly:

   ```python
   from tinker_cookbook.renderers.base import ToolSpec
   from tinker_cookbook.renderers import ToolCall
   ```

2. Implement the small OpenAI-message/tool conversion shim in Platoon, matching cookbook behavior.

3. Continue using cookbook renderer APIs for the real model-specific work:

   ```python
   renderer.create_conversation_prefix_with_tools(...)
   renderer.build_generation_prompt(...)
   renderer.parse_response(...)
   ```

This uses tinker-cookbook for the important tool-calling implementation while keeping Platoon's RL-specific wrapper intact.

Alternative if we want maximum code reuse:

Create a thin adapter around `TinkerLiteLLMProvider` instead of `TinkerLLM`, inject our sampling client with `provider.set_client(...)`, and wrap/override its sampling pipeline to record `TinkerLLMInteraction`. This is more invasive than adding the cookbook-style message preparation to our current `TinkerLLM`, because the cookbook provider does not expose a public hook for interaction capture.

### 1. Add OpenAI tool conversion helpers

Use stable cookbook public types, not the private cookbook LiteLLM helper functions.

```python
from tinker_cookbook.renderers.base import ToolSpec
```

Add helpers near `_canonicalize_messages(...)`:

```python
def _convert_openai_tools(self, tools: Any) -> list[ToolSpec]:
    if not tools:
        return []
    if not isinstance(tools, list):
        raise ValueError(f"Expected tools to be a list, got {type(tools)}")

    out: list[ToolSpec] = []
    for tool in tools:
        if not isinstance(tool, dict):
            raise ValueError(f"Expected tool dict, got {type(tool)}")
        if tool.get("type") != "function":
            continue
        func = tool.get("function")
        if not isinstance(func, dict):
            raise ValueError("OpenAI function tool missing function object")
        out.append(
            ToolSpec(
                name=func["name"],
                description=func.get("description", ""),
                parameters=func.get("parameters", {}),
            )
        )
    return out
```

### 2. Convert OpenAI-style historical `tool_calls`

`_canonicalize_messages(...)` currently casts messages. That is fragile when OpenHands sends assistant messages with OpenAI dict `tool_calls`. Some cookbook renderers expect `ToolCall` pydantic objects when rerendering assistant history.

Add a normalizer that:

- copies each message dict,
- converts assistant `tool_calls` entries with `TinkerToolCall.model_validate(tc)`,
- preserves `tool_call_id` and `name` on tool result messages,
- leaves string/list content untouched.

This mirrors cookbook `_convert_openai_messages(...)`.

### 3. Inject tool declarations before rendering

Update `_prepare_model_input(...)` to read `tools` from LiteLLM optional params:

```python
def _prepare_model_input(self, **kwargs: Any) -> ModelInput:
    messages = kwargs.pop("messages", None)
    canonical_messages = self._canonicalize_messages(messages)

    optional_params = cast(dict[str, Any], kwargs.get("optional_params", {}))
    tools = optional_params.get("tools")
    tool_choice = optional_params.get("tool_choice")

    if tools and tool_choice != "none":
        tool_specs = self._convert_openai_tools(tools)
        if tool_specs:
            system_prompt = ""
            remaining = list(canonical_messages)
            if remaining and remaining[0]["role"] == "system":
                first = remaining.pop(0)
                content = first.get("content", "")
                system_prompt = content if isinstance(content, str) else self._normalize_message_content(content)

            prefix = self.renderer.create_conversation_prefix_with_tools(tool_specs, system_prompt)
            canonical_messages = prefix + remaining

    return self.renderer.build_generation_prompt(canonical_messages)
```

If a renderer does not implement tool calling, let `NotImplementedError` surface as a clear request failure. That is better than silently hiding tools from the model.

### 4. Return OpenAI-compatible finish reasons

When parsed `tool_calls` are present, the OpenAI-compatible `finish_reason` should be `"tool_calls"` rather than the raw Tinker/vLLM stop reason. Cookbook does this in its LiteLLM provider.

In `_parse_response(...)`, set:

```python
finish_reason = "tool_calls" if tool_calls else seq.stop_reason
```

Then pass `finish_reason=finish_reason` to `Choices(...)`.

This helps OpenHands and LiteLLM treat the response as a tool-call turn.

### 5. Preserve training interaction capture

No special backend changes are needed for training data. `TinkerLLM._record_interaction(...)` records:

- `obs`: the rendered prompt, now including tool declarations and prior tool results,
- `action`: the generated tokens, possibly containing a model-native tool call.

That is what we want. The sampled action remains the model’s raw token sequence.

## Why SkyRL/Tinker Backend Changes Are Not Required

The SkyRL local Tinker backend in `~/skyrl_tinker` exposes a token-oriented sample API:

- `skyrl/tinker/api.py` `SampleRequest`
- `skyrl/tinker/types.py` `SampleInput`
- `skyrl/backends/skyrl_train_backend.py` sample forwarding
- `skyrl/backends/skyrl_train/inference_servers/remote_inference_client.py` `/inference/v1/generate` forwarding

Those paths carry rendered token IDs and sampling parameters. They do not need to know about tools when we use the cookbook renderer pattern.

The only reason to modify the SkyRL/Tinker backend would be if we wanted Tinker itself to expose an OpenAI chat-completions API and delegate tool schema handling to vLLM `/v1/chat/completions`. That is a different design and is not necessary for CodeScout.

## Tests to Add

Add focused unit tests for `platoon/train/tinker/proxy.py`:

1. `tools` are converted and passed into `renderer.create_conversation_prefix_with_tools(...)`.
2. An existing first system message is merged into the renderer prefix, not duplicated.
3. `tool_choice="none"` skips tool prefix injection.
4. Assistant history with OpenAI `tool_calls` is converted to cookbook `ToolCall` objects before rendering.
5. Parsed response tool calls become LiteLLM/OpenAI `message.tool_calls`.
6. Parsed response tool calls set `finish_reason == "tool_calls"`.

For CodeScout, add or run a smoke test with a fake renderer/sampling client that emits a valid Qwen3 tool call block and verify the OpenHands-facing response includes:

```json
{
  "choices": [
    {
      "finish_reason": "tool_calls",
      "message": {
        "role": "assistant",
        "tool_calls": [
          {
            "type": "function",
            "function": {
              "name": "TerminalTool",
              "arguments": "..."
            }
          }
        ]
      }
    }
  ]
}
```

## Expected Result for CodeScout

With `renderer_name: qwen3_instruct`, OpenHands will continue to send OpenAI-compatible `tools` to the FastAPI proxy. Platoon will inject those schemas into the Qwen3 prompt using `tinker-cookbook`, Tinker will sample text tokens as before, Platoon will parse Qwen3 tool-call blocks back into OpenAI `tool_calls`, and OpenHands will execute `TerminalTool` / `LocalizationFinishTool` normally.

The net effect is cookbook-style tool calling with the existing Tinker sample backend.
