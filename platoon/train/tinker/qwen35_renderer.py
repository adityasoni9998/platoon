"""Qwen3.5 non-thinking renderer that preserves generation prefills in history."""

import json
import re

from tinker_cookbook.renderers import register_renderer
from tinker_cookbook.renderers.base import (
    Message,
    RenderContext,
    TextPart,
    ToolCall,
    ToolSpec,
    UnparsedToolCall,
    parse_response_for_stop_token,
)
from tinker_cookbook.renderers.qwen3_5 import _FUNCTION_BLOCK_RE, Qwen3_5DisableThinkingRenderer

# Upstream's \s* around the value strips indentation from old_str/new_str; keep it.
_PARAM_BLOCK_RE = re.compile(r"<parameter=(?P<name>[^>\n]+)>(?P<value>.*?)</parameter>", re.DOTALL)


def _text(content) -> str:
    return content if isinstance(content, str) else "".join(p["text"] for p in content if p["type"] == "text")


def _decode_value(text: str, schema: dict | None) -> object:
    # As in vLLM/SGLang (and Prime's renderers): declared strings stay verbatim, so "123", "null"
    # and quoted literals are not turned into other types; everything else is JSON when it parses.
    if schema is not None and schema.get("type") in ("string", ["string"]):
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


class Qwen35DisableThinkingRenderer(Qwen3_5DisableThinkingRenderer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # tool name -> parameter name -> JSON schema, from the tools declared in the prompt.
        self._param_schemas: dict[str, dict[str, dict]] = {}

    def create_conversation_prefix_with_tools(self, tools: list[ToolSpec], system_prompt: str = "") -> list[Message]:
        # parse_response() does not receive the tools, so remember their parameter schemas here.
        for tool in tools:
            self._param_schemas[tool["name"]] = (tool.get("parameters") or {}).get("properties") or {}
        # Match the HF template: the system message is |trim'd, and each tool is the OpenAI
        # {"type": "function", "function": ...} envelope serialized by tojson (non-ASCII kept).
        prefix = super().create_conversation_prefix_with_tools(tools, system_prompt.strip())
        if tools:
            flat = "\n".join(json.dumps(tool) for tool in tools)
            wrapped = "\n".join(json.dumps({"type": "function", "function": tool}, ensure_ascii=False) for tool in tools)
            content = prefix[0]["content"]
            assert content.count(flat) == 1, "unexpected upstream Qwen3.5 tool declaration format"
            prefix[0]["content"] = content.replace(flat, wrapped)
        return prefix
    def _assistant_header_suffix(self, message: Message, ctx: RenderContext) -> str:
        # Every generation starts after this prefill. Keep it in earlier turns
        # too, so adding a user message does not invalidate the sampled prefix.
        return "<think>\n\n</think>\n\n"

    def build_generation_prompt(self, messages, role="assistant", prefill=None):
        # Qwen3.5's template puts consecutive tool results in one user turn; the parent opens one per
        # result. Join them so the tool wrapper emits <tool_response> blocks back to back.
        merged: list[Message] = []
        for message in messages:
            if message["role"] == "tool" and merged and merged[-1]["role"] == "tool":
                joined = _text(merged[-1]["content"]) + "\n</tool_response>\n<tool_response>\n" + _text(message["content"])
                merged[-1] = {**merged[-1], "content": joined}
            else:
                merged.append(message)
        return super().build_generation_prompt(merged, role=role, prefill=prefill)

    def _parse_qwen3_5_tool_call_xml(self, raw_text: str) -> ToolCall | UnparsedToolCall:
        match = _FUNCTION_BLOCK_RE.match(raw_text)
        if not match or not match.group("name").strip():
            return UnparsedToolCall(raw_text=raw_text, error="Malformed Qwen3.5 tool call XML")
        body = match.group("body")
        schemas = self._param_schemas.get(match.group("name").strip(), {})
        arguments: dict[str, object] = {}
        pos = 0
        for param in _PARAM_BLOCK_RE.finditer(body):
            name = param.group("name").strip()
            if body[pos : param.start()].strip() or not name:
                return UnparsedToolCall(raw_text=raw_text, error="Malformed Qwen3.5 tool call parameters")
            # Strip only the template's newline on each side of the value.
            value = param.group("value").removeprefix("\n").removesuffix("\n")
            arguments[name] = _decode_value(value, schemas.get(name))
            pos = param.end()
        if body[pos:].strip():
            return UnparsedToolCall(raw_text=raw_text, error="Unexpected trailing content inside <function> block")
        return ToolCall(
            function=ToolCall.FunctionBody(name=match.group("name").strip(), arguments=json.dumps(arguments))
        )

    def _format_tool_call_xml(self, tool_call: ToolCall) -> str:
        args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
        lines = [f"<tool_call>\n<function={tool_call.function.name}>"]
        for name, value in args.items():
            # JSON for non-strings, so true/null round-trip instead of becoming True/None.
            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            lines.append(f"<parameter={name}>\n{text}\n</parameter>")
        lines.append("</function>\n</tool_call>")
        return "\n".join(lines)

    def _format_tool_calls_chunks(self, message: Message) -> list[TextPart]:
        # Upstream always prepends "\n\n", but the parsed content already ends with the
        # model's own separator, so every tool-call turn would otherwise break prefix merging.
        return [TextPart(type="text", text="\n".join(self._format_tool_call_xml(tc) for tc in message["tool_calls"]))]

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        message, parse_success = super().parse_response(response)
        if not parse_success or not message.get("unparsed_tool_calls"):
            return message, parse_success
        # As in qwen3_instruct_exact: if any tool call is malformed, return the whole raw response as
        # text. The proxy drops unparsed calls, so otherwise the model's output would vanish (often
        # leaving an empty reply) and the next prompt would no longer extend the sampled tokens.
        return parse_response_for_stop_token(response, self.tokenizer, self._end_message_token)

    def _postprocess_parsed_message(self, message: Message) -> None:
        super()._postprocess_parsed_message(message)
        content = message.get("content")
        if message.get("tool_calls") and isinstance(content, list):
            # Drop the whitespace between parallel calls; rendering rejoins them with "\n".
            message["content"] = [p for p in content if p["type"] != "text" or p["text"].strip()]


def register_qwen35_renderer() -> None:
    register_renderer(
        "qwen3_5_disable_thinking_preserve_prefix",
        lambda tokenizer, image_processor=None: Qwen35DisableThinkingRenderer(
            tokenizer, image_processor=image_processor, strip_thinking_from_history=False
        ),
    )
