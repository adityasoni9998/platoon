"""Existing exact-renderer fallback and tool-call behavior."""

import json

from platoon.train.tinker.qwen3_exact_renderer import Qwen3InstructExactRenderer


class TextTokenizer:
    """Small reversible tokenizer for parser tests; no model download needed."""

    def encode(self, text, add_special_tokens=False):
        parts = text.split("<|im_end|>")
        tokens = []
        for index, part in enumerate(parts):
            if index:
                tokens.append(-1)
            tokens.extend(map(ord, part))
        return tokens

    def decode(self, tokens):
        return "".join("<|im_end|>" if token == -1 else chr(token) for token in tokens)


def tool_call(name, arguments):
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}) + "\n</tool_call>"


def test_all_malformed_calls_keep_raw_text_fallback():
    renderer = Qwen3InstructExactRenderer(TextTokenizer())
    response = '<tool_call>\n{"name": "terminal", "arguments": }\n</tool_call>'

    message, success = renderer.parse_response(renderer.tokenizer.encode(response + "<|im_end|>"))

    assert success
    assert not message.get("tool_calls")
    assert message["content"] == response


def test_truncated_response_keeps_existing_failure_behavior():
    renderer = Qwen3InstructExactRenderer(TextTokenizer())
    response = tool_call("terminal", {"command": "ls"})

    message, success = renderer.parse_response(renderer.tokenizer.encode(response))

    assert not success
    assert message["content"] == response


def test_unknown_tool_name_is_forwarded():
    renderer = Qwen3InstructExactRenderer(TextTokenizer())
    response = tool_call("unknown_tool", {"query": "x"})

    message, success = renderer.parse_response(renderer.tokenizer.encode(response + "<|im_end|>"))

    assert success
    assert message["tool_calls"][0].function.name == "unknown_tool"
