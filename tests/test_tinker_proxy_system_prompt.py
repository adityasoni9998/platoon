"""System instructions must survive OpenAI text-block payloads with tools."""

from unittest.mock import Mock

import pytest

from platoon.train.tinker.proxy import TinkerLLM


@pytest.mark.parametrize(
    "content",
    [
        "Search the repository.\nUse four turns.",
        [
            {"type": "text", "text": "Search the repository.\n"},
            {
                "type": "text",
                "text": "Use four turns.",
                "cache_control": {"type": "ephemeral"},
            },
        ],
    ],
)
def test_tool_prompt_preserves_system_instructions(content):
    renderer = Mock()
    renderer.create_conversation_prefix_with_tools.side_effect = (
        lambda tools, system_prompt: [{"role": "system", "content": system_prompt}]
    )
    provider = TinkerLLM(
        model_name="test",
        renderer=renderer,
        tokenizer=Mock(),
        sampling_client=Mock(),
    )
    user = {"role": "user", "content": [{"type": "text", "text": "Find the bug."}]}
    tool = {
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "Search code",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    provider._prepare_model_input(
        messages=[{"role": "system", "content": content}, user],
        optional_params={"tools": [tool]},
    )

    specs, system_prompt = renderer.create_conversation_prefix_with_tools.call_args.args
    assert system_prompt == "Search the repository.\nUse four turns."
    assert specs[0]["name"] == "terminal"
    renderer.build_generation_prompt.assert_called_once_with(
        [{"role": "system", "content": system_prompt}, user]
    )
