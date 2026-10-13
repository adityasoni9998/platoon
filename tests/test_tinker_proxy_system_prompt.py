"""System instructions must survive OpenAI text-block payloads with tools."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import tinker

from platoon.train.tinker.proxy import TinkerLLM, TinkerLLMProxySession


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
    renderer.create_conversation_prefix_with_tools.side_effect = lambda tools, system_prompt: [
        {"role": "system", "content": system_prompt}
    ]
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
    renderer.build_generation_prompt.assert_called_once_with([{"role": "system", "content": system_prompt}, user])


@pytest.mark.asyncio
async def test_async_prompt_preparation_and_parsing_run_off_loop_and_preserve_sessions():
    loop_thread = threading.get_ident()

    def render(messages):
        assert threading.get_ident() != loop_thread
        return tinker.ModelInput.from_ints([10, int(messages[0]["content"])])

    def parse(tokens):
        assert threading.get_ident() != loop_thread
        return {"role": "assistant", "content": str(tokens[0])}, True

    def batch_decode(sequences):
        assert threading.get_ident() != loop_thread
        return [str(tokens[0]) for tokens in sequences]

    async def sample(*, prompt, **kwargs):
        assert threading.get_ident() == loop_thread
        await asyncio.sleep(0)
        return SimpleNamespace(
            sequences=[SimpleNamespace(tokens=[prompt.to_ints()[-1] + 100], logprobs=[-0.1], stop_reason="stop")]
        )

    renderer = Mock()
    renderer.get_stop_sequences.return_value = []
    renderer.build_generation_prompt.side_effect = render
    renderer.parse_response.side_effect = parse
    tokenizer = Mock()
    tokenizer.batch_decode.side_effect = batch_decode
    provider = TinkerLLM(
        model_name="test",
        renderer=renderer,
        tokenizer=tokenizer,
        sampling_client=SimpleNamespace(sample_async=AsyncMock(side_effect=sample)),
    )

    async def request(number):
        async with TinkerLLMProxySession() as session:
            response = await provider.acompletion(messages=[{"role": "user", "content": str(number)}])
            assert list(session.interactions) == [response.id]
            interaction = session.interactions[response.id]
            assert interaction.obs.to_ints() == [10, number]
            assert interaction.action.tokens == [number + 100]
            assert interaction.action.logprobs == [-0.1]
            return response.id

    ids = await asyncio.gather(request(1), request(2))
    assert len(set(ids)) == 2
