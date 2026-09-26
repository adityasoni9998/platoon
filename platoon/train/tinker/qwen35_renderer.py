"""Qwen3.5 non-thinking renderer that preserves generation prefills in history."""

from tinker_cookbook.renderers import register_renderer
from tinker_cookbook.renderers.base import Message, RenderContext
from tinker_cookbook.renderers.qwen3_5 import Qwen3_5DisableThinkingRenderer


class Qwen35DisableThinkingRenderer(Qwen3_5DisableThinkingRenderer):
    def _assistant_header_suffix(self, message: Message, ctx: RenderContext) -> str:
        # Every generation starts after this prefill. Keep it in earlier turns
        # too, so adding a user message does not invalidate the sampled prefix.
        return "<think>\n\n</think>\n\n"


def register_qwen35_renderer() -> None:
    register_renderer(
        "qwen3_5_disable_thinking_preserve_prefix",
        lambda tokenizer, image_processor=None: Qwen35DisableThinkingRenderer(
            tokenizer, image_processor=image_processor, strip_thinking_from_history=False
        ),
    )
