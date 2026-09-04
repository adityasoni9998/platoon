from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PATCHES_PATH = REPO_ROOT / "platoon/train/areal/patches.py"


def _load_patches_module(module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, PATCHES_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _install_fake_areal_client(monkeypatch, *, upstream_fix: bool = False):
    areal_module = types.ModuleType("areal")
    areal_module.__path__ = []
    experimental_module = types.ModuleType("areal.experimental")
    experimental_module.__path__ = []
    openai_module = types.ModuleType("areal.experimental.openai")
    openai_module.__path__ = []
    client_module = types.ModuleType("areal.experimental.openai.client")

    class FakeGenerationHyperparameters:
        def __init__(self, max_new_tokens: int):
            self.max_new_tokens = max_new_tokens
            self.max_tokens = 32768

    class FakeModelRequest:
        def __init__(
            self,
            rid: str = "request",
            input_ids: list[int] | None = None,
            gconfig: FakeGenerationHyperparameters | None = None,
        ):
            self.rid = rid
            self.input_ids = input_ids or []
            self.gconfig = gconfig or FakeGenerationHyperparameters(512)

    class FakeRequestResource:
        def __init__(self, engine_max_tokens: int | None, prompt_length: int, max_new_tokens: int):
            self.engine_max_tokens = engine_max_tokens
            self.prompt_length = prompt_length
            self.max_new_tokens = max_new_tokens

        async def create(self):
            # Exercise task-local isolation by yielding between setting the
            # patch context and constructing the request.
            await asyncio.sleep(0)
            return client_module.ModelRequest(
                input_ids=[0] * self.prompt_length,
                gconfig=FakeGenerationHyperparameters(self.max_new_tokens),
            )

    class FakeCompletions(FakeRequestResource):
        pass

    class FakeResponses(FakeRequestResource):
        pass

    client_module.GenerationHyperparameters = FakeGenerationHyperparameters
    client_module.ModelRequest = FakeModelRequest
    client_module.AsyncCompletionsWithReward = FakeCompletions
    client_module.AsyncResponsesWithReward = FakeResponses
    if upstream_fix:
        client_module._resolve_max_total_tokens = lambda prompt_len, max_new_tokens, cap: min(
            prompt_len + max_new_tokens, cap or 32768
        )

    areal_module.experimental = experimental_module
    experimental_module.openai = openai_module
    openai_module.client = client_module
    monkeypatch.setitem(sys.modules, "areal", areal_module)
    monkeypatch.setitem(sys.modules, "areal.experimental", experimental_module)
    monkeypatch.setitem(sys.modules, "areal.experimental.openai", openai_module)
    monkeypatch.setitem(sys.modules, "areal.experimental.openai.client", client_module)
    return client_module


def test_total_token_patch_covers_both_api_paths_and_is_task_local(monkeypatch):
    patches = _load_patches_module("platoon_areal_total_token_patch_test")
    client_module = _install_fake_areal_client(monkeypatch)

    patches._patch_areal_openai_total_token_limit()
    patches._patch_areal_openai_total_token_limit()

    async def run_requests():
        return await asyncio.gather(
            client_module.AsyncCompletionsWithReward(65000, 64000, 2048).create(),
            client_module.AsyncResponsesWithReward(40000, 30000, 2048).create(),
            client_module.AsyncResponsesWithReward(None, 32000, 2048).create(),
        )

    completion_request, response_request, fallback_request = asyncio.run(run_requests())

    assert completion_request.gconfig.max_tokens == 65000
    assert response_request.gconfig.max_tokens == 32048
    assert fallback_request.gconfig.max_tokens == 32768

    # Direct ModelRequest construction is outside an OpenAI request context and
    # therefore retains AReaL's normal default.
    direct_request = client_module.ModelRequest(
        input_ids=[0] * 64000,
        gconfig=client_module.GenerationHyperparameters(2048),
    )
    assert direct_request.gconfig.max_tokens == 32768


def test_total_token_patch_is_a_noop_when_upstream_fix_exists(monkeypatch):
    patches = _load_patches_module("platoon_areal_total_token_upstream_test")
    client_module = _install_fake_areal_client(monkeypatch, upstream_fix=True)
    original_init = client_module.ModelRequest.__init__
    original_chat_create = client_module.AsyncCompletionsWithReward.create
    original_responses_create = client_module.AsyncResponsesWithReward.create

    patches._patch_areal_openai_total_token_limit()

    assert client_module.ModelRequest.__init__ is original_init
    assert client_module.AsyncCompletionsWithReward.create is original_chat_create
    assert client_module.AsyncResponsesWithReward.create is original_responses_create
