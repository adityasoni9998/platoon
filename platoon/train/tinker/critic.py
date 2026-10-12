"""Critic creation through the unmodified Tinker 0.16.1 SDK.

The high-level ServiceClient factory does not expose model_role. This is the
same lower-level SDK creation path used by Miles' DAPO client; subsequent
operations use the returned standard TrainingClient.
"""

import asyncio

import tinker
from tinker import types
from tinker.lib.client_connection_pool_type import ClientConnectionPoolType


async def create_critic(service: tinker.ServiceClient, base_model: str) -> tinker.TrainingClient:
    holder = service.holder
    model_seq_id = holder.get_training_client_id()

    async def create():
        request = types.CreateModelRequest(
            session_id=holder.get_session_id(),
            model_seq_id=model_seq_id,
            base_model=base_model,
            lora_config=types.LoraConfig(rank=0, train_mlp=True, train_attn=True, train_unembed=True),
        )
        with holder.aclient(ClientConnectionPoolType.TRAIN) as client:
            future = await client.models.create(request=request, extra_body={"model_role": "critic"})
        result = await _model_result(holder, future)
        if not isinstance(result, types.CreateModelResponse):
            raise TypeError(f"Unexpected critic creation response: {type(result)!r}")
        return result.model_id

    model_id = await holder.run_coroutine_threadsafe(create())
    return tinker.TrainingClient(holder, model_seq_id=model_seq_id, model_id=model_id)


async def _model_result(holder, future):
    while True:
        with holder.aclient(ClientConnectionPoolType.TRAIN) as client:
            result = await client.futures.retrieve(request=types.FutureRetrieveRequest(request_id=future.request_id))
        if getattr(result, "type", None) == "try_again":
            await asyncio.sleep(0.2)
            continue
        if isinstance(result, types.RequestFailedResponse):
            raise RuntimeError(result.error)
        return result
