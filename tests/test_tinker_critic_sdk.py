"""Exercise unmodified SDK serialization, futures, and critic creation locally."""

import asyncio
import json

import httpx
import pytest
import tinker

from platoon.train.tinker.critic import create_critic
from platoon.train.tinker.ppo import save_pair


@pytest.mark.asyncio
async def test_standard_sdk_critic_lifecycle(monkeypatch, tmp_path):
    monkeypatch.setenv("TINKER_TELEMETRY", "0")
    requests, results = [], {}
    pending_critic_creation = True

    async def handle(request):
        nonlocal pending_critic_creation
        route = request.url.path.rsplit("/", 1)[-1]
        body = json.loads(request.content) if request.content else {}
        requests.append((route, body))
        if route == "create_session":
            return httpx.Response(200, json={"session_id": "session"})
        if route == "session_heartbeat":
            return httpx.Response(200, json={})
        if route == "create_sampling_session":
            return httpx.Response(200, json={"sampling_session_id": "sampling"})
        if route == "retrieve_future":
            result = results[body["request_id"]]
            if result.get("model_id") == "critic" and pending_critic_creation:
                pending_critic_creation = False
                return httpx.Response(
                    200,
                    json={
                        "type": "try_again",
                        "request_id": body["request_id"],
                        "queue_state": "active",
                    },
                )
            return httpx.Response(200, json=result)
        if route == "create_model":
            result = {"type": "create_model", "model_id": body.get("model_role", "policy")}
        elif route in ("forward", "forward_backward"):
            assert body["model_id"] == "critic"
            payload = body[route + "_input"]
            assert payload["loss_fn"] == "ppo"
            result = {
                "loss_fn_output_type": "ArrayRecord",
                "loss_fn_outputs": [{"values": {"data": [0.2, 0.4], "dtype": "float32", "shape": [2]}}],
                "metrics": {"loss:sum": 0.18},
            }
        elif route == "optim_step":
            result = {"type": "optim_step", "metrics": {"skipped": 0.0}}
        elif route == "save_weights":
            result = {"type": "save_weights", "path": f"tinker://{body['model_id']}/weights/{body['path']}"}
        elif route == "save_weights_for_sampler":
            assert body["model_id"] == "policy"
            result = {"type": "save_weights_for_sampler", "path": f"tinker://policy/sampler_weights/{body['path']}"}
        elif route == "load_weights":
            assert body["optimizer"] is True
            result = {"type": "load_weights", "path": body["path"]}
        else:
            raise AssertionError(f"Unexpected SDK route: {route}")
        request_id = str(len(results))
        results[request_id] = result
        return httpx.Response(200, json={"request_id": request_id})

    transport = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    service = tinker.ServiceClient(base_url="http://sdk-test", api_key="tml-test", http_client=transport)
    try:
        actor = await service.create_lora_training_client_async("test", rank=0)
        critic = await asyncio.wait_for(create_critic(service, "test"), timeout=10)
        assert type(actor) is type(critic) is tinker.TrainingClient
        row = tinker.Datum(
            model_input=tinker.ModelInput.from_ints([10, 20]),
            loss_fn_inputs={"target_tokens": tinker.TensorData(data=[20, 30], dtype="int64", shape=[2])},
        )
        result = await (await critic.forward_async([row], loss_fn="ppo")).result_async()
        assert result.loss_fn_outputs[0]["values"].data == [0.2, 0.4]
        for key, values in {"values": [0.2, 0.4], "returns": [0.0, 1.0], "weights": [0.0, 0.5]}.items():
            row.loss_fn_inputs[key] = tinker.TensorData(data=values, dtype="float32", shape=[2])
        await (
            await critic.forward_backward_async([row], loss_fn="ppo", loss_fn_config={"value_clip": 0.2})
        ).result_async()
        await (await critic.optim_step_async(tinker.AdamParams(learning_rate=1e-5))).result_async()
        saved = await (await critic.save_state_async("checkpoint")).result_async()
        await (await critic.load_state_with_optimizer_async(saved.path)).result_async()
        creates = [body for route, body in requests if route == "create_model"]
        assert creates[0].get("model_role", "policy") == "policy"
        assert creates[1]["model_role"] == "critic" and creates[1]["lora_config"]["rank"] == 0
        assert creates[0]["model_seq_id"] != creates[1]["model_seq_id"]
        backward = next(body["forward_backward_input"] for route, body in requests if route == "forward_backward")
        fields = backward["data"][0]["loss_fn_inputs"]
        assert set(fields) == {"target_tokens", "values", "returns", "weights"}
        assert fields["weights"]["data"] == [0.0, 0.5]
        assert backward["loss_fn_config"] == {"value_clip": 0.2}
        sequences = [
            body["seq_id"]
            for route, body in requests
            if route in ("forward", "forward_backward", "optim_step", "save_weights", "load_weights")
        ]
        assert sequences == sorted(set(sequences)) and len(sequences) == 5
        state = await save_pair(actor, critic, tmp_path, batch=10, wandb_run_id="run")
        assert state["paths"] == {
            "critic": "tinker://critic/weights/000010", "policy": "tinker://policy/weights/000010"
        }
        assert state["sampler_path"] == "tinker://policy/sampler_weights/000010"
        sampling_client = actor.create_sampling_client(state["sampler_path"])
        assert isinstance(sampling_client, tinker.SamplingClient)
        session = next(body for route, body in requests if route == "create_sampling_session")
        assert session["model_path"] == state["sampler_path"]
    finally:
        service.holder.close()
        await transport.aclose()
