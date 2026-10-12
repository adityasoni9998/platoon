"""One sampled math answer becomes one trajectory with an outcome reward."""

import asyncio
import os
from pathlib import Path
from typing import Callable

from platoon.config_defs import RolloutConfig
from platoon.envs.base import Task
from platoon.episode.trajectory import TrajectoryCollection, TrajectoryStep
from platoon.utils.llm_client import LiteLLMClient
from platoon.visualization.event_sinks import JsonlFileSink


async def run_rollout(task: Task, config: RolloutConfig, *, tokenizer, grader: Callable) -> dict:
    client = LiteLLMClient(model=config.model_name, base_url=config.model_endpoint, api_key=config.model_api_key)
    try:
        async with asyncio.timeout(config.timeout):
            params = config.inference_params
            response = await client.async_chat_completion(
                messages=task.misc["messages"],
                temperature=params.temperature,
                top_p=params.top_p,
                max_tokens=params.max_completion_tokens,
            )
            choice = response.choices[0]
            # Parsed chat content can strip </think>, which Miles' grader requires.
            tokens = choice.token_ids
            if not tokens:
                raise ValueError("DAPO rollout requires nonempty token_ids from the Tinker proxy")
            text = tokenizer.decode(tokens, skip_special_tokens=False)
            reward = float(await asyncio.to_thread(grader, text, task.misc["label"]))
    finally:
        await client.aclose()

    collection = TrajectoryCollection()
    events_path = Path(config.output_dir) / "events" / f"events_{task.id}_{collection.id}.jsonl"
    collection.register_event_handlers(
        JsonlFileSink(str(events_path), collection_id=collection.id, process_id=os.getpid())
    )
    trajectory = collection.create_trajectory()
    collection.set_trajectory_task(trajectory.id, task)
    trajectory.reward = reward
    trajectory.misc["truncated"] = choice.finish_reason == "length"
    collection.add_trajectory_step(
        trajectory.id,
        TrajectoryStep(
            misc={
                "action_misc": {"completion_id": response.id},
                "response": text,
                "finish_reason": choice.finish_reason,
            }
        ),
    )
    collection.finish_trajectory(trajectory.id)
    return collection.to_dict()


def reward_processor(trajectory: dict) -> tuple[float, dict]:
    reward = trajectory["reward"]
    return reward, {"reward/correct": reward, "reward/truncated": float(trajectory["misc"]["truncated"])}
