"""Run one end-to-end CodeScout rollout against an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

from platoon.config_defs import InferenceParams, RolloutConfig

from platoon.codescout_rebench.env import multilevel_localization_f1_reward
from platoon.codescout_rebench.rollout import run_rollout
from platoon.codescout_rebench.tasks import get_task

DEFAULT_TASK_ID = "asottile__dead-21"
SYSTEM_PROMPT_MARKER = "specialized code localization agent"


def _find_localization_submission(value) -> list[dict] | None:
    if isinstance(value, dict):
        if value.get("kind") == "LocalizationFinishAction":
            locations = value.get("locations")
            if isinstance(locations, list):
                return locations
        for child in value.values():
            found = _find_localization_submission(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_localization_submission(child)
            if found is not None:
                return found
    return None


def _check_endpoint(endpoint: str, api_key: str) -> None:
    request = Request(
        f"{endpoint.rstrip('/')}/models", headers={"Authorization": f"Bearer {api_key}"}
    )
    with urlopen(request, timeout=30) as response:
        if not 200 <= response.status < 300:
            raise RuntimeError(f"Model endpoint health check returned HTTP {response.status}")


async def _run(args: argparse.Namespace) -> None:
    endpoint = args.endpoint.rstrip("/")
    _check_endpoint(endpoint, args.api_key)
    task = get_task(args.task_id)
    config = RolloutConfig(
        model_name=args.model,
        model_endpoint=endpoint,
        model_api_key=args.api_key,
        max_steps=args.max_steps,
        output_dir=str(args.output_dir),
        verbose=True,
        timeout=args.timeout,
        return_dict=True,
        inference_params=InferenceParams(
            temperature=args.temperature,
            max_completion_tokens=args.max_completion_tokens,
        ),
        extra={"max_input_tokens": args.max_input_tokens},
    )
    result = await run_rollout(task, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / f"{args.task_id}.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    serialized = json.dumps(result)
    if SYSTEM_PROMPT_MARKER not in serialized:
        raise AssertionError("The trajectory did not contain the custom system prompt")
    locations = _find_localization_submission(result)
    if locations is None:
        raise AssertionError("The agent did not call LocalizationFinishAction")
    trajectories = list(result["trajectories"].values())
    if len(trajectories) != 1 or trajectories[0].get("error_message"):
        raise AssertionError("Expected one error-free trajectory")
    steps = trajectories[0]["steps"]
    actions = [event for step in steps for event in (step.get("action_events") or [])]
    if not actions or actions[-1].get("action", {}).get("kind") != "LocalizationFinishAction":
        raise AssertionError("The final action was not the custom finish tool")
    observations = [event for step in steps for event in (step.get("observation_events") or [])]
    if not any(
        event.get("observation", {}).get("kind") == "LocalizationFinishObservation"
        for event in observations
    ):
        raise AssertionError("The remote custom finish executor did not return its observation")
    reward, details = multilevel_localization_f1_reward(task.misc, locations)
    if reward < args.minimum_localization_reward:
        raise AssertionError(
            f"Localization reward {reward} is below "
            f"{args.minimum_localization_reward}: {details}; locations={locations}"
        )

    print(
        json.dumps(
            {
                "task_id": args.task_id,
                "locations": locations,
                "localization_reward": reward,
                "trajectory_reward": trajectories[0]["reward"],
                "num_turns": steps[-1].get("misc", {}).get("reward_misc", {}).get("num_turns"),
                "reward_details": details,
                "result_path": str(result_path),
                "status": "passed",
            },
            indent=2,
            sort_keys=True,
        )
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True, help="Public endpoint ending in /v1")
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "sk-smoke"))
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--task-id", default=DEFAULT_TASK_ID)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-input-tokens", type=int, default=40000)
    parser.add_argument("--max-completion-tokens", type=int, default=2048)
    parser.add_argument("--minimum-localization-reward", type=float, default=3.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/codescout_rebench_smoke"),
    )
    return parser


if __name__ == "__main__":
    asyncio.run(_run(_parser().parse_args()))
