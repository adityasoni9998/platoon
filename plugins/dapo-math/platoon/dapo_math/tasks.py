"""Load the prompt/label JSONL used by Miles' DAPO PPO example."""

import json
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

from platoon.envs.base import Task


@dataclass
class DataConfig:
    max_samples: int | None = None


def load_tasks(config: DataConfig) -> dict[str, Task]:
    path = hf_hub_download(repo_id="zhuzilin/dapo-math-17k", filename="dapo-math-17k.jsonl", repo_type="dataset")
    if config.max_samples is not None and config.max_samples <= 0:
        raise ValueError("data.max_samples must be positive")
    tasks = {}
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row["prompt"]
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{path}:{line_number}: expected a nonempty prompt")
            if not isinstance(row["label"], (str, float, int)):
                raise ValueError(f"{path}:{line_number}: expected a string or numeric label")
            task_id = f"dapo-math-{line_number}"
            tasks[task_id] = Task(id=task_id, max_steps=1, misc={"messages": messages, "label": row["label"]})
            if config.max_samples is not None and len(tasks) >= config.max_samples:
                break
    if not tasks:
        raise ValueError(f"No DAPO problems found in {path}")
    return tasks
