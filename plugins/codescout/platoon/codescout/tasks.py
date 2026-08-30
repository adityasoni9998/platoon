from typing import Dict, Optional

import numpy as np
from datasets import load_dataset
from platoon.envs.base import Task

EVAL_NAMED_AGENT_SERVER_IMAGE = (
    "docker.io__adityasoni8__codescout-agent-server-modal-workspace:29fafa4b"
)
USER_PROMPT_FILENAME = "user_prompt.j2"
NUM_RETRIES_SANDBOX_START = 3
data_loaded: bool = False
train_data_map: Optional[Dict[str, Task]] = {}
val_data_map: Optional[Dict[str, Task]] = {}


def create_task_from_instance(x: dict) -> Task:
    task = Task(
        id=x["instance_id"],
        misc=x,
    )
    return task


def load_data():
    global data_loaded, train_data_map, val_data_map
    if data_loaded:
        return train_data_map, val_data_map

    # Iterate over the Hugging Face dataset directly. Converting it to pandas
    # turns nested lists such as edited_modules and edited_entities into NumPy
    # arrays, which cannot be used with boolean expressions like ``value or []``
    # during localization reward calculation.
    dataset = load_dataset("adityasoni17/SWE-smith-py-code-search", split="train")
    np.random.seed(42)
    split_indices = np.random.rand(len(dataset)) < 0.9
    for row_index, source_row in enumerate(dataset):
        if not source_row["problem_statement"]:
            continue

        row = dict(source_row)
        train_data_map[row["instance_id"]] = create_task_from_instance(row)
        if not split_indices[row_index]:
            val_data_map[row["instance_id"]] = create_task_from_instance(dict(row))
    data_loaded = True
    print(
        f"Loaded {len(train_data_map)} training instances and "
        f"{len(val_data_map)} validation instances.",
        flush=True,
    )
    return train_data_map, val_data_map


def get_task(task_id: str) -> Task:
    load_data()
    global train_data_map, val_data_map
    if task_id in train_data_map:
        return train_data_map[task_id]
    elif task_id in val_data_map:
        return val_data_map[task_id]
    else:
        raise ValueError(f"Task ID {task_id} not found in training or validation data.")
