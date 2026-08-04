import threading
from typing import Dict, Optional

import numpy as np
from datasets import load_dataset

from platoon.envs.base import Task

EVAL_AGENT_SERVER_IMAGE = "docker.io/adityasoni8/eval-agent-server"
SDK_SHORT_SHA = "43376f1"
ENV_SETUP_COMMANDS = ["export PIP_CACHE_DIR=~/.cache/pip"]
USER_PROMPT_FILENAME = "default.j2"
APPTAINER_CACHE_DIR = "/data/user_data/adityabs/apptainer_cache/"
data_loaded: bool = False
train_data_map: Optional[Dict[str, Task]] = {}
val_data_map: Optional[Dict[str, Task]] = {}
_data_load_lock = threading.Lock()


def create_task_from_instance(x: dict) -> Task:
    task = Task(
        id=x['instance_id'],
        misc=x,
    )
    return task


def load_data():
    global data_loaded, train_data_map, val_data_map
    if data_loaded:
        return train_data_map, val_data_map

    with _data_load_lock:
        if data_loaded:
            return train_data_map, val_data_map

        dataset_localization = load_dataset(
            "adityasoni17/SWE-smith-py-code-search", split="train"
        )
        file_changes_by_id = dict(
            zip(
                dataset_localization["instance_id"],
                dataset_localization["file_changes"],
                strict=True,
            )
        )
        dataset = load_dataset("SWE-bench/SWE-smith-py", split="train")

        np.random.seed(42)
        split_indices = np.random.rand(len(dataset)) < 0.9
        for row_index, source_row in enumerate(dataset):
            instance_id = source_row["instance_id"]
            if instance_id not in file_changes_by_id or not source_row["problem_statement"]:
                continue

            row = dict(source_row)
            row["file_changes"] = file_changes_by_id[instance_id]
            train_data_map[instance_id] = create_task_from_instance(row)
            if not split_indices[row_index]:
                val_data_map[instance_id] = create_task_from_instance(dict(row))

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
