import threading
from typing import Dict, Optional

import numpy as np
from datasets import load_dataset
from platoon.envs.base import Task

from platoon.issue_resolution_sail.easy_instances import EASY_SWE_SMITH_INSTANCE_IDS

EVAL_AGENT_SERVER_IMAGE = "docker.io/adityasoni8/eval-agent-server"
SDK_GIT_SHA = "d3fd1c497860fdcba49e3304f348b10a4f1c370b"
SDK_SHORT_SHA = SDK_GIT_SHA[:7]
USER_PROMPT_FILENAME = "default.j2"
NUM_RETRIES_SANDBOX_START = 3
data_loaded: bool = False
train_data_map: Optional[Dict[str, Task]] = {}
val_data_map: Optional[Dict[str, Task]] = {}
_data_load_lock = threading.Lock()


def create_task_from_instance(x: dict) -> Task:
    task = Task(
        id=x["instance_id"],
        misc=x,
    )
    return task


def agent_server_image_for_instance(instance: dict) -> str:
    """Registry tag emitted by the easy-subset Modal image builder."""
    base_image = str(instance["image_name"]).lower().strip()
    custom_tag = base_image.rsplit("/", 1)[-1].split(":", 1)[0]
    return f"{EVAL_AGENT_SERVER_IMAGE}:{SDK_SHORT_SHA}-{custom_tag}-source-minimal"



def load_data():
    global data_loaded, train_data_map, val_data_map
    if data_loaded:
        return train_data_map, val_data_map

    with _data_load_lock:
        if data_loaded:
            return train_data_map, val_data_map

        dataset_localization = load_dataset("adityasoni17/SWE-smith-py-code-search", split="train")
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
            if (
                instance_id not in EASY_SWE_SMITH_INSTANCE_IDS
                or instance_id not in file_changes_by_id
                or not source_row["problem_statement"]
            ):
                continue

            row = dict(source_row)
            row["file_changes"] = file_changes_by_id[instance_id]
            train_data_map[instance_id] = create_task_from_instance(row)
            if not split_indices[row_index]:
                val_data_map[instance_id] = create_task_from_instance(dict(row))

        missing_ids = EASY_SWE_SMITH_INSTANCE_IDS.difference(train_data_map)
        if missing_ids:
            raise RuntimeError(
                "The legacy easy SWE-Smith subset is incomplete; missing "
                f"{len(missing_ids)} instance(s), including {sorted(missing_ids)[:5]}"
            )

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
