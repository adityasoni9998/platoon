import hashlib
import threading
from typing import Dict, Optional

import numpy as np
from datasets import load_dataset
from platoon.envs.base import Task

from platoon.issue_resolution.easy_instances import EASY_SWE_SMITH_INSTANCE_IDS

EVAL_AGENT_SERVER_IMAGE = "docker.io/adityasoni8/eval-agent-server"
EVAL_NAMED_AGENT_SERVER_IMAGE_REPOSITORY = EVAL_AGENT_SERVER_IMAGE.replace("/", "__")
SDK_GIT_SHA = "5acdf05ae2fc6224f92114db6f557b181cd0f280"
SDK_SHORT_SHA = SDK_GIT_SHA[:7]
USER_PROMPT_FILENAME = "default.j2"
NUM_RETRIES_SANDBOX_START = 3
MAX_MODAL_IMAGE_NAME_LENGTH = 64
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


def _truncate_modal_image_component(value: str, replacements: tuple[tuple[str, str], ...]) -> str:
    """Match the naming used when the SWE-Smith images were published to Modal."""
    if len(value) <= MAX_MODAL_IMAGE_NAME_LENGTH:
        return value
    for old, new in replacements:
        value = value.replace(old, new)
        if len(value) <= MAX_MODAL_IMAGE_NAME_LENGTH:
            return value
    digest = hashlib.sha256(value.encode()).hexdigest()[:8]
    return f"{value[: MAX_MODAL_IMAGE_NAME_LENGTH - 9]}-{digest}"


def named_agent_server_image_for_instance(instance: dict) -> str:
    """Return the published Modal Image name for a SWE-Smith base image."""
    base_image = str(instance["image_name"]).lower().strip()
    custom_tag = base_image.rsplit("/", 1)[-1].split(":", 1)[0]
    tag = f"{SDK_SHORT_SHA}-{custom_tag}-source-minimal"
    repository = _truncate_modal_image_component(
        EVAL_NAMED_AGENT_SERVER_IMAGE_REPOSITORY,
        (),
    )
    tag = _truncate_modal_image_component(
        tag,
        ((".x86_64", ""), ("-source-minimal", "-src-min")),
    )
    return f"{repository}:{tag}"


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
