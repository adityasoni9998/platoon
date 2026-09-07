from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Mapping
from typing import Dict, Optional

import numpy as np
from datasets import load_dataset
from platoon.envs.base import Task

DATASET_NAME = "adityasoni17/SWE-rebench"
DATASET_SPLIT = "filtered"

EVAL_AGENT_SERVER_IMAGE = "docker.io/adityasoni8/eval-agent-server"
SDK_GIT_SHA = "5acdf05ae2fc6224f92114db6f557b181cd0f280"
SDK_SHORT_SHA = SDK_GIT_SHA[:7]
SWEREBENCH_REPAIR_VERSION = "swerebench-testbed-reinstall-v2"
USER_PROMPT_FILENAME = "default.j2"
NUM_RETRIES_SANDBOX_START = 3
MAX_DOCKER_CUSTOM_TAG_LENGTH = 64
MAX_MODAL_IMAGE_NAME_LENGTH = 64

data_loaded: bool = False
train_data_map: Optional[Dict[str, Task]] = {}
val_data_map: Optional[Dict[str, Task]] = {}
_data_load_lock = threading.Lock()


def create_task_from_instance(instance: dict) -> Task:
    return Task(id=instance["instance_id"], misc=instance)


def _install_command(instance: Mapping[str, object]) -> str:
    install_config = instance.get("install_config")
    if isinstance(install_config, str):
        install_config = json.loads(install_config)
    if not isinstance(install_config, Mapping):
        raise ValueError("SWE-rebench row is missing install_config")
    install_command = install_config.get("install")
    if not isinstance(install_command, str) or not install_command.strip():
        raise ValueError("SWE-rebench row is missing install_config.install")
    return install_command


def _sanitize_image_component(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-" else "_" for character in value
    )


def _docker_custom_tag(instance: Mapping[str, object]) -> str:
    base_image = instance.get("docker_image")
    if not isinstance(base_image, str) or not base_image:
        raise ValueError("SWE-rebench row is missing a non-empty docker_image field")

    repair_hash = hashlib.sha256(
        json.dumps(
            {
                "version": SWEREBENCH_REPAIR_VERSION,
                "install": _install_command(instance),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:7]
    custom_tag = (
        f"{_sanitize_image_component(base_image)}-{SWEREBENCH_REPAIR_VERSION}-{repair_hash}"
    )
    if len(custom_tag) <= MAX_DOCKER_CUSTOM_TAG_LENGTH:
        return custom_tag
    digest = hashlib.sha256(custom_tag.encode()).hexdigest()[:12]
    prefix_length = MAX_DOCKER_CUSTOM_TAG_LENGTH - len(digest) - 1
    return f"{custom_tag[:prefix_length]}-{digest}"


def _truncate_modal_component(
    value: str,
    replacements: tuple[tuple[str, str], ...],
) -> str:
    if len(value) <= MAX_MODAL_IMAGE_NAME_LENGTH:
        return value
    for old, new in replacements:
        value = value.replace(old, new)
        if len(value) <= MAX_MODAL_IMAGE_NAME_LENGTH:
            return value
    digest = hashlib.sha256(value.encode()).hexdigest()[:8]
    return f"{value[: MAX_MODAL_IMAGE_NAME_LENGTH - 9]}-{digest}"


def named_agent_server_image_for_instance(instance: Mapping[str, object]) -> str:
    """Return the Modal name published by the swerebench_modal image builder."""
    repository = EVAL_AGENT_SERVER_IMAGE
    if ":" in repository.rsplit("/", 1)[-1]:
        repository = repository.rsplit(":", 1)[0]
    repository = _truncate_modal_component(repository.replace("/", "__"), ())
    tag = f"{SDK_SHORT_SHA}-{_docker_custom_tag(instance)}-source-minimal"
    tag = _truncate_modal_component(
        tag,
        ((".x86_64", ""), ("-source-minimal", "-src-min")),
    )
    return f"{repository}:{tag}"


def load_data() -> tuple[dict[str, Task], dict[str, Task]]:
    global data_loaded, train_data_map, val_data_map
    if data_loaded:
        return train_data_map, val_data_map

    with _data_load_lock:
        if data_loaded:
            return train_data_map, val_data_map

        dataset = load_dataset(
            DATASET_NAME,
            split=DATASET_SPLIT,
        )
        rng = np.random.RandomState(42)
        validation_rows = rng.rand(len(dataset)) >= 0.9
        loaded_train: dict[str, Task] = {}
        loaded_validation: dict[str, Task] = {}

        max_instances = int(os.environ.get("SWEREBENCH_MAX_INSTANCES", "0"))
        for row_index, source_row in enumerate(dataset):
            instance = dict(source_row)
            instance_id = instance.get("instance_id")
            if not isinstance(instance_id, str) or not instance_id:
                continue
            problem_statement = instance.get("problem_statement")
            if not isinstance(problem_statement, str) or not problem_statement.strip():
                continue

            # Validate the fields that identify the already-published Modal image
            # while loading, instead of failing after a rollout worker starts.
            named_agent_server_image_for_instance(instance)
            task = create_task_from_instance(instance)
            loaded_train[instance_id] = task
            if validation_rows[row_index]:
                loaded_validation[instance_id] = create_task_from_instance(dict(instance))
            if max_instances > 0 and len(loaded_train) >= max_instances:
                break

        if not loaded_train:
            raise RuntimeError("No usable SWE-rebench instances were loaded")

        train_data_map = loaded_train
        val_data_map = loaded_validation
        data_loaded = True
        print(
            f"Loaded {len(train_data_map)} training instances and "
            f"{len(val_data_map)} validation instances.",
            flush=True,
        )
        return train_data_map, val_data_map


def get_task(task_id: str) -> Task:
    train_map, validation_map = load_data()
    if task_id in train_map:
        return train_map[task_id]
    if task_id in validation_map:
        return validation_map[task_id]
    raise ValueError(f"Task ID {task_id} not found in training or validation data.")
