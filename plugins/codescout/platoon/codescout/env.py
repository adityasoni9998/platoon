"""CodeScout localization rewards and OpenHands environment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from openhands.sdk.event import ActionEvent, MessageEvent
from platoon.openhands.env import OpenHandsEnv
from platoon.utils.openhands_utils import is_finished

from platoon.codescout.custom_tools.localization_finish import (
    LocalizationFinishAction,
)


def get_structured_locations(events: Sequence[Any]) -> list[dict] | None:
    """Return the single structured localization submission, if valid."""
    finish_events = [
        event
        for event in events
        if isinstance(event, ActionEvent)
        and event.source == "agent"
        and isinstance(event.action, LocalizationFinishAction)
    ]
    if len(finish_events) != 1:
        return None

    return [
        {
            "file": location.file,
            "class_name": location.class_name,
            "function_name": location.function_name,
        }
        for location in finish_events[0].action.locations
    ]


def count_llm_calls(events: Sequence[Any]) -> int:
    """Count distinct model turns, grouping parallel tool calls together."""
    response_ids = {
        event.llm_response_id
        for event in events
        if (
            isinstance(event, (ActionEvent, MessageEvent))
            and event.source == "agent"
            and getattr(event, "llm_response_id", None) is not None
        )
    }
    return len(response_ids)


def parse_structured_outputs(
    structured_locations: Sequence[dict],
) -> tuple[list[str], list[str], list[str]]:
    """Convert structured locations into file, module, and entity labels."""
    files: set[str] = set()
    modules: set[str] = set()
    entities: set[str] = set()

    for location in structured_locations:
        file_path = location.get("file")
        class_name = location.get("class_name")
        function_name = location.get("function_name")
        if not isinstance(file_path, str) or not file_path.strip():
            return [], [], []

        files.add(file_path)
        if class_name:
            modules.add(f"{file_path}:{class_name}")
        elif function_name:
            modules.add(f"{file_path}:{function_name}")

        if class_name and function_name:
            entities.add(f"{file_path}:{class_name}.{function_name}")
        elif function_name:
            entities.add(f"{file_path}:{function_name}")

    return list(files), list(modules), list(entities)


def compute_f1_score(predicted: Sequence[str], expected: Sequence[str]) -> float:
    """Compute exact-match F1, returning zero for empty ground truth."""
    predicted_set = set(predicted)
    expected_set = set(expected)
    if not expected_set:
        return 0.0

    true_positives = len(predicted_set & expected_set)
    precision = true_positives / len(predicted_set) if predicted_set else 0.0
    recall = true_positives / len(expected_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def multilevel_localization_f1_reward(
    instance: dict,
    structured_locations: list[dict] | None = None,
    file_level_weight: float = 1.0,
    module_level_weight: float = 1.0,
    entity_level_weight: float = 1.0,
) -> tuple[float, dict[str, float]]:
    """Score submitted locations against file/module/entity ground truth."""
    if structured_locations is None:
        return 0.0, {
            "multilevel_localization_f1_reward": 0.0,
            "file_reward": 0.0,
            "module_reward": 0.0,
            "entity_reward": 0.0,
        }

    expected_files: set[str] = set()
    expected_modules: set[str] = set()
    expected_entities: set[str] = set()
    for file_change in instance.get("file_changes", []):
        if file_change.get("file"):
            expected_files.add(file_change["file"])
        changes = file_change.get("changes") or {}
        expected_modules.update(changes.get("edited_modules") or [])
        expected_entities.update(changes.get("edited_entities") or [])

    predicted_files, predicted_modules, predicted_entities = parse_structured_outputs(
        structured_locations
    )
    file_score = compute_f1_score(predicted_files, expected_files)
    module_score = compute_f1_score(predicted_modules, expected_modules)
    entity_score = compute_f1_score(predicted_entities, expected_entities)
    reward = (
        file_score * file_level_weight
        + module_score * module_level_weight
        + entity_score * entity_level_weight
    )
    return reward, {
        "multilevel_localization_f1_reward": reward,
        "file_reward": file_score,
        "module_reward": module_score,
        "entity_reward": entity_score,
    }


class CodeScoutEnv(OpenHandsEnv):
    """OpenHands environment with CodeScout's localization reward."""

    async def evaluate(self) -> tuple[float, dict]:
        if not is_finished(await self.observe()):
            return 0.0, {}

        events = self._conversation.state.events
        structured_locations = get_structured_locations(events)
        num_turns = count_llm_calls(events)
        turn_limit_reward = 1.0 if num_turns == 4 else 0.0
        metadata: dict[str, float | int] = {"num_turns": num_turns}
        if structured_locations is None:
            return turn_limit_reward, metadata

        localization_reward, localization_metadata = multilevel_localization_f1_reward(
            self.task.misc,
            structured_locations,
        )
        metadata.update(localization_metadata)
        return turn_limit_reward + localization_reward, metadata
