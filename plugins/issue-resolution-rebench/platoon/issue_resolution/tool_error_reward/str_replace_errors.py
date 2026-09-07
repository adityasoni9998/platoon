from openhands.sdk.event import ObservationEvent


def compute_str_replace_reward(events: list):
    file_editor_calls = 0
    file_editor_errors = 0
    for event in events:
        if not isinstance(event, ObservationEvent):
            continue
        if event.tool_name != "file_editor":
            continue
        file_editor_calls += 1
        if getattr(event.observation, "is_error", False):
            file_editor_errors += 1
    fail_fraction = file_editor_errors / file_editor_calls if file_editor_calls else 0.0
    reward = 1.0 - fail_fraction if file_editor_calls else 0.0
    reward_info = {
        "str_replace_reward": reward,
        "file_editor_fail_fraction": fail_fraction,
        "file_editor_calls": file_editor_calls,
        "file_editor_errors": file_editor_errors,
    }
    return reward, reward_info
