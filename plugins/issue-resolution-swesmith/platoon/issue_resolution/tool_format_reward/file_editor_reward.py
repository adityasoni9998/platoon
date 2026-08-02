from openhands.sdk.event import ObservationEvent

def compute_file_editor_reward(events: list):
    reward = 1.0
    edit_attempted = False
    for event in events:
        if not isinstance(event, ObservationEvent):
            continue
        if event.tool_name != "file_editor":
            continue
        if getattr(event.observation, "command", None) in ["insert", "str_replace"]:
            edit_attempted = True
        if getattr(event.observation, "is_error", False):
            reward = 0.0
            break
    if not edit_attempted:
        reward = 0.0
    reward_info = {
        "file_editor_reward": reward,
        "file_editor_edit_attempted": edit_attempted,
    }
    return reward, reward_info
