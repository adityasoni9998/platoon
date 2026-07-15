from openhands.sdk.event import ObservationEvent

def compute_str_replace_reward(events: list):
    reward = 1.0
    for event in events:
        if not isinstance(event, ObservationEvent):
            continue
        if event.tool_name != "file_editor":
            continue
        if getattr(event.observation, "is_error", False):
            reward = 0.0
            break
    reward_info = {"str_replace_reward": reward}
    return reward, reward_info