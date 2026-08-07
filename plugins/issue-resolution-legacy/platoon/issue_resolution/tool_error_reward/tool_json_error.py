from openhands.sdk.conversation import get_agent_final_response

TOOL_ERROR_MARKERS = ["<tool_call>", "</tool_call>"]

# Return 1.0 if agent did not terminate as a result of a JSON/XML format error being interpreted as a message event.
def compute_tool_json_error_reward(events: list):
    agent_final_msg: str | None = get_agent_final_response(events)
    if agent_final_msg is None or agent_final_msg.strip() == "":
        return 1.0, {"tool_json_error_reward": 1.0}
    tool_error_occurred = any(marker in agent_final_msg for marker in TOOL_ERROR_MARKERS)
    reward = 0.0 if tool_error_occurred else 1.0
    return reward, {"tool_json_error_reward": reward}
    