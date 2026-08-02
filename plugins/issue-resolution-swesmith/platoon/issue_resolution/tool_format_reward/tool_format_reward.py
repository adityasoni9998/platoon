from platoon.issue_resolution.tool_format_reward.file_editor_reward import compute_file_editor_reward
from platoon.issue_resolution.tool_format_reward.agent_error_reward import compute_agent_error_reward

def compute_tool_format_reward(events: list) -> tuple[float, dict]:
    file_editor_reward, file_editor_reward_info = compute_file_editor_reward(events)
    agent_error_reward, agent_error_reward_info = compute_agent_error_reward(events)

    # Combine the rewards via weighted average.
    tool_format_reward = 0.8 * file_editor_reward + 0.2 * agent_error_reward
    tool_format_info = {**file_editor_reward_info, **agent_error_reward_info}
    tool_format_info["tool_format_reward"] = tool_format_reward

    return tool_format_reward, tool_format_info