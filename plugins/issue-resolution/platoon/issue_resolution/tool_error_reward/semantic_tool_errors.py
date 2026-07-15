from openhands.sdk.conversation import get_agent_final_response
from openhands.sdk.event import AgentErrorEvent, MessageEvent, ObservationEvent
from openhands.sdk.llm.message import content_to_str


TOOL_CALL_MARKERS = ("<tool_call>", "</tool_call>")


def _event_text(event) -> str:
    if isinstance(event, MessageEvent):
        return "".join(content_to_str(event.llm_message.content))

    observation = getattr(event, "observation", None)
    content = getattr(observation, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(content_to_str(content))


def has_str_replace_issue(events: list) -> bool:
    for event in events:
        if not isinstance(event, ObservationEvent):
            continue
        if event.tool_name != "file_editor":
            continue
        if getattr(event.observation, "is_error", False):
            return True
    return False


def has_terminal_exit_or_command_not_found(events: list) -> bool:
    for event in events:
        if not isinstance(event, ObservationEvent):
            continue
        if event.tool_name != "terminal":
            continue

        observation = event.observation
        exit_code = getattr(observation, "exit_code", None)
        if exit_code not in (None, 0):
            return True

        if "command not found" in _event_text(event).lower():
            return True
    return False


def has_agent_error_event(events: list) -> bool:
    return any(isinstance(event, AgentErrorEvent) for event in events)


def has_tool_json_format_error(events: list) -> bool:
    agent_final_msg = get_agent_final_response(events)
    if agent_final_msg is None or agent_final_msg.strip() == "":
        return False
    return any(marker in agent_final_msg for marker in TOOL_CALL_MARKERS)


def detect_semantic_tool_errors(state) -> dict[str, bool]:
    events = state.conversation_state.events
    return {
        "str_replace_issue": has_str_replace_issue(events),
        "terminal_exit_or_command_not_found": has_terminal_exit_or_command_not_found(events),
        "agent_error_event": has_agent_error_event(events),
        "tool_json_format_error": has_tool_json_format_error(events),
    }
