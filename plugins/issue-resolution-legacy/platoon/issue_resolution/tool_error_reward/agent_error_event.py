from openhands.sdk.event import AgentErrorEvent

# Return 1.0 if AgentErrorEvent is not present in event stream, otherwise return 0.0
def compute_agent_error_reward(events: list) -> tuple[float, dict]:
    for event in events:
        if isinstance(event, AgentErrorEvent):
            return 0.0, {"agent_error_event_reward": 0.0}
    return 1.0, {"agent_error_event_reward": 1.0}