"""Recover terminal dense ORM diagnostics from turn-reward trajectory events.

Final patch scores come from final_settlement, not a sum of turn rewards or
the best intermediate patch. Formatting rewards use the whole conversation,
matching the original ORM's event-based functions without importing the SDK.
"""
import json


def _events(path, audit):
    with path.open() as stream:
        for line in stream:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                audit['malformed_relevant_lines'] += 1


def read_turn_rollout(path, audit):
    task = root = None
    finished = False
    last_info = {}
    last_response = ''
    agent_error = False
    calls = errors = 0
    seen_events = set()
    logged_reward = None
    for event in _events(path, audit):
        kind = event.get('type')
        if kind == 'trajectory_created' and event['trajectory'].get('parent_info') is None:
            root = event['trajectory']['id']
        if root is None or event.get('trajectory_id') != root:
            continue
        if kind == 'trajectory_task_set':
            task = event['task']['id']
        elif kind == 'trajectory_finished':
            finished = True
            logged_reward = event.get('reward')
            if (event.get('error_message') or '').startswith(('LLMServiceUnavailableError:', 'LLMTimeoutError:')):
                audit['llm_infrastructure_error_rollouts'] += 1
        elif kind == 'trajectory_step_added':
            step = event['step']
            misc = step.get('misc') or {}
            if (misc.get('action_misc') or {}).get('completion_id'):
                info = misc.get('reward_misc')
                last_info = info if isinstance(info, dict) else {}
                audit['logged_agent_turns'] += 1
            for item in (step.get('action_events') or []) + (step.get('observation_events') or []):
                event_id = item.get('id')
                if event_id is not None:
                    if event_id in seen_events:
                        audit['duplicate_conversation_events_ignored'] += 1
                        continue
                    seen_events.add(event_id)
                event_kind = item.get('kind')
                if event_kind == 'AgentErrorEvent':
                    agent_error = True
                if event_kind == 'ObservationEvent' and item.get('tool_name') == 'file_editor':
                    calls += 1
                    errors += bool((item.get('observation') or {}).get('is_error', False))
                # Match get_agent_final_response: the last finish-tool action or
                # raw agent MessageEvent, not ordinary tool-call thoughts.
                if item.get('source') == 'agent':
                    if event_kind == 'ActionEvent' and item.get('tool_name') == 'finish':
                        last_response = (item.get('action') or {}).get('message') or ''
                    elif event_kind == 'MessageEvent':
                        content = (item.get('llm_message') or {}).get('content') or []
                        last_response = content if isinstance(content, str) else ''.join(
                            part.get('text', '') if part.get('type') == 'text' else
                            f"[Image: {len(part.get('image_urls', []))} URLs]" if part.get('type') == 'image' else ''
                            for part in content)
    rewards = {'overall_reward': logged_reward}
    if not finished:
        return task, root, False, rewards
    if last_info and last_info.get('reward_mode') != 'turn':
        raise ValueError(f'Expected turn-reward diagnostics: {path}')
    if not last_info or not last_info.get('final_settlement'):
        # Do not silently substitute an intermediate snapshot for the final one.
        audit['missing_final_reward_state_rollouts'] += 1
        return task, root, True, rewards
    audit['rollouts_with_reward_dict'] += 1
    audit['turn_reward_final_states'] += 1
    audit['trajectory_file_editor_calls'] += calls
    audit['trajectory_file_editor_errors'] += errors
    rewards.update(last_info)
    # Deliberately do not merge earlier localization/test dictionaries: an
    # empty final patch must not inherit an earlier nonempty patch's F1 scores.
    rewards.update(
        tool_json_error_reward=float(not any(tag in last_response for tag in ('<tool_call>', '</tool_call>'))),
        agent_error_event_reward=float(not agent_error),
        str_replace_reward=1-errors/calls if calls else 0.0,
    )
    for key in ('tool_json_error_reward', 'agent_error_event_reward', 'str_replace_reward'):
        if rewards[key] != last_info.get(key):
            audit['trajectory_reconstruction_changed_' + key] += 1
    return task, root, True, rewards
