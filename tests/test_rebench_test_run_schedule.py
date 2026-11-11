"""Source hashes of turn snapshots, and the interval schedule of the test-run reward."""

from types import SimpleNamespace

import pytest
from openhands.sdk.event import ActionEvent, ObservationEvent
from openhands.sdk.llm import MessageToolCall
from openhands.tools.terminal import TerminalAction
from platoon.issue_resolution.test_run_reward.schedule import (
    EMPTY_SOURCE_HASH,
    interval_rewards,
    schedule_test_rewards,
    source_hash,
)

from platoon.utils.openhands_utils import group_actions

BASE = {"patch": "diff", "before_sources": {"pkg/a.py": "old"}, "after_sources": {"pkg/a.py": "new"}}


def with_files(before=None, after=None, errors=None):
    return {"patch": "diff", "before_sources": {**BASE["before_sources"], **(before or {})},
            "after_sources": {**BASE["after_sources"], **(after or {})}, "source_errors": errors or {}}


@pytest.mark.parametrize("record", [
    with_files(after={"reproduce_issue.py": "x"}, errors={"base:reproduce_issue.py": "missing"}),
    with_files(after={"debug/check_fix.py": "x"}, errors={"base:debug/check_fix.py": "missing"}),
    with_files(before={"tests/test_a.py": "old"}, after={"tests/test_a.py": "new"}),
    with_files(after={"tests/test_issue.py": "x"}, errors={"base:tests/test_issue.py": "missing"}),
    with_files(after={"pkg/conftest.py": "x"}, errors={"base:pkg/conftest.py": "missing"}),
    with_files(after={"out.json": "{}"}, errors={"base:out.json": "missing"}),
    with_files(errors={"base:pkg/__pycache__/a.pyc": "missing", ":pkg/__pycache__/a.pyc": "binary"}),
])
def test_tests_scratch_files_and_caches_do_not_change_the_source_hash(record):
    assert source_hash(record) == source_hash(BASE)


@pytest.mark.parametrize("record", [
    with_files(after={"pkg/new_module.py": "x"}, errors={"base:pkg/new_module.py": "missing"}),
    with_files(before={"pkg/b.py": "gone"}, errors={":pkg/b.py": "deleted"}),
    with_files(before={"examples/demo.py": "old"}, after={"examples/demo.py": "new"}),  # existing file
    {**BASE, "after_sources": {"pkg/a.py": "newer"}},
])
def test_source_edits_change_the_source_hash(record):
    assert source_hash(record) != source_hash(BASE)


def test_empty_or_missing_snapshots_have_the_empty_source_hash():
    assert source_hash(None) == source_hash({"patch": ""}) == EMPTY_SOURCE_HASH
    assert source_hash(BASE) != EMPTY_SOURCE_HASH


@pytest.mark.parametrize("turns, expected", [
    # No source edit at all: no reward, however much the agent tests.
    ([("t1", "", True), ("t2", "", False), ("t3", "", True)], set()),
    # Reproduce, edit, verify, edit again, verify again.
    ([("t1", "", True), ("t2", "", True), ("t3", "A", False), ("t4", "A", True), ("t5", "B", False),
      ("t6", "B", True)], {"t2", "t4", "t6"}),
    # Repeated tests in one interval earn once; an interval without a test earns nothing.
    ([("t1", "", False), ("t2", "A", True), ("t3", "A", True), ("t4", "B", False)], {"t3"}),
    # A test in the editing turn verifies that edit; reverting the source is an edit too.
    ([("t1", "A", True), ("t2", "A", False)], {"t1"}),
    ([("t1", "A", True), ("t2", "", True)], {"t1", "t2"}),
    ([], set()),
])
def test_test_run_reward_intervals(turns, expected):
    hashes = {"": EMPTY_SOURCE_HASH}
    turns = [(cid, hashes.get(h, h), meaningful) for cid, h, meaningful in turns]
    assert interval_rewards(turns, 0.05) == {cid: 0.05 for cid in expected}


def terminal_turn(completion, command, output, exit_code=0):
    call = f"{completion}-call"
    action = ActionEvent(
        thought=[], action=TerminalAction(command=command), tool_name="terminal", tool_call_id=call,
        llm_response_id=completion,
        tool_call=MessageToolCall(id=call, name="terminal", arguments="{}", origin="completion"),
    )
    result = SimpleNamespace(content=[SimpleNamespace(text=output)], exit_code=exit_code, timeout=False)
    observation = ObservationEvent.model_construct(
        tool_name="terminal", tool_call_id=call, action_id=action.id, observation=result)
    return [action, observation]


def test_schedule_from_conversation_events():
    passed = "=== 2 passed in 0.10s ==="
    events = [
        *terminal_turn("reproduce", "pytest tests/test_a.py", "=== 1 failed in 0.10s ===", 1),
        *terminal_turn("edit", "sed -i s/old/new/ pkg/a.py", ""),
        *terminal_turn("verify", "cd /testbed && python -m pytest tests/test_a.py", passed),
        *terminal_turn("write-repro", "cat > reproduce.py <<'EOF'\nimport pkg\nEOF", ""),
        *terminal_turn("verify-again", "pytest tests", passed),
    ]
    _, action_to_completion, _, _ = group_actions(events)
    completion_snapshots = {"edit": "p1", "write-repro": "p2"}
    records = {"p1": BASE, "p2": with_files(after={"reproduce.py": "import pkg"},
                                            errors={"base:reproduce.py": "missing"})}
    rewards, details = schedule_test_rewards(
        events, action_to_completion, completion_snapshots, records, 0.05)
    # The repro script is not a source edit, so the two verifications share one interval.
    assert rewards == {"reproduce": 0.05, "verify-again": 0.05}
    assert [details[c]["meaningful_test_run"] for c in details] == [True, False, True, False, True]
    assert details["verify"]["source_hash"] == details["verify-again"]["source_hash"] == source_hash(BASE)
    assert [c for c in details if details[c]["source_changed"]] == ["edit"]
