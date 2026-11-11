"""Detecting turns that run tests, and whether their outputs are meaningful (outputs from real rollouts)."""

from types import SimpleNamespace

import pytest
from platoon.issue_resolution.test_run_reward.detect import (
    classify_invocation,
    detect_test_runs,
    turn_runs_meaningful_test,
)

PYTEST_PASS = "collected 3 items\n\ntests/test_x.py ...  [100%]\n\n============ 3 passed in 0.12s ============"
PYTEST_FAIL = "tests/test_x.py .F\n=== FAILURES ===\n...\n=== 1 failed, 1 passed, 2 warnings in 0.40s ==="
PYTEST_QUIET = ".F\n1 failed, 1 passed in 0.31s"
COLLECTION = (
    "==== ERRORS ====\n_ ERROR collecting tests/test_x.py _\n"
    "ImportError while importing test module\n=== 1 error in 0.71s ==="
)
NOT_FOUND = "=== no tests ran in 0.00s ===\nERROR: file or directory not found: tests/test_missing.py"
UNITTEST_OK = "..........\n" + "-" * 70 + "\nRan 10 tests in 2.861s\n\nOK"
UNITTEST_NONE = "\n" + "-" * 70 + "\nRan 0 tests in 0.000s\n\nOK"
REPO_TRACEBACK = (
    'Traceback (most recent call last):\n  File "/testbed/reproduce_issue.py", line 3, in <module>\n    parse("x")\n'
    '  File "/testbed/sqlglot/parser.py", line 120, in parse\n    raise ParseError("Invalid expression")\n'
    "sqlglot.errors.ParseError: Invalid expression"
)
SCRIPT_TRACEBACK = (
    'Traceback (most recent call last):\n  File "/testbed/test_headers.py", line 13, in <module>\n'
    "    cls_headers = HTTPProvider.get_request_headers()\n"
    "TypeError: get_request_headers() missing 1 required positional argument"
)
ASSERTION = (
    'Traceback (most recent call last):\n  File "/testbed/test_fix.py", line 9, in <module>\n'
    "    assert out == 3\nAssertionError"
)
SNIPPET_TRACEBACK = (
    'Traceback (most recent call last):\n  File "<string>", line 1, in <module>\n'
    "AttributeError: no attribute 'from_list'"
)
SCRIPT_SYNTAX = (
    '  File "/testbed/tests/test_errors.py", line 40\n'
    "    with pytest.raises(ValueError):\n    ^\nSyntaxError: invalid syntax"
)


def turn(*calls, tool="terminal"):
    """Logged-JSON form of one turn: (command, output, exit_code[, timeout]) per terminal call."""
    actions, observations = [], []
    for index, (command, output, exit_code, *timeout) in enumerate(calls):
        actions.append({"tool_name": tool, "tool_call_id": f"c{index}", "action": {"command": command}})
        observation = {"content": [{"type": "text", "text": output}], "exit_code": exit_code,
                       "timeout": bool(timeout and timeout[0])}
        observations.append({"tool_call_id": f"c{index}", "observation": observation})
    return actions, observations


def verdict(command, output, exit_code=0, timeout=False):
    (run,) = detect_test_runs(*turn((command, output, exit_code, timeout)))
    return run.kind, run.meaningful, run.reason


@pytest.mark.parametrize("command, output, exit_code, expected", [
    ("cd /testbed && python -m pytest tests/test_x.py -v", PYTEST_PASS, 0, (True, "tests executed")),
    ("pytest tests/test_x.py", PYTEST_FAIL, 1, (True, "tests executed")),
    ("cd /testbed && pytest -q tests/test_x.py", PYTEST_QUIET, 1, (True, "tests executed")),
    ("python -m pytest tests/test_x.py", COLLECTION, 2, (False, "collection failure")),
    ("python3 -m pytest tests/test_missing.py -v", NOT_FOUND, 4, (False, "collection failure")),
    ("python -m unittest tests.test_x -v", UNITTEST_OK, 0, (True, "tests executed")),
    ("python -m unittest unittests/test_y.py -v", UNITTEST_NONE, 0, (False, "no test summary")),
    ("cd /testbed && ./runtests.py queries", UNITTEST_OK, 0, (True, "tests executed")),
    ("python manage.py test app", UNITTEST_OK, 0, (True, "tests executed")),
])
def test_framework_runs_are_meaningful_only_when_tests_execute(command, output, exit_code, expected):
    assert verdict(command, output, exit_code) == ("framework", *expected)


@pytest.mark.parametrize("command, output, exit_code, expected", [
    ("python3 /testbed/reproduce_issue.py", REPO_TRACEBACK, 1, (True, "raised in repository code")),
    ("python3 /testbed/test_headers.py", SCRIPT_TRACEBACK, 1, (False, "raised outside repository code")),
    ("python3 test_fix.py", ASSERTION, 1, (True, "assertion failed")),
    ("python3 /testbed/tests/test_errors.py", SCRIPT_SYNTAX, 1, (False, "script does not compile")),
    ("python3 /testbed/test_hash_secrets.py", "All tests passed!", 0, (True, "clean run with output")),
    ("python3 /testbed/test_lrt.py", "error: No playable streams found\n=== Errors ===", 1,
     (True, "failed with output")),
    ("python3 /testbed/test_quiet.py", "", 0, (False, "no output")),
    ("python3 /testbed/missing.py", "python3: can't open file 'missing.py': [Errno 2] No such file or directory", 2,
     (False, "missing file or command")),
    ("python /testbed/test_unit.py", UNITTEST_OK, 0, (True, "tests executed")),  # a script that calls unittest.main()
])
def test_script_runs(command, output, exit_code, expected):
    assert verdict(command, output, exit_code) == ("script", *expected)


@pytest.mark.parametrize("command, output, expected", [
    ('python3 -c "import nipype; print(nipype.__file__)"', "/testbed/nipype/__init__.py",
     (False, "snippet without failure")),
    ('python3 -c "import tskit; tskit.TreeSequence.from_list([1])"', SNIPPET_TRACEBACK,
     (False, "raised outside repository code")),
    ('python3 -c "from sqlglot import parse; parse(\'x\')"',
     REPO_TRACEBACK.replace('/testbed/reproduce_issue.py', '<string>'), (True, "raised in repository code")),
    ('python3 -c "import json; print(json.dumps({}))"', "{}", (False, "snippet imports only the standard library")),
])
def test_snippets_count_only_when_they_raise_in_repository_code(command, output, expected):
    assert verdict(command, output, 1) == ("script", *expected)


def test_quoted_separators_do_not_split_snippets():
    # A semicolon or newline inside the quotes belongs to the snippet, not to the shell.
    assert classify_invocation('python -c "import sqlglot; print(1)"') == ("snippet", "import sqlglot; print(1)")
    assert classify_invocation("python -c 'import a\nimport b'") == ("snippet", "import a\nimport b")


@pytest.mark.parametrize("command", [
    "pip install -e .", "python setup.py develop", "pytest --help", "python -m pytest --collect-only -q",
    "python -m pip list", "ls tests", "grep -rn pytest setup.cfg", "python --version", "cat test_x.py",
])
def test_non_test_commands_are_ignored(command):
    assert classify_invocation(command) is None


@pytest.mark.parametrize("command", [
    "cd /testbed && timeout 60 python3 reproduce.py", "PYTHONPATH=/testbed python reproduce.py",
    "source /opt/conda/etc/profile.d/conda.sh; conda activate testbed; python reproduce.py",
])
def test_wrappers_and_chains_still_find_the_script(command):
    assert classify_invocation(command) == ("script", "reproduce.py")


def test_timeouts_and_other_tools_are_not_meaningful():
    assert verdict("pytest tests", "", -1, timeout=True) == ("framework", False, "timeout")
    assert detect_test_runs(*turn(("pytest tests", PYTEST_PASS, 0), tool="file_editor")) == []


def test_turn_is_meaningful_when_any_call_is():
    actions, observations = turn(("python3 broken.py", SCRIPT_SYNTAX, 1), ("pytest tests/test_x.py", PYTEST_PASS, 0))
    assert [run.meaningful for run in detect_test_runs(actions, observations)] == [False, True]
    assert turn_runs_meaningful_test(actions, observations)
    assert not turn_runs_meaningful_test(*turn(("python3 broken.py", SCRIPT_SYNTAX, 1)))


def test_sdk_objects_work_like_logged_json():
    action = SimpleNamespace(tool_name="terminal", tool_call_id="c0", action=SimpleNamespace(command="pytest tests"))
    text = SimpleNamespace(text=PYTEST_PASS)
    result = SimpleNamespace(content=[text], exit_code=0, timeout=False)
    observation = SimpleNamespace(tool_call_id="c0", observation=result)
    assert turn_runs_meaningful_test([action], [observation])
