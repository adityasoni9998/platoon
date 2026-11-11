"""Detect terminal calls that run tests, and whether their output says something about the code.

A call runs a test when one of its shell commands invokes a test framework (pytest, unittest, nose,
tox, Django-style runners) or a Python script (``python file.py`` or ``python -c``). Its output is
*meaningful* when it reflects the repository's code, not a broken script, a missing file or a
timeout:

- framework runs: at least one test executed, with no collection errors;
- scripts: a traceback raised in the repository's own code, a failed assertion, or output from a
  script that compiles (whatever its exit code);
- ``python -c`` snippets: only a traceback raised in repository code or a failed assertion. A
  snippet that runs cleanly explores the code rather than testing it.

Works on OpenHands SDK events and on their logged JSON form alike.
"""

import re
import shlex
import sys
from dataclasses import dataclass

STDLIB = frozenset(sys.stdlib_module_names)
SEPARATORS = re.compile(r"&&|\|\||[;|\n]")
PYTHON = re.compile(r"^python[0-9.]*$")
FRAMEWORK_MODULES = {"pytest", "unittest", "nose", "nose2"}
FRAMEWORK_COMMANDS = {"pytest", "py.test", "tox", "nosetests", "nose2"}
RUNNER_SCRIPTS = {"runtests.py", "run_tests.py"}  # Django and similar repository test runners.
NON_RUN_FLAGS = {"-h", "--help", "--version", "--collect-only", "--co"}
WRAPPERS = {"timeout", "time", "env", "sudo", "nohup", "stdbuf"}
PYTHON_FLAGS = {"-u", "-B", "-O", "-W", "ignore"}

# Frameworks: a summary with at least one executed test.
PYTEST_SUMMARY = re.compile(r"\b(\d+) (?:passed|failed|errors?|xfailed|xpassed)\b.* in [\d.]+s")
UNITTEST_SUMMARY = re.compile(r"^Ran (\d+) tests? in ", re.M)
COLLECTION_FAILURE = re.compile(
    r"errors? during collection|ImportError while importing test module|no tests ran", re.I
)
# Scripts: failures of the script itself rather than of the code under test.
SCRIPT_BROKEN = re.compile(r"can't open file|No such file or directory|command not found")
NOT_COMPILED = re.compile(r"^(SyntaxError|IndentationError|TabError)\b", re.M)
FRAME = re.compile(r'File "([^"]+)", line \d+')


@dataclass(frozen=True)
class TestRun:
    kind: str  # "framework" or "script"
    command: str
    meaningful: bool
    reason: str


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _output(observation) -> str:
    content = _get(observation, "content")
    if isinstance(content, str):
        return content
    return "".join(str(_get(part, "text", "")) for part in content or [])


def _segments(command: str) -> list[list[str]]:
    """Split a shell command into simple commands at unquoted ;, &&, ||, | and newlines."""
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|\n")
    lexer.whitespace = " \t\r"
    lexer.commenters = ""
    segments, current = [], []
    try:
        for token in lexer:
            if token and set(token) <= set(";&|\n"):
                segments.append(current)
                current = []
            else:
                current.append(token)
    except ValueError:  # unbalanced quotes: fall back to a quote-unaware split
        return [segment.split() for segment in SEPARATORS.split(command)]
    return [*segments, current]


def _strip_wrappers(tokens: list[str]) -> list[str]:
    while tokens and ("=" in tokens[0] and not tokens[0].startswith("-") or tokens[0] in WRAPPERS):
        tokens = tokens[1:]
        if tokens and re.fullmatch(r"-?\d+[smh]?", tokens[0]):  # timeout 60 ...
            tokens = tokens[1:]
    return tokens


def classify_invocation(command: str):
    """Return ("framework", None), ("script", path), ("snippet", code) or None."""
    for segment in _segments(command):
        tokens = _strip_wrappers(segment)
        if not tokens or NON_RUN_FLAGS & set(tokens):
            continue
        program = tokens[0].rsplit("/", 1)[-1]
        if program in FRAMEWORK_COMMANDS or program in RUNNER_SCRIPTS:
            return "framework", None
        if not PYTHON.match(program):
            continue
        args = [token for token in tokens[1:] if token not in PYTHON_FLAGS]
        if len(args) >= 2 and args[0] == "-m":
            if args[1] in FRAMEWORK_MODULES:
                return "framework", None
            continue
        if len(args) >= 2 and args[0] == "-c":
            return "snippet", args[1]
        if args and args[0].endswith(".py"):
            name = args[0].rsplit("/", 1)[-1]
            if name in RUNNER_SCRIPTS or name == "manage.py" and "test" in args[1:2]:
                return "framework", None
            if name != "setup.py":
                return "script", args[0]
    return None


def _imports_non_stdlib(code: str) -> bool:
    modules = re.findall(r"(?:^|[;\s])(?:from|import)\s+([A-Za-z_]\w*)", code)
    return any(module not in STDLIB for module in modules)


def _framework_verdict(output: str) -> tuple[bool, str]:
    if COLLECTION_FAILURE.search(output):
        return False, "collection failure"
    counts = PYTEST_SUMMARY.findall(output) + UNITTEST_SUMMARY.findall(output)
    if any(int(count) for count in counts):
        return True, "tests executed"
    return False, "no test summary"


def _script_verdict(output: str, exit_code, repo_path: str, script) -> tuple[bool, str]:
    """``script`` is the script's path, or None for a ``python -c`` snippet."""
    if "Traceback (most recent call last)" in output:
        frames = FRAME.findall(output)
        last = frames[-1] if frames else ""
        own = last == "<string>" or bool(script) and (
            last == script or last.endswith("/" + script.lstrip("./"))
        )
        in_repo = last.startswith(repo_path.rstrip("/") + "/") and "/site-packages/" not in last
        if in_repo and not own:
            return True, "raised in repository code"
        if re.search(r"^AssertionError\b", output, re.M):
            return True, "assertion failed"
        return False, "raised outside repository code"
    if SCRIPT_BROKEN.search(output):
        return False, "missing file or command"
    if script is None:
        return False, "snippet without failure"
    if NOT_COMPILED.search(output):
        return False, "script does not compile"
    if not output.strip():
        return False, "no output"
    return True, "clean run with output" if exit_code == 0 else "failed with output"


def detect_test_runs(action_events, observation_events, repo_path="/testbed") -> list[TestRun]:
    """Classify every terminal call in one turn that runs a test or a script."""
    observations = {
        _get(event, "tool_call_id"): _get(event, "observation")
        for event in observation_events or []
    }
    runs = []
    for event in action_events or []:
        if _get(event, "tool_name") != "terminal":
            continue
        command = _get(_get(event, "action"), "command") or ""
        invocation = classify_invocation(command)
        if invocation is None:
            continue
        kind = "framework" if invocation[0] == "framework" else "script"
        observation = observations.get(_get(event, "tool_call_id"))
        if observation is None:
            runs.append(TestRun(kind, command, False, "no observation"))
            continue
        output, exit_code = _output(observation), _get(observation, "exit_code")
        summary = PYTEST_SUMMARY.search(output) or UNITTEST_SUMMARY.search(output)
        if _get(observation, "timeout"):
            meaningful, reason = False, "timeout"
        elif kind == "framework" or summary:
            meaningful, reason = _framework_verdict(output)
        elif invocation[0] == "snippet" and not _imports_non_stdlib(invocation[1]):
            meaningful, reason = False, "snippet imports only the standard library"
        else:
            script = invocation[1] if invocation[0] == "script" else None
            meaningful, reason = _script_verdict(output, exit_code, repo_path, script)
        runs.append(TestRun(kind, command, meaningful, reason))
    return runs


def turn_runs_meaningful_test(action_events, observation_events, repo_path="/testbed") -> bool:
    runs = detect_test_runs(action_events, observation_events, repo_path)
    return any(run.meaningful for run in runs)
