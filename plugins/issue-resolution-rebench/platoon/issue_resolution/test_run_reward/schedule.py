"""Source-code hashes of turn snapshots, and the interval schedule for the test-run reward.

A *source hash* identifies a snapshot's source-code edits. It is built from the snapshot's file
dictionaries rather than its patch text, and ignores files that are not source code: test files
(new or existing), new reproduction/scratch scripts, new non-Python files and caches. Writing or
editing tests therefore never counts as a source edit.

The schedule rewards testing around source edits. Rollouts without any source edit get nothing.
Otherwise every turn that changes the source hash starts a new interval, and the rollout's end
closes the last one. Each interval's last turn that ran a meaningful test earns the reward: the
reproduction before the first edit, the verification of each edit before the next one, and the
final verification after the last edit. An edit's turn starts the next interval, so a test run in
the same turn counts as verifying that edit.
"""

import hashlib
import json
import re
from collections import defaultdict

from platoon.issue_resolution.localization_reward.localization_reward import is_test_path
from platoon.issue_resolution.test_run_reward.detect import turn_runs_meaningful_test

SCRATCH_NAME = re.compile(
    r"^(reproduc|repro|debug|verify|check|demo|example|scratch|tmp|temp)", re.I
)
CACHE_PATH = re.compile(
    r"(^|/)(__pycache__|\.pytest_cache|\.mypy_cache|\.tox)/|\.pyc$|(^|/)\.coverage"
)
EMPTY_SOURCE_HASH = hashlib.sha256(b"[]").hexdigest()


def _paths(record) -> set[str]:
    paths = set(record.get("before_sources") or {}) | set(record.get("after_sources") or {})
    # Unreadable sources (binary files, or the missing side of an added/deleted file) appear only
    # as "<ref>:<path>" errors.
    paths.update(ref.split(":", 1)[1] for ref in record.get("source_errors") or {})
    return paths


def is_source_edit(path: str, added: bool) -> bool:
    if CACHE_PATH.search(path) or is_test_path(path):
        return False
    if added:
        name = path.rsplit("/", 1)[-1]
        return path.endswith(".py") and not SCRATCH_NAME.match(name)
    return True


def source_hash(record) -> str:
    """Hash the snapshot's edited source files: (path, contents after the edit, None if deleted)."""
    if not record or not record.get("patch"):
        return EMPTY_SOURCE_HASH
    before, after = record.get("before_sources") or {}, record.get("after_sources") or {}
    # A file has no base version when the patch adds it (unreadable binaries also land here, and
    # are then ignored as non-Python additions).
    files = [(path, after.get(path)) for path in sorted(_paths(record))
             if is_source_edit(path, added=path not in before)]
    return hashlib.sha256(json.dumps(files).encode()).hexdigest()


def turns_by_completion(events, action_to_completion) -> list[tuple[str, list, list]]:
    """Group a conversation's actions and observations by completion, in completion order."""
    actions, observations, order = defaultdict(list), defaultdict(list), []
    for event in events:
        completion = action_to_completion.get(getattr(event, "id", None))
        if completion is not None:
            if completion not in actions:
                order.append(completion)
            actions[completion].append(event)
            continue
        completion = action_to_completion.get(getattr(event, "action_id", None))
        if completion is not None:
            observations[completion].append(event)
    return [(completion, actions[completion], observations[completion]) for completion in order]


def boundaries(turns) -> set[int]:
    """Indices of the turns that change the source hash. turns: (completion, hash, meaningful)."""
    previous, changes = EMPTY_SOURCE_HASH, set()
    for index, (_, turn_source_hash, _) in enumerate(turns):
        if turn_source_hash != previous:
            changes.add(index)
        previous = turn_source_hash
    return changes


def interval_rewards(turns, weight: float) -> dict[str, float]:
    """turns: (completion, source_hash, ran_meaningful_test) in order. Returns rewarded completions.

    Without a source edit there are no intervals to reward.
    """
    cuts = boundaries(turns)
    if not cuts:
        return {}
    rewards, last_test = {}, None
    for index, (completion, _, meaningful) in enumerate(turns):
        if index in cuts:  # close the open interval
            if last_test is not None:
                rewards[last_test] = weight
            last_test = None
        if meaningful:
            last_test = completion
    if last_test is not None:  # the rollout's end closes the last interval
        rewards[last_test] = weight
    return rewards


def schedule_test_rewards(events, action_to_completion, completion_snapshots, snapshot_records,
                          weight: float, repo_path: str = "/testbed"):
    """Per-completion test-run rewards, plus each completion's diagnostics."""
    turns, patch_hash = [], None
    for completion, actions, observations in turns_by_completion(events, action_to_completion):
        patch_hash = completion_snapshots.get(completion, patch_hash)
        record = snapshot_records.get(patch_hash) if patch_hash else None
        meaningful = turn_runs_meaningful_test(actions, observations, repo_path)
        turns.append((completion, source_hash(record), meaningful))
    rewards, cuts = interval_rewards(turns, weight), boundaries(turns)
    details = {
        completion: {"source_hash": hash_, "meaningful_test_run": meaningful,
                     "source_changed": index in cuts}
        for index, (completion, hash_, meaningful) in enumerate(turns)
    }
    return rewards, details
