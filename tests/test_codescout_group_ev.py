import json
import sys
from statistics import pvariance

import pytest

from scripts.reward_plots.compare_critic_ev import group_history, group_variances, read_group_rollout
from scripts.reward_plots import compare_critic_ev


def test_group_baseline_is_rollout_mean_but_ev_is_turn_weighted():
    groups = [
        [{"reward": 0.0, "turns": 1}, {"reward": 2.0, "turns": 3}],
        [{"reward": 2.0, "turns": 2}, {"reward": 4.0, "turns": 1}],
    ]
    (a, r), audit = group_variances(groups, 2)
    rewards = [0, 2, 2, 2, 2, 2, 4]
    advantages = [-1, 1, 1, 1, -1, -1, 1]
    assert a**2 == pytest.approx(pvariance(advantages))
    assert r**2 == pytest.approx(pvariance(rewards))
    assert audit == {"groups": 2, "rollouts": 4, "turns": 7}


def test_incomplete_ambiguous_and_unfinished_groups_are_excluded():
    valid = {"reward": 1.0, "turns": 2}
    groups = [[valid], [valid] * 3, [valid, {"reward": None, "turns": 1}],
              [valid, {"reward": float("nan"), "turns": 1}], [valid, valid]]
    moments, audit = group_variances(groups, 2)
    assert moments == (0.0, 0.0)  # Equal-reward groups are retained, not silently dropped.
    assert audit["groups"] == 1
    assert audit["incomplete_or_ambiguous_groups_excluded"] == 2
    assert audit["unfinished_or_invalid_groups_excluded"] == 2


def write_rollout(path, tid, reward):
    events = [
        {"type": "trajectory_created", "trajectory": {"id": tid, "parent_info": None}},
        {"type": "trajectory_task_set", "trajectory_id": tid, "task": {"id": "task"}},
        {"type": "trajectory_step_added", "trajectory_id": tid, "step": {"misc": {}}},
    ]
    # Three tools and a repeated event still represent one completion, not four turns.
    for completion in ("first", "first", "second"):
        events.append({"type": "trajectory_step_added", "trajectory_id": tid,
                       "step": {"misc": {"action_misc": {"completion_id": completion}},
                                "action_events": [{}, {}, {}]}})
    events.append({"type": "trajectory_finished", "trajectory_id": tid, "reward": reward,
                   "error_message": "Exhausted budget"})
    path.write_text("\n".join(json.dumps(e) for e in events) + '\n{"type":')


def test_jsonl_counts_completions_and_preserves_policy_versions(tmp_path):
    folder = tmp_path / "rollouts/train/7/events"
    folder.mkdir(parents=True)
    write_rollout(folder / "a.jsonl", "a", 0)
    write_rollout(folder / "b.jsonl", "b", 2)
    assert read_group_rollout(folder / "a.jsonl") == [
        {"id": "a", "task": "task", "turns": 2, "reward": 0}]
    history, audit = group_history(tmp_path, 2)
    assert history[0]["train/step"] == 7
    assert audit["7"]["turns"] == 4
    # One group with a constant baseline cannot explain variance within that group.
    assert history[0]["train/critic/raw_advantages_std"] == history[0]["train/critic/returns_std"]


def test_zero_return_variance_is_undefined_not_perfect_ev(tmp_path):
    folder = tmp_path / "rollouts/train/0/events"
    folder.mkdir(parents=True)
    write_rollout(folder / "a.jsonl", "a", 1)
    write_rollout(folder / "b.jsonl", "b", 1)
    history, audit = group_history(tmp_path, 2)
    assert history == []
    assert audit["0"]["undefined_ev_zero_return_variance"]


def test_group_refresh_needs_no_wandb(tmp_path, monkeypatch):
    folder = tmp_path / "rollouts/train/0/events"
    folder.mkdir(parents=True)
    for i in range(8):
        write_rollout(folder / f"{i}.jsonl", str(i), i % 2)
    monkeypatch.setattr(compare_critic_ev, "GROUP_RUN", tmp_path)
    monkeypatch.setitem(sys.modules, "wandb", None)
    result = compare_critic_ev.fetch(
        ("CodeScout", "codescout-platoon-tinker", "a221kfp1"), tmp_path, refresh=True)
    assert result["audit"]["0"]["turns"] == 16
    assert len(result["history"]) == 1
    assert result["source"] == str(tmp_path)
