"""The rebench reward breakdown: the final patch's rewards, test fractions and F1 scores, unweighted tool
formatting (clean-turn fractions and a clean-rollout flag), and every key for every trajectory."""

import pytest
import tinker

from platoon.issue_resolution.reward_breakdown import reward_breakdown
from platoon.train.tinker.config_defs import WorkflowConfig
from platoon.train.tinker.workflows.base import record_stats
from platoon.train.tinker.workflows.group_rollout_workflow import GroupRolloutWorkflow
from platoon.utils.stats_tracker import StatsTracker
from platoon.utils.tinker_data_processing import TrajectoryCollectionResult, TrajectoryStats


def turn(**reward_misc):
    return {"misc": {"action_misc": {"completion_id": "c"}, "reward_misc": reward_misc}}


PENALTIES = dict(tool_json_penalty=0.0, agent_error_penalty=0.0, file_editor_penalty=0.0)
KEYS = {f"reward/final/{name}" for name in ("localization_reward", "test_reward", "binary_reward", "f2p_pass_rate",
                                            "p2p_pass_rate", "file_f1", "module_f1", "function_f1")} | {
    "reward/file_editor_clean_turn_frac", "reward/agent_error_clean_turn_frac", "reward/tool_json_clean"}
DEFAULTS = {key: 0.0 for key in KEYS} | {
    "reward/final/p2p_pass_rate": 1.0, "reward/file_editor_clean_turn_frac": 1.0,
    "reward/agent_error_clean_turn_frac": 1.0, "reward/tool_json_clean": 1.0}


def test_final_rewards_come_from_the_last_turn_and_tool_formatting_is_unweighted():
    trajectory = {
        "reward": 0.37,
        "steps": [
            turn(**{**PENALTIES, "tool_json_penalty": -0.05, "file_editor_penalty": -0.05},
                 localization_reward=0.2, composite_reward=0.3),
            turn(**{**PENALTIES, "agent_error_penalty": -0.05, "file_editor_penalty": -0.05},
                 localization_reward=0.3, composite_reward=0.5, binary_reward=0.0, f2p_pass_fraction=0.5,
                 p2p_fail_fraction=0.2,
                 localization={"file_f1_reward": 0.5, "module_f1_reward": 0.3, "entity_f1_reward": 0.1}),
            turn(**PENALTIES, final_settlement=True, localization_reward=0.6, composite_reward=0.9,
                 binary_reward=1.0, f2p_pass_fraction=0.8, p2p_fail_fraction=0.25,
                 localization={"file_f1_reward": 1.0, "module_f1_reward": 0.5, "entity_f1_reward": 0.3}),
        ],
    }
    reward, metrics = reward_breakdown(trajectory)
    assert reward == 0.37  # the trajectory's reward is unchanged
    assert metrics == pytest.approx({
        "reward/final/localization_reward": 0.6,
        "reward/final/test_reward": 0.9,
        "reward/final/binary_reward": 1.0,
        "reward/final/f2p_pass_rate": 0.8,
        "reward/final/p2p_pass_rate": 0.75,
        "reward/final/file_f1": 1.0,
        "reward/final/module_f1": 0.5,
        "reward/final/function_f1": 0.3,
        # 3 turns: 2 with a file-editor error, 1 with an agent error, 1 with a JSON error.
        "reward/file_editor_clean_turn_frac": 1 / 3,
        "reward/agent_error_clean_turn_frac": 2 / 3,
        "reward/tool_json_clean": 0.0,
    })


def test_tool_formatting_ignores_the_penalty_weight():
    # Any non-zero penalty counts the turn, whatever its size.
    steps = [turn(**{**PENALTIES, "file_editor_penalty": -0.2}), turn(**PENALTIES), turn(**PENALTIES),
             turn(**{**PENALTIES, "file_editor_penalty": -1.0})]
    _, metrics = reward_breakdown({"reward": 0.0, "steps": steps})
    assert metrics["reward/file_editor_clean_turn_frac"] == 0.5
    assert metrics["reward/agent_error_clean_turn_frac"] == 1.0 and metrics["reward/tool_json_clean"] == 1.0


def test_without_a_patch_change_the_last_turn_rewards_and_last_full_scores_apply():
    # The last turn left the patch unchanged: it reports the rewards but not the full scores.
    steps = [
        turn(**PENALTIES, localization_reward=0.5, composite_reward=0.9, binary_reward=0.0, f2p_pass_fraction=0.7,
             p2p_fail_fraction=0.0, localization={"file_f1_reward": 0.8}),
        turn(**PENALTIES, localization_reward=0.5, composite_reward=0.9),
    ]
    _, metrics = reward_breakdown({"reward": 0.5, "steps": steps})
    assert metrics["reward/final/test_reward"] == 0.9 and metrics["reward/final/f2p_pass_rate"] == 0.7
    assert metrics["reward/final/file_f1"] == 0.8 and metrics["reward/final/module_f1"] == 0.0


def test_terminal_reward_runs_use_the_final_whole_conversation_checks():
    # Terminal rewards: earlier turns have no reward info; the final info scores the whole conversation.
    final = turn(model_patch="diff", localization_reward=0.4, binary_reward=1.0, f2p_pass_fraction=1.0,
                 p2p_fail_fraction=0.0, composite_reward=1.0,
                 localization={"file_f1_reward": 0.6, "module_f1_reward": 0.4, "entity_f1_reward": 0.2},
                 tool_json_error_reward=0.0, agent_error_event_reward=1.0, str_replace_reward=0.0,
                 file_editor_fail_fraction=0.25, file_editor_calls=4, file_editor_errors=1)
    _, metrics = reward_breakdown({"reward": 1.0, "steps": [{"misc": {}}, final]})
    assert set(metrics) == KEYS
    assert metrics["reward/final/binary_reward"] == 1.0 and metrics["reward/final/file_f1"] == 0.6
    assert metrics["reward/tool_json_clean"] == 0.0
    assert metrics["reward/agent_error_clean_turn_frac"] == 1.0
    assert metrics["reward/file_editor_clean_turn_frac"] == 0.75


@pytest.mark.parametrize("steps", [[], [{"misc": {}}, {}], [turn(**PENALTIES)]])
def test_without_reward_info_every_key_reports_its_default(steps):
    _, metrics = reward_breakdown({"reward": 0.0, "steps": steps})
    assert metrics == DEFAULTS


async def _group(*trajectories):
    workflow = GroupRolloutWorkflow.__new__(GroupRolloutWorkflow)
    workflow.config = WorkflowConfig(algorithm="ppo", group_size=len(trajectories))
    workflow.tracker = StatsTracker("train")

    async def episode(data, index):
        reward, breakdown = reward_breakdown(trajectories[index])
        datum = tinker.Datum(model_input=tinker.ModelInput.from_ints([1, 2]), loss_fn_inputs={})
        return TrajectoryCollectionResult(
            datums=[datum], task_reward=reward, root_rewards_dict=breakdown,
            trajectory_stats=[TrajectoryStats(str(index), reward, 1, 2, 2, 1, is_root=True, rewards_dict=breakdown)],
        )

    workflow.arun_episode_single = episode
    return await workflow.arun_episode({"task_id": "task"})


@pytest.mark.asyncio
async def test_groups_of_different_sizes_and_scores_export_together():
    # A partial group with an unscored rollout, then a scored group of two: every key is in both.
    tracker = StatsTracker("train")
    record_stats(await _group({"reward": 0.0, "steps": [{"misc": {}}]}), tracker)
    scored = {"reward": 0.6, "steps": [turn(**PENALTIES, localization_reward=0.5, composite_reward=0.9)]}
    record_stats(await _group(scored, scored), tracker)
    stats = tracker.export()
    assert stats["train/reward/final/test_reward/avg"] == pytest.approx(0.6)
    assert stats["train/root_reward/final/test_reward/avg"] == pytest.approx(0.6)
