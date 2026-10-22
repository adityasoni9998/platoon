"""Regression checks for reconstructing the old ORM from turn rewards."""
from importlib.util import find_spec
import json
from pathlib import Path
import sys
import tempfile
import unittest

if find_spec('matplotlib') is None:
    raise unittest.SkipTest('Plotting tests require matplotlib')

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/reward_plots'))
from plot_orm_comparison import extract


def step(info, events=(), root='root', completion='completion'):
    return dict(type='trajectory_step_added', trajectory_id=root,
                step=dict(misc=dict(action_misc=dict(completion_id=completion), reward_misc=info),
                          action_events=list(events), observation_events=[]))


def state(**kwargs):
    return dict(reward_mode='turn', final_settlement=True, **kwargs)


class TurnRewardORMTest(unittest.TestCase):
    def extract(self, steps, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.jsonl'
            events = [dict(type='trajectory_created', trajectory=dict(id='root', parent_info=None)),
                      dict(type='trajectory_task_set', trajectory_id='root', task=dict(id='task')),
                      *steps, dict(type='trajectory_finished', trajectory_id='root', reward=-5)]
            path.write_text(''.join(json.dumps(e)+'\n' for e in events))
            return extract(path, **kwargs)

    def test_final_empty_patch_replaces_previous_localization_and_accumulates_formatting(self):
        editor = dict(id='edit-error', kind='ObservationEvent', tool_name='file_editor', observation={'is_error': True})
        events = [editor, dict(id='agent-error', kind='AgentErrorEvent'),
                  dict(id='edit-ok', kind='ObservationEvent', tool_name='file_editor', observation={'is_error': False})]
        previous = state(localization={'file_f1_reward': 1, 'module_f1_reward': 1, 'entity_f1_reward': 1})
        previous['final_settlement'] = False
        final = state(localization_reward=0, binary_reward=0, f2p_pass_fraction=0,
                      p2p_fail_fraction=0, composite_reward=0,
                      agent_error_event_reward=1, str_replace_reward=0)
        child = step(state(), [dict(id='child-error', kind='ObservationEvent', tool_name='file_editor', observation={'is_error': True})], root='child')
        result = self.extract([step(previous, events), child, step(final, [editor])], turn_rewards=True)
        v, audit = result[3], result[5]
        self.assertEqual(v['overall_localization'], 0)
        self.assertEqual(v['agent_error_reward'], 0)
        self.assertEqual(v['str_replace_reward'], .5)
        self.assertAlmostEqual(v['overall_reward'], .075)
        self.assertEqual(audit['trajectory_file_editor_calls'], 2)
        self.assertEqual(audit['duplicate_conversation_events_ignored'], 1)

    def test_final_response_formatting_matches_last_agent_message(self):
        bad = dict(id='bad', kind='MessageEvent', source='agent',
                   llm_message={'content': [{'type': 'text', 'text': '<tool_'}, {'type': 'text', 'text': 'call>oops'}]})
        r = self.extract([step(state(), [bad])], turn_rewards=True)
        self.assertEqual(r[3]['json_format_tool'], 0)
        good = dict(id='finish', kind='ActionEvent', source='agent', tool_name='finish', action={'message': 'done'})
        r = self.extract([step(state(), [bad]), step(state(), [good])], turn_rewards=True)
        self.assertEqual(r[3]['json_format_tool'], 1)

    def test_missing_components_keep_existing_defaults_and_weights(self):
        r = self.extract([step(state())], turn_rewards=True)
        self.assertEqual(r[3]['p2p_pass_rate'], 0)
        self.assertEqual(r[3]['str_replace_reward'], 0)
        self.assertAlmostEqual(r[3]['overall_reward'], .03)
        r = self.extract([step(state(binary_reward=1, f2p_pass_fraction=1, p2p_fail_fraction=0,
                                     localization={'file_f1_reward': 1, 'module_f1_reward': 1, 'entity_f1_reward': 1}))], turn_rewards=True)
        self.assertAlmostEqual(r[3]['overall_reward'], .95)

    def test_incomplete_final_state_is_not_replaced_by_an_earlier_score(self):
        info = state(binary_reward=1)
        info['final_settlement'] = False
        r = self.extract([step(info)], turn_rewards=True, skip_missing_rewards=True)
        self.assertIsNone(r[3])
        self.assertEqual(r[5]['missing_final_reward_state_rollouts'], 1)

    def test_ordinary_orm_extraction_keeps_existing_behavior(self):
        r = self.extract([step({'binary_reward': 1, 'f2p_pass_fraction': 1, 'p2p_fail_fraction': 0,
                               'localization': {'file_f1_reward': 1, 'module_f1_reward': 1, 'entity_f1_reward': 1},
                               'tool_json_error_reward': 1, 'agent_error_event_reward': 1, 'str_replace_reward': 1})])
        self.assertAlmostEqual(r[3]['overall_reward'], 1)
        self.assertNotIn('turn_reward_final_states', r[5])


if __name__ == '__main__':
    unittest.main()
