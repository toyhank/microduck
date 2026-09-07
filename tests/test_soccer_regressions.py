"""Controller timing and evaluator regressions; no display or robot required."""
import unittest
from types import SimpleNamespace
import numpy as np
from microduck_soccer.control import SoccerStateMachine, SoccerState
from microduck_soccer.evaluation import SoccerEvaluator, EpisodeMetrics


def ball(visible=True, cy=180, distance=1.0, bearing=0.08):
    return SimpleNamespace(visible=visible, cy=cy, distance=distance, bearing=bearing)


def goal():
    return SimpleNamespace(visible=False)


class ControllerTests(unittest.TestCase):
    def test_kick_trigger_and_policy_are_synchronous_for_25_ticks(self):
        fsm = SoccerStateMachine()
        fsm.state = SoccerState.ALIGN_KICK
        fsm.stabilize_timer = 1.0
        outputs = [fsm.update(ball(), goal(), 1 + i * .02) for i in range(26)]
        self.assertTrue(outputs[0][-1])
        self.assertEqual(sum(o[-1] for o in outputs), 1)
        self.assertEqual(sum(o[-2] == 'kick_right' for o in outputs), 25)
        self.assertEqual(outputs[-1][0], SoccerState.GOAL_CHECK)
        self.assertEqual(outputs[-1][-2], 'stand')

    def test_terminal_deadline_stops_walking_in_same_tick(self):
        fsm = SoccerStateMachine()
        fsm.state = SoccerState.TERMINAL_APPROACH
        fsm.terminal_timer = 2.0
        result = fsm.update(ball(False), goal(), 2.0)
        self.assertEqual(result[0], SoccerState.ALIGN_KICK)
        self.assertEqual(result[1:4], (0.0, 0.0, 0.0))
        self.assertEqual(result[-2], 'stand')

    def test_distance_changes_speed_and_large_error_stops_advance(self):
        fsm = SoccerStateMachine()
        far = fsm.update(ball(distance=1), goal(), 0)
        near = fsm.update(ball(distance=.18), goal(), .02)
        turned = fsm.update(ball(bearing=.8), goal(), .04)
        self.assertLess(near[1], far[1])
        self.assertEqual(turned[1], 0)

    def test_occlusion_above_bottom_does_not_start_blind_advance(self):
        fsm = SoccerStateMachine()
        fsm.update(ball(cy=170), goal(), 0)
        result = fsm.update(ball(False), goal(), .1)
        self.assertEqual(result[1], 0)
        self.assertNotEqual(result[0], SoccerState.TERMINAL_APPROACH)
        self.assertEqual(fsm.update(ball(False), goal(), 1)[0], SoccerState.SEARCH_BALL)

    def test_badly_aligned_ball_at_bottom_does_not_trigger_kick_sequence(self):
        fsm = SoccerStateMachine()
        fsm.update(ball(bearing=.8), goal(), 0)
        result = fsm.update(ball(cy=238, bearing=.8), goal(), .1)
        self.assertEqual(result[0], SoccerState.APPROACH_BALL)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.evaluator = SoccerEvaluator(foot_geom_id=0, ball_geom_id=1, foot_site_id=0)
        self.metrics = EpisodeMetrics()
        self.data = SimpleNamespace(
            xpos=np.array([[0., 0., .12], [.07, 0., .035]]),
            site_xpos=np.array([[0., 0., .035]]),
            qvel=np.zeros(6), ncon=0, contact=[])

    def evaluate(self, mode='kick_right'):
        self.evaluator.evaluate_step(self.data, 0, 1, 0, self.metrics, 0,
                                     ball_qvel_adr=0, active_mode=mode)

    def test_proximity_and_velocity_do_not_fabricate_contact(self):
        self.data.qvel[0] = 1.0
        self.evaluate()
        self.assertFalse(self.metrics.kick_contact)
        self.assertFalse(self.metrics.foot_contact)

    def test_walk_contact_is_not_a_kick_and_kick_contact_is_latched(self):
        self.data.ncon = 1
        self.data.contact = [SimpleNamespace(geom1=1, geom2=0, dist=-.001)]
        self.evaluate('walk')
        self.assertTrue(self.metrics.foot_contact)
        self.assertFalse(self.metrics.kick_contact)
        self.evaluate('kick_right')
        self.assertTrue(self.metrics.kick_contact)
        self.data.ncon = 0
        self.data.contact = []
        self.evaluate('stand')
        self.assertTrue(self.metrics.kick_contact)

    def test_no_goal_before_forward_plane_crossing(self):
        self.data.xpos[1] = [2.76, 0, .035]
        self.evaluate()
        self.assertFalse(self.metrics.goal_scored)
        self.data.xpos[1] = [2.81, 0, .035]
        self.evaluate()
        self.assertTrue(self.metrics.goal_scored)

    def test_crossing_above_crossbar_is_not_goal(self):
        self.data.xpos[1] = [2.76, 0, .5]
        self.evaluate()
        self.data.xpos[1] = [2.81, 0, .5]
        self.evaluate()
        self.assertFalse(self.metrics.goal_scored)


if __name__ == '__main__':
    unittest.main()
