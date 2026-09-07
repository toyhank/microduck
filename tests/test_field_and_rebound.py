"""
Unit tests for field boundaries, boundary evaluation, and in-bounds rebound kicking.
"""
import unittest
import numpy as np
import mujoco

from microduck_soccer.field import (
    check_ball_field_status, is_in_bounds, is_goal, is_out_of_bounds,
    BallFieldStatus, FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX,
    GOAL_X, GOAL_Y, GOAL_WIDTH,
)
from microduck_soccer.evaluation import SoccerEvaluator, EpisodeMetrics
from microduck_soccer.control.calibrated_visual import CalibratedVisualSoccerController
from microduck_soccer.control.state_machine import SoccerState
from microduck_soccer.perception import BallDetection, GoalDetection
from microduck_soccer.policy import DEFAULT_POSE
from microduck_soccer.assets import get_scene_xml_path


class FieldBoundaryTests(unittest.TestCase):
    def test_ball_inside_pitch_is_in_bounds(self):
        # Center kickoff spot
        self.assertTrue(is_in_bounds([1.2, 0.0]))
        self.assertEqual(check_ball_field_status([1.2, 0.0]), BallFieldStatus.IN_BOUNDS)
        
        # Quarter corners inside field
        self.assertTrue(is_in_bounds([0.0, 0.5]))
        self.assertTrue(is_in_bounds([2.0, -0.8]))
        self.assertTrue(is_in_bounds([-0.3, 0.9]))
        self.assertFalse(is_out_of_bounds([1.2, 0.0]))
        self.assertFalse(is_goal([1.2, 0.0]))

    def test_ball_beyond_touchlines_is_out_of_bounds(self):
        # Left sideline is y = 1.0; 1.05 is out of bounds
        self.assertTrue(is_out_of_bounds([1.2, 1.05]))
        self.assertEqual(check_ball_field_status([1.2, 1.05]), BallFieldStatus.OUT_OF_BOUNDS)
        self.assertFalse(is_in_bounds([1.2, 1.05]))

        # Right sideline is y = -1.0; -1.1 is out of bounds
        self.assertTrue(is_out_of_bounds([1.2, -1.1]))
        self.assertEqual(check_ball_field_status([1.2, -1.1]), BallFieldStatus.OUT_OF_BOUNDS)

    def test_ball_behind_backline_is_out_of_bounds(self):
        # Backline is x = -0.4; -0.5 is out of bounds
        self.assertTrue(is_out_of_bounds([-0.5, 0.0]))
        self.assertEqual(check_ball_field_status([-0.5, 0.0]), BallFieldStatus.OUT_OF_BOUNDS)

    def test_ball_crossing_goal_line_inside_posts_is_goal(self):
        # Inside goal opening: x = 2.85, y = 0.0
        self.assertTrue(is_goal([2.85, 0.0, 0.1]))
        self.assertEqual(check_ball_field_status([2.85, 0.0, 0.1]), BallFieldStatus.GOAL)
        self.assertFalse(is_out_of_bounds([2.85, 0.0, 0.1]))

        # Goal corners inside width (width is 0.8, so y in [-0.4, 0.4])
        self.assertTrue(is_goal([2.95, 0.35, 0.15]))
        self.assertTrue(is_goal([2.95, -0.35, 0.15]))

    def test_ball_crossing_endline_outside_posts_is_out_of_bounds(self):
        # Outside post: x = 2.85, y = 0.55 (missed to the left)
        self.assertTrue(is_out_of_bounds([2.85, 0.55]))
        self.assertEqual(check_ball_field_status([2.85, 0.55]), BallFieldStatus.OUT_OF_BOUNDS)
        self.assertFalse(is_goal([2.85, 0.55]))


class ReboundControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(get_scene_xml_path())

    def inputs(self, new_frame=False):
        return dict(joint_positions=DEFAULT_POSE, orientation=np.array([1., 0., 0., 0.]),
                    angular_velocity=np.zeros(3), new_frame=new_frame)

    def test_out_of_bounds_ball_stops_controller(self):
        controller = CalibratedVisualSoccerController(self.model)
        # Position ball out of bounds
        controller.localizer.ball_xy = np.array([1.2, 1.25])
        controller.localizer.ball_seen = 1.0
        controller.localizer.goal_xy = np.array([2.8, 0.0])
        controller.localizer.goal_seen = 1.0

        state, vx, vy, vyaw, mode, trig = controller.update(
            BallDetection(), GoalDetection(), 1.0, **self.inputs()
        )
        self.assertEqual(state, SoccerState.OUT_OF_BOUNDS)
        self.assertEqual((vx, vy, vyaw), (0.0, 0.0, 0.0))
        self.assertEqual(mode, 'stand')
        self.assertFalse(trig)

    def test_in_bounds_ball_after_kick_enables_rebound_approach(self):
        controller = CalibratedVisualSoccerController(self.model)
        # Setup finished kick at t=1.0
        controller.motion.state = 'KICK'
        controller.motion.deadline = 1.0
        controller.motion.kicks = 1
        controller.localizer.goal_xy = np.array([2.8, 0.0])
        controller.localizer.goal_seen = 0.5

        # Finish kick
        controller.update(BallDetection(), GoalDetection(), 1.0, **self.inputs())
        self.assertEqual(controller.motion.state, 'WAIT_BALL')

        # During WAIT_BALL (e.g. at t=1.5), camera observes ball stopped at x=1.8 in-bounds
        controller.localizer.ball_xy = np.array([1.8, 0.05])
        controller.localizer.ball_seen = 1.5

        # Advance past WAIT_BALL deadline (deadline is 1.0 + 1.5 = 2.5)
        state, vx, vy, vyaw, mode, trig = controller.update(
            BallDetection(), GoalDetection(), 2.6, **self.inputs()
        )
        # Controller should transition into APPROACH_BALL to chase the rebound
        self.assertEqual(state, SoccerState.APPROACH_BALL)
        self.assertEqual(mode, 'walk')
        self.assertGreater(vx, 0.0)


class EvaluatorReboundMetricsTests(unittest.TestCase):
    def test_rebound_kicks_and_goals_are_tracked(self):
        evaluator = SoccerEvaluator(foot_geom_id=0, ball_geom_id=1, foot_site_id=0)
        metrics = EpisodeMetrics()

        class MockData:
            xpos = np.array([[0., 0., 0.12], [1.2, 0., 0.035]])
            site_xpos = np.array([[0., 0., 0.035]])
            qvel = np.zeros(6)
            ncon = 0
            contact = []

        d = MockData()

        # Step 1: Initial walk
        evaluator.evaluate_step(d, 0, 1, 0, metrics, 0.0, active_mode='walk')
        self.assertEqual(metrics.total_kicks, 0)
        self.assertEqual(metrics.rebound_shots, 0)
        self.assertTrue(metrics.in_bounds)
        self.assertFalse(metrics.out_of_bounds)

        # Step 2: First kick triggered
        evaluator.evaluate_step(d, 0, 1, 0, metrics, 1.0, active_mode='kick_right')
        self.assertEqual(metrics.total_kicks, 1)
        self.assertEqual(metrics.rebound_shots, 0)

        # Step 3: Ball rolls forward, still in bounds; robot stands
        d.xpos[1] = [1.8, 0.0, 0.035]
        evaluator.evaluate_step(d, 0, 1, 0, metrics, 2.0, active_mode='stand')
        self.assertEqual(metrics.total_kicks, 1)
        self.assertTrue(metrics.in_bounds)

        # Step 4: Robot approaches and fires Rebound Kick (Kick #2)
        evaluator.evaluate_step(d, 0, 1, 0, metrics, 3.0, active_mode='kick_right')
        self.assertEqual(metrics.total_kicks, 2)
        self.assertEqual(metrics.rebound_shots, 1)

        # Step 5: Ball enters goal on rebound kick
        d.xpos[1] = [2.85, 0.0, 0.035]
        evaluator.evaluate_step(d, 0, 1, 0, metrics, 4.0, active_mode='stand')
        self.assertTrue(metrics.goal_scored)
        self.assertTrue(metrics.rebound_goal)


if __name__ == '__main__':
    unittest.main()
