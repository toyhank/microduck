"""Unit tests for Goalkeeper Microduck robot and controller."""
import unittest
import numpy as np
import mujoco
from pathlib import Path

from microduck_soccer.assets import get_goalkeeper_scene_xml_path
from microduck_soccer.control.goalkeeper import GoalkeeperController, GoalkeeperState
from microduck_soccer.policy import DEFAULT_POSE


class GoalkeeperTests(unittest.TestCase):
    def setUp(self):
        self.gk = GoalkeeperController(goal_x=2.65, post_y=0.38)

    def test_goalkeeper_scene_loading(self):
        xml_path = get_goalkeeper_scene_xml_path()
        m = mujoco.MjModel.from_xml_path(xml_path)
        d = mujoco.MjData(m)
        self.assertGreaterEqual(m.nbody, 35)
        self.assertEqual(m.nu, 30)
        mujoco.mj_step(m, d)
        self.assertFalse(np.isnan(d.qpos).any())

    def test_guard_line_lateral_tracking(self):
        # Ball at center
        mode, vx, vy, vyaw = self.gk.update(
            np.array([1.5, 0.0, 0.035]), np.array([0.0, 0.0, 0.0]),
            np.array([2.65, 0.0, 0.116]), 0.0
        )
        self.assertEqual(self.gk.state, GoalkeeperState.GUARD_LINE)
        self.assertEqual(mode, "stand")

        # Ball shifted to +Y: Goalkeeper facing -X must move to world +Y, which is local -Y
        mode, vx, vy, vyaw = self.gk.update(
            np.array([1.5, 0.25, 0.035]), np.array([0.0, 0.0, 0.0]),
            np.array([2.65, 0.0, 0.116]), 0.1
        )
        self.assertEqual(self.gk.state, GoalkeeperState.GUARD_LINE)
        self.assertEqual(mode, "walk")
        self.assertLess(vy, 0.0)  # local -Y corresponds to world +Y

    def test_save_dive_shot_intercept(self):
        # Incoming fast shot towards right post (y = 0.20)
        ball_pos = np.array([1.8, 0.05, 0.035])
        ball_vel = np.array([1.5, 0.25, 0.0])  # vx = 1.5 towards goal
        mode, vx, vy, vyaw = self.gk.update(
            ball_pos, ball_vel, np.array([2.65, 0.0, 0.116]), 0.5
        )
        self.assertEqual(self.gk.state, GoalkeeperState.SAVE_DIVE)
        self.assertEqual(mode, "walk")
        # Goalkeeper should quickly shift laterally to intercept
        self.assertLess(vy, 0.0)

    def test_clear_ball_triggers_kick(self):
        # Ball stopped right in front of goalkeeper
        ball_pos = np.array([2.55, 0.02, 0.035])
        ball_vel = np.array([0.01, 0.0, 0.0])
        mode, vx, vy, vyaw = self.gk.update(
            ball_pos, ball_vel, np.array([2.65, 0.0, 0.116]), 1.0
        )
        self.assertEqual(self.gk.state, GoalkeeperState.CLEAR_BALL)
        self.assertEqual(mode, "kick")


if __name__ == "__main__":
    unittest.main()
