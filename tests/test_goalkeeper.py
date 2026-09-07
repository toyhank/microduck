"""Unit tests for Goalkeeper Microduck robot and controller."""
import unittest
import numpy as np
import mujoco
from pathlib import Path

from microduck_soccer.assets import get_goalkeeper_scene_xml_path
from microduck_soccer.control.goalkeeper import GoalkeeperController, VisualGoalkeeperController, GoalkeeperState
from microduck_soccer.perception.ball_detector import BallDetection
from microduck_soccer.policy import DEFAULT_POSE


class GoalkeeperTests(unittest.TestCase):
    def setUp(self):
        self.gk = GoalkeeperController(goal_x=2.65, post_y=0.38)
        self.vgk = VisualGoalkeeperController(post_y=0.38)

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

    def test_visual_goalkeeper_guard_line(self):
        # Centered ball
        det_center = BallDetection(cx=160, cy=120, radius=5.0, distance=1.2, bearing=0.0, visible=True)
        mode, vx, vy, vyaw = self.vgk.update(det_center, 0.0)
        self.assertEqual(self.vgk.state, GoalkeeperState.GUARD_LINE)
        self.assertEqual(mode, "stand")

        # Ball on right side (bearing > 0 -> screen right / world +Y -> cmd_vy < 0)
        det_right = BallDetection(cx=200, cy=120, radius=5.0, distance=1.2, bearing=0.15, visible=True)
        mode, vx, vy, vyaw = self.vgk.update(det_right, 0.1)
        self.assertEqual(self.vgk.state, GoalkeeperState.GUARD_LINE)
        self.assertEqual(mode, "walk")
        self.assertLess(vy, 0.0)

        # Ball on left side (bearing < 0 -> screen left / world -Y -> cmd_vy > 0)
        det_left = BallDetection(cx=120, cy=120, radius=5.0, distance=1.2, bearing=-0.15, visible=True)
        mode, vx, vy, vyaw = self.vgk.update(det_left, 0.2)
        self.assertEqual(self.vgk.state, GoalkeeperState.GUARD_LINE)
        self.assertEqual(mode, "walk")
        self.assertGreater(vy, 0.0)

    def test_visual_goalkeeper_save_dive(self):
        # Frame 1: ball at 1.5m
        det1 = BallDetection(cx=165, cy=120, radius=4.0, distance=1.5, bearing=0.05, visible=True)
        self.vgk.update(det1, 1.0, new_frame=True)

        # Frame 2 (0.1s later): ball fast approaching at 1.0m
        det2 = BallDetection(cx=180, cy=120, radius=6.0, distance=1.0, bearing=0.12, visible=True)
        mode, vx, vy, vyaw = self.vgk.update(det2, 1.1, new_frame=True)
        self.assertEqual(self.vgk.state, GoalkeeperState.SAVE_DIVE)
        self.assertEqual(mode, "walk")
        self.assertLess(vy, 0.0)  # side dive towards incoming shot

    def test_visual_goalkeeper_clearance(self):
        # Ball very close at feet
        det_loose = BallDetection(cx=160, cy=180, radius=18.0, distance=0.20, bearing=0.02, visible=True)
        mode, vx, vy, vyaw = self.vgk.update(det_loose, 2.0, new_frame=True)
        self.assertEqual(self.vgk.state, GoalkeeperState.CLEAR_BALL)
        self.assertEqual(mode, "kick")

    def test_visual_goalkeeper_not_visible(self):
        # Lost ball: stand or hold center
        det_none = BallDetection(visible=False)
        mode, vx, vy, vyaw = self.vgk.update(det_none, 3.0, new_frame=True)
        self.assertEqual(self.vgk.state, GoalkeeperState.GUARD_LINE)
        self.assertEqual(mode, "stand")


if __name__ == "__main__":
    unittest.main()
