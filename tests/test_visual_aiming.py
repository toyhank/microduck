"""Perception and sensor-only shooting regressions."""
from pathlib import Path
import unittest
import cv2
import mujoco
import numpy as np
from microduck_soccer.perception import BallDetection, GoalDetection, GoalDetector
from microduck_soccer.perception.visual_localization import VisualLocalization
from microduck_soccer.control.calibrated_visual import CalibratedVisualSoccerController
from microduck_soccer.policy import DEFAULT_POSE
from microduck_soccer.assets import get_scene_xml_path

ROOT = Path(__file__).resolve().parents[1]


class GoalVisionTests(unittest.TestCase):
    def test_blue_sky_is_not_a_centered_goal(self):
        hsv = np.full((240, 320, 3), [105, 146, 140], dtype=np.uint8)
        self.assertFalse(GoalDetector().detect(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)).visible)
        cv2.rectangle(hsv, (220, 100), (279, 139), (111, 229, 255), 3)
        goal = GoalDetector().detect(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
        self.assertTrue(goal.visible)
        self.assertGreater(goal.bearing, .4)
        self.assertLess(goal.width, 80)


class SensorVisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(get_scene_xml_path())

    def test_private_kinematics_does_not_mutate_live_physics(self):
        live = mujoco.MjData(self.model)
        before = live.qpos.copy()
        loc = VisualLocalization(self.model)
        loc.update_sensors(0., DEFAULT_POSE, np.array([1., 0., 0., 0.]))
        loc.ball_xy = np.array([.09, -.055])
        self.assertTrue(np.isfinite(loc.foot_clearance()))
        np.testing.assert_array_equal(live.qpos, before)
        self.assertIsNot(loc.kinematics, live)

    def test_clipped_ball_does_not_refresh_stale_position(self):
        loc = VisualLocalization(self.model)
        loc.update_sensors(0., DEFAULT_POSE, np.array([1., 0., 0., 0.]))
        loc.ball_xy = np.array([.3, .02])
        clipped = BallDetection(cx=160, cy=235, radius=20, visible=True)
        loc.observe(clipped, GoalDetection(), 1.)
        np.testing.assert_array_equal(loc.ball_xy, [.3, .02])
        self.assertEqual(loc.ball_seen, -float('inf'))

    def test_projected_sphere_recovers_ground_position(self):
        loc = VisualLocalization(self.model)
        loc.update_sensors(0., DEFAULT_POSE, np.array([1., 0., 0., 0.]))
        cam = loc.kinematics.cam_xpos[loc.camera].copy()
        cam[2] += loc.height_estimate
        rotation = loc.kinematics.cam_xmat[loc.camera].reshape(3, 3)
        point = np.array([.6, -.04, .035])
        center = rotation.T @ (point-cam)
        axis = center / np.linalg.norm(center)
        tangent = np.cross(axis, [0., 1., 0.])
        tangent /= np.linalg.norm(tangent)
        second = np.cross(axis, tangent)
        alpha = np.arcsin(.035/np.linalg.norm(center))
        angles = np.linspace(0, 2*np.pi, 100, endpoint=False)
        rays = np.cos(alpha)*axis + np.sin(alpha)*(np.cos(angles)[:,None]*tangent + np.sin(angles)[:,None]*second)
        pixels = np.column_stack((160+loc.focal*rays[:,0]/-rays[:,2], 120-loc.focal*rays[:,1]/-rays[:,2]))
        middle, radius = cv2.minEnclosingCircle(pixels.astype(np.float32))
        ball = BallDetection(cx=middle[0], cy=middle[1], radius=radius, visible=True, contour=pixels.reshape(-1,1,2))
        loc.observe(ball, GoalDetection(), 0.)
        np.testing.assert_allclose(loc.ball_xy, point[:2], atol=1e-5)

    def inputs(self, new_frame=False):
        return dict(joint_positions=DEFAULT_POSE, orientation=np.array([1.,0.,0.,0.]),
                    angular_velocity=np.zeros(3), new_frame=new_frame)

    def test_missing_or_stale_goal_cannot_trigger_a_kick(self):
        controller = CalibratedVisualSoccerController(self.model)
        loc = controller.localizer
        loc.ball_xy = np.array([.085, -.055])
        loc.ball_seen = 1.
        controller.motion.deadline = 1.
        result = controller.update(BallDetection(), GoalDetection(), 1., **self.inputs())
        self.assertFalse(result[-1])
        self.assertNotEqual(result[-2], 'kick_right')
        loc.goal_xy = np.array([2.8, 0.])
        loc.goal_seen = -1.
        result = controller.update(BallDetection(), GoalDetection(), 1.02, **self.inputs())
        self.assertFalse(result[-1])

    def test_issued_kick_finishes_then_invalidates_ball_memory(self):
        controller = CalibratedVisualSoccerController(self.model)
        controller.motion.state = 'KICK'
        controller.motion.deadline = 1.5
        controller.localizer.ball_xy = np.array([.085, -.055])
        controller.localizer.goal_xy = np.array([2.8, 0.])
        for i in range(25):
            result = controller.update(BallDetection(), GoalDetection(), 1.+i*.02, **self.inputs())
            self.assertEqual(result[-2], 'kick_right')
            self.assertFalse(result[-1])
        result = controller.update(BallDetection(), GoalDetection(), 1.5, **self.inputs())
        self.assertEqual(result[-2], 'stand')
        self.assertIsNone(controller.localizer.ball_xy)

    def test_uncertain_look_returns_head_and_never_kicks(self):
        controller = CalibratedVisualSoccerController(self.model, look_before_kick=True)
        controller.look_phase = 'LOOK_DOWN'
        controller.look_started = 0.
        controller.rest_head = DEFAULT_POSE[5:7].copy()
        for i in range(250):
            result = controller.update(BallDetection(), GoalDetection(), i*.02, **self.inputs(new_frame=i%5==0))
            self.assertFalse(result[-1])
            self.assertEqual(result[-2], 'stand')
        self.assertEqual(controller.state, 'LOOK_NOT_CONFIRMED')
        np.testing.assert_array_equal(controller.head_command, [0.,0.])

    def test_look_waits_for_head_restore_and_body_stability(self):
        controller = CalibratedVisualSoccerController(self.model, look_before_kick=True)
        controller.look_phase = 'RAISE_HEAD'
        controller.look_started = 0.
        controller.rest_head = DEFAULT_POSE[5:7].copy()
        controller.look_confirmed_until = 10.
        inputs = self.inputs()
        inputs['angular_velocity'] = np.array([1.,0.,0.])
        controller.update(BallDetection(), GoalDetection(), 1.5, **inputs)
        self.assertEqual(controller.look_phase, 'RAISE_HEAD')
        inputs['angular_velocity'][:] = 0.
        inputs['joint_positions'] = DEFAULT_POSE.copy()
        inputs['joint_positions'][6] += .3
        controller.update(BallDetection(), GoalDetection(), 1.6, **inputs)
        self.assertEqual(controller.look_phase, 'RAISE_HEAD')
        controller.localizer.velocity[:] = 0.
        inputs['joint_positions'] = DEFAULT_POSE
        for i in range(30):
            result = controller.update(BallDetection(), GoalDetection(), 1.7+i*.02, **inputs)
            self.assertFalse(result[-1])
        self.assertIsNone(controller.look_phase)
        np.testing.assert_array_equal(controller.head_command, [0.,0.])


if __name__ == '__main__':
    unittest.main()
