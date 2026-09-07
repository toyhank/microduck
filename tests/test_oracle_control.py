"""Oracle geometry and kick-gating checks independent of physics assets."""
import math
import unittest
import numpy as np
from microduck_soccer.control.oracle import OracleSoccerController


class OracleTests(unittest.TestCase):
    def call(self, c, now=1.1, rotation=0., robot=None, yaw=None,
             velocity=(0.,0.), ball_velocity=(0.,0.), scored=False):
        r=np.array([[math.cos(rotation),-math.sin(rotation)],
                    [math.sin(rotation),math.cos(rotation)]])
        ball=r@np.array([1.,0.]); goal=r@np.array([2.8,0.])
        # Isolated probe at x=.085, y=-.055 measured a -.321 rad shot bias.
        heading=rotation+.321
        body=np.array([[math.cos(heading),-math.sin(heading)],
                       [math.sin(heading),math.cos(heading)]])
        robot=ball-body@np.array([.085,-.055]) if robot is None else np.array(robot)
        return c.update(now,robot,heading if yaw is None else yaw,np.array(velocity),0.,
                        ball,np.array(ball_velocity),goal,scored)

    def test_compensated_stable_pose_triggers_real_policy_once(self):
        c=OracleSoccerController()
        self.assertEqual(self.call(c)[-2:],('kick_right',True))
        self.assertEqual(self.call(c,now=1.12)[-2:],('kick_right',False))
        self.assertEqual(self.call(c,now=1.6)[-2:],('stand',False))

    def test_rotating_entire_scene_preserves_kick_decision(self):
        self.assertEqual(self.call(OracleSoccerController(),rotation=math.pi/2)[-2:],
                         ('kick_right',True))

    def test_wrong_stance_or_moving_ball_does_not_trigger(self):
        for kwargs in ({'robot':(.8,0.)},{'yaw':1.2},{'ball_velocity':(.4,0.)}):
            with self.subTest(kwargs=kwargs):
                self.assertFalse(self.call(OracleSoccerController(),**kwargs)[-1])

    def test_goal_stops_controller(self):
        self.assertEqual(self.call(OracleSoccerController(),scored=True),
                         ('DONE',0.,0.,0.,'stand',False))


if __name__=='__main__':
    unittest.main()
