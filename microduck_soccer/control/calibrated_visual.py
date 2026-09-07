"""Visual aiming with calibrated camera geometry and short encoder odometry."""
import numpy as np
from .pose_controller import PoseSoccerController
from .state_machine import SoccerState
from ..perception.visual_localization import VisualLocalization


class CalibratedVisualSoccerController:
    def __init__(self, model, look_before_kick=False):
        self.localizer = VisualLocalization(model)
        self.motion = PoseSoccerController()
        self.state = 'SETTLE'
        self.goal_scored = False
        self.last_kick_end = -float('inf')
        self.look_before_kick = look_before_kick
        self.head_command = np.zeros(2)
        self.look_phase = None
        self.look_started = 0.
        self.look_samples = []
        self.look_confirmed_until = -float('inf')
        self.look_count = 0
        self.look_confirmations = 0
        self.rest_head = None
        self.look_target = np.array([-.6,.9])

    def update(self, ball, goal, now, *, joint_positions, orientation,
               angular_velocity, new_frame):
        loc = self.localizer
        loc.update_sensors(now, joint_positions, orientation)
        if new_frame and (self.look_phase is None or (
                self.look_phase == 'RAISE_HEAD' and now-self.look_started > .8)):
            loc.observe(ball, goal, now)
        if self.look_phase is not None:
            return self._inspect_ball(ball, now, joint_positions, angular_velocity, new_frame)
        # Finish an issued kick even while the camera cannot see the ball.
        if self.motion.state == 'KICK' and now >= self.motion.deadline:
            loc.ball_xy = None
            loc.ball_seen = -float('inf')
            loc.drift_velocity[:] = 0.
            self.last_kick_end = now
            self.motion.state = 'WAIT_BALL'
            self.motion.deadline = now + 1.5
        if self.motion.state == 'WAIT_BALL' and now < self.motion.deadline:
            self.state = SoccerState.GOAL_CHECK
            return self.state, 0., 0., 0., 'stand', False

        # Goal celebration (confirmed by evaluator) or field boundary out-of-bounds checks
        if self.goal_scored:
            self.state = SoccerState.CELEBRATE
            return self.state, 0., 0., 0., 'stand', False

        if loc.ball_xy is not None:
            from ..field import check_ball_field_status, BallFieldStatus
            status = check_ball_field_status(loc.ball_xy)
            if status == BallFieldStatus.OUT_OF_BOUNDS:
                self.state = SoccerState.OUT_OF_BOUNDS
                return self.state, 0., 0., 0., 'stand', False

        usable_ball = loc.ball_xy is not None and now-loc.ball_seen < 7.
        has_goal = loc.goal_xy is not None
        is_aiming = (self.motion.state == 'SETTLE')
        usable_goal = has_goal and (now - loc.goal_seen < 1.5 if is_aiming else now - loc.goal_seen < 20.0)

        if self.motion.state != 'KICK' and not (usable_ball and usable_goal):
            self.state = SoccerState.SEARCH_BALL
            if now < 1.:
                return self.state, 0., 0., 0., 'stand', False
            # Back away to regain the near-field view; otherwise scan in place.
            near = loc.ball_xy is not None and np.linalg.norm(loc.ball_xy-loc.xy) < .4
            return self.state, -.4 if near else 0., 0., 0. if near else 1., 'walk', False
        result = self.motion.update(now, loc.xy, loc.yaw, loc.velocity,
                                    float(np.linalg.norm(angular_velocity)),
                                    loc.ball_xy, loc.ball_velocity, loc.goal_xy,
                                    scored=self.goal_scored,
                                    foot_clearance=loc.foot_clearance())
        if result[-1] and self.look_before_kick and now > self.look_confirmed_until:
            # The navigation controller requested a kick, but no kick action
            # has reached the policy yet. Inspect first, then re-evaluate aim.
            self.motion.state = 'SETTLE'
            self.motion.kicks -= 1
            self.look_phase, self.look_started = 'LOOK_DOWN', now
            self.look_samples = []
            self.look_target = np.array([-.6,.9])
            self.rest_head = np.asarray(joint_positions)[5:7].copy()
            self.look_count += 1
            self.state = 'LOOK_DOWN'
            return self.state, 0., 0., 0., 'stand', False
        names = {'APPROACH': SoccerState.APPROACH_BALL,
                 'SETTLE': SoccerState.ALIGN_KICK, 'KICK': SoccerState.KICK,
                 'WAIT_BALL': SoccerState.GOAL_CHECK, 'DONE': SoccerState.CELEBRATE}
        self.state = names.get(result[0], result[0])
        return (self.state, *result[1:])

    def _inspect_ball(self, ball, now, joint_positions, angular_velocity, new_frame):
        loc = self.localizer
        elapsed = now-self.look_started
        if self.look_phase == 'LOOK_DOWN':
            if elapsed > 1.6 and len(self.look_samples) < 3:
                blend = min(1.,(elapsed-1.6)/.6)
                self.look_target = np.array([-.6,.9]) + blend*np.array([-.5,.2])
            self.head_command[:] = self.look_target * min(1., elapsed/.6)
            if new_frame and elapsed >= .9 and np.linalg.norm(angular_velocity) < .4:
                estimate = loc.observe_near_ball(ball, now)
                if estimate is not None:
                    self.look_samples.append(estimate)
            if (elapsed >= 1.6 and len(self.look_samples) >= 3) or elapsed >= 3.2:
                if len(self.look_samples) >= 3:
                    samples = np.array(self.look_samples[-5:])
                    center = np.median(samples, axis=0)
                    if np.max(np.linalg.norm(samples-center, axis=1)) < .008:
                        loc.ball_xy = center
                        loc.ball_velocity[:] = 0.
                        loc.drift_velocity[:] = 0.
                        loc.ball_seen = now
                        self.look_confirmed_until = now + 3.
                        self.look_confirmations += 1
                self.look_phase, self.look_started = 'RAISE_HEAD', now
        elif self.look_phase == 'RAISE_HEAD':
            self.head_command[:] = self.look_target * max(0., 1.-elapsed/.6)
            head_ready = np.max(np.abs(np.asarray(joint_positions)[5:7]-self.rest_head)) < .08
            stable = np.linalg.norm(angular_velocity) < .3 and np.linalg.norm(loc.velocity) < .03
            if elapsed >= 1.4 and head_ready and stable:
                self.head_command[:] = 0.
                if now > self.look_confirmed_until:
                    self.look_phase = 'LOOK_NOT_CONFIRMED'
                else:
                    self.look_phase = None
                    self.motion.state, self.motion.deadline = 'SETTLE', now + .2
            elif elapsed > 3.:
                self.head_command[:] = 0.
                self.look_phase = 'LOOK_NOT_CONFIRMED'
        self.state = self.look_phase or SoccerState.ALIGN_KICK
        return self.state, 0., 0., 0., 'stand', False
