"""Position-based shot placement shared by estimated and oracle navigation.

Callers own localization: this module only receives numeric estimates and
returns commands. It cannot read a simulator, camera, or robot state.
"""
import math
import numpy as np


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class PoseSoccerController:
    def __init__(self, kick_direction_bias=-0.31):
        self.state = 'SETTLE'
        self.deadline = 1.0
        self.kicks = 0
        self.filtered_velocity = np.zeros(2)
        self.target_offset = np.array([0.080, -0.055])
        self.stop_offset = np.zeros(2)
        self.yaw_stop_bias = 0.0
        self.positioning_settle = False
        self.last_clearance = float("inf")
        # Isolated physical probes: right-foot strikes leave about 0.31 rad
        # to the right of the trunk heading in the useful lateral contact band.
        self.kick_direction_bias = kick_direction_bias

    def update(self, now, robot_xy, yaw, robot_velocity, angular_speed,
               ball_xy, ball_velocity, goal_xy, scored=False, foot_clearance=float("inf")):
        robot_xy, ball_xy, goal_xy = map(np.asarray, (robot_xy, ball_xy, goal_xy))
        self.filtered_velocity = .85 * self.filtered_velocity + .15 * np.asarray(robot_velocity)
        heading = math.atan2(*(goal_xy - ball_xy)[::-1]) - self.kick_direction_bias
        shot_rotation = np.array([[math.cos(heading), -math.sin(heading)],
                                  [math.sin(heading), math.cos(heading)]])
        body_rotation = np.array([[math.cos(yaw), -math.sin(yaw)],
                                  [math.sin(yaw), math.cos(yaw)]])
        ball_local = body_rotation.T @ (ball_xy - robot_xy)
        yaw_error = wrap(heading - yaw)
        base_target = ball_xy - shot_rotation @ self.target_offset
        target = base_target + shot_rotation @ self.stop_offset
        nav_yaw_error = wrap(heading + self.yaw_stop_bias - yaw)
        position_error = body_rotation.T @ (target - robot_xy)
        closing = foot_clearance < self.last_clearance - .001
        self.last_clearance = foot_clearance
        trigger = False
        if scored:
            self.state = 'DONE'
        if self.state == 'DONE':
            return self.state, 0., 0., 0., 'stand', False
        if self.state == 'KICK':
            if now + 1e-9 < self.deadline:
                return self.state, 0., 0., 0., 'kick_right', False
            self.state, self.deadline = 'WAIT_BALL', now + 1.5
        if self.state == 'WAIT_BALL':
            if now < self.deadline or (now < self.deadline + 2.0 and np.linalg.norm(ball_velocity) > .06):
                return self.state, 0., 0., 0., 'stand', False
            self.state = 'APPROACH'
        if self.state == 'SETTLE':
            if now < self.deadline:
                return self.state, 0., 0., 0., 'stand', False
            in_zone = (.070 <= ball_local[0] <= .098 and -.067 <= ball_local[1] <= -.025)
            # Bilinear interpolation of isolated standing-kick measurements.
            # Evaluate the actual settled pose, not just the nominal body yaw.
            rows = np.array([[-.324, -.314, -.302, -.117, .124],
                             [-.334, -.321, -.301, -.093, .132],
                             [-.359, -.344, -.262, -.054, .178]])
            lateral = [-.065, -.055, -.045, -.035, -.025]
            biases = [np.interp(ball_local[1], lateral, row) for row in rows]
            actual_bias = float(np.interp(ball_local[0], [.075, .085, .095], biases))
            goal_bearing = math.atan2(*(goal_xy-ball_xy)[::-1])
            shot_error = wrap(yaw + actual_bias - goal_bearing)
            margin = math.atan2(.28, float(np.linalg.norm(goal_xy-ball_xy)))
            aimed = abs(shot_error) < margin
            stable = np.linalg.norm(self.filtered_velocity) < .055 and angular_speed < .6
            if in_zone and aimed and stable and np.linalg.norm(ball_velocity) < .05:
                self.state, self.deadline = 'KICK', now + .5
                self.kicks += 1
                return self.state, 0., 0., 0., 'kick_right', True
            # Learn the displacement/yaw remaining after the gait-to-stand
            # transition, then pre-compensate the next positioning attempt.
            if self.positioning_settle:
                residual = shot_rotation.T @ (base_target - robot_xy)
                self.stop_offset = np.clip(self.stop_offset + .6 * residual, -.06, .06)
                self.yaw_stop_bias = float(np.clip(self.yaw_stop_bias + .6 * yaw_error, -.20, .20))
            self.state = 'APPROACH'
        # First move behind the ball; only approach the final strike point
        # once the ball-to-goal line is reachable without crossing the ball.
        goal_frame = shot_rotation.T @ (robot_xy - ball_xy)
        staging = abs(goal_frame[1] - .055) > .12 or goal_frame[0] > -.05
        if staging:
            waypoint = ball_xy - shot_rotation @ np.array([.35, -.045])
            position_error = body_rotation.T @ (waypoint - robot_xy)
        # The bundled walking network has a substantial command dead zone.
        # A tiny proportional command does not cause tiny physical motion.
        # Use measured minimum active commands with a positional deadband.
        def active_command(error, deadband, minimum, maximum, gain):
            if abs(error) <= deadband:
                return 0.0
            return math.copysign(min(maximum, minimum + gain * abs(error)), error)
        vx = active_command(position_error[0], .006, .19, .35, 1.2)
        vy = active_command(position_error[1], .006, .23, .35, 1.2)
        vyaw = active_command(nav_yaw_error, .035, 1.0, 1.5, 1.2)
        if abs(nav_yaw_error) > .6:
            vx = vy = 0.
        at_target = np.linalg.norm(position_error) < .012 and abs(nav_yaw_error) < .06
        imminent_contact = closing and foot_clearance < .035 and np.linalg.norm(ball_xy-robot_xy) < .25
        if not staging and (at_target or imminent_contact):
            self.positioning_settle = True
            self.state, self.deadline = 'SETTLE', now + .6
            return self.state, 0., 0., 0., 'stand', False
        return self.state, vx, vy, vyaw, 'walk', trigger
