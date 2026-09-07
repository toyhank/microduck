"""Visual approach followed by a bounded, calibrated blind strike advance.

The terminal advance is open loop: its duration must be calibrated for the
camera, walking policy and floor. Simulator ground truth is never read here.
"""
import numpy as np

from .visual_servo import VisualServoController


class SoccerState:
    SEARCH_BALL = "SEARCH_BALL"
    APPROACH_BALL = "APPROACH_BALL"
    TERMINAL_APPROACH = "TERMINAL_APPROACH"
    ALIGN_KICK = "ALIGN_KICK"
    KICK = "KICK"
    GOAL_CHECK = "GOAL_CHECK"
    CELEBRATE = "CELEBRATE"


class SoccerStateMachine:
    def __init__(self, mode="strict", kick_duration_sec=0.5,
                 terminal_duration_sec=1.52, settle_duration_sec=0.30,
                 image_height=240):
        if min(kick_duration_sec, terminal_duration_sec, settle_duration_sec) <= 0:
            raise ValueError("Motion durations must be positive")
        self.mode = mode
        self.state = SoccerState.SEARCH_BALL
        self.kick_duration_sec = kick_duration_sec
        self.terminal_duration_sec = terminal_duration_sec
        self.settle_duration_sec = settle_duration_sec
        self.image_height = image_height
        self.servo = VisualServoController(min_vx=0.30)
        self.last_ball_time = -float("inf")
        self.last_seen_cy = 0
        self.last_ball_bearing = 0.0
        self.last_goal_bearing = 0.0
        self.has_seen_goal = False
        self.terminal_timer = 0.0
        self.kick_timer = 0.0
        self.stabilize_timer = 0.0
        self.celebrate_timer = 0.0
        self.goal_scored = False

    def update(self, ball_det, goal_det, current_time, imu_yaw=0.0):
        """Return state, vx, vy, yaw rate, policy and a one-shot kick trigger.

        In simulation, current_time MUST be MuJoCo's d.time. Hardware callers
        use a monotonic clock. Goal bearing is telemetry; goal aiming remains
        a separate, unimplemented behavior.
        """
        trigger_kick = False
        if ball_det.visible:
            self.last_ball_time = current_time
            self.last_seen_cy = ball_det.cy
            self.last_ball_bearing = ball_det.bearing
        if goal_det.visible:
            self.last_goal_bearing = goal_det.bearing
            self.has_seen_goal = True

        # Resolve transitions before selecting the policy. In particular,
        # trigger_kick and the first kick action must occur in the same tick.
        if self.state == SoccerState.SEARCH_BALL:
            if ball_det.visible:
                self.state = SoccerState.APPROACH_BALL
        elif self.state == SoccerState.APPROACH_BALL:
            aligned = abs(self.last_ball_bearing - self.servo.target_bearing_offset) < 0.22
            at_bottom = ball_det.visible and ball_det.cy >= self.image_height * 230 / 240
            lost_below = (not ball_det.visible
                          and self.last_seen_cy >= self.image_height * 210 / 240
                          and current_time - self.last_ball_time < 0.8)
            if aligned and (at_bottom or lost_below):
                self.state = SoccerState.TERMINAL_APPROACH
                self.terminal_timer = current_time + self.terminal_duration_sec
            elif not ball_det.visible and current_time - self.last_ball_time >= 0.6:
                self.state = SoccerState.SEARCH_BALL
        elif self.state == SoccerState.TERMINAL_APPROACH:
            if current_time + 1e-9 >= self.terminal_timer:
                self.state = SoccerState.ALIGN_KICK
                self.stabilize_timer = current_time + self.settle_duration_sec
        elif self.state == SoccerState.ALIGN_KICK:
            if current_time + 1e-9 >= self.stabilize_timer:
                self.state = SoccerState.KICK
                self.kick_timer = current_time + self.kick_duration_sec
                trigger_kick = True
        elif self.state == SoccerState.KICK:
            if current_time + 1e-9 >= self.kick_timer:
                self.state = SoccerState.GOAL_CHECK
                self.stabilize_timer = current_time + 1.5
        elif self.state == SoccerState.GOAL_CHECK:
            if self.goal_scored:
                self.state = SoccerState.CELEBRATE
                self.celebrate_timer = current_time + 4.0
            elif current_time + 1e-9 >= self.stabilize_timer:
                self.state = SoccerState.SEARCH_BALL
        elif self.state == SoccerState.CELEBRATE:
            if current_time + 1e-9 >= self.celebrate_timer:
                self.state = SoccerState.SEARCH_BALL
                self.goal_scored = False

        vx = vy = vyaw = 0.0
        mode = "stand"
        if self.state == SoccerState.SEARCH_BALL:
            mode, vyaw = "walk", 0.40
        elif self.state == SoccerState.APPROACH_BALL:
            mode = "walk"
            if ball_det.visible:
                vx, vyaw = self.servo.compute_approach_velocity(ball_det.bearing, ball_det.distance)
                error = ball_det.bearing - self.servo.target_bearing_offset
                vy = float(np.clip(0.05 - 1.2 * error, -0.15, 0.15))
            # An unexplained occlusion is not permission to keep walking.
        elif self.state == SoccerState.TERMINAL_APPROACH:
            mode, vx, vy = "walk", 0.35, 0.06
        elif self.state == SoccerState.KICK:
            mode = "kick_right"
        return self.state, vx, vy, vyaw, mode, trigger_kick
