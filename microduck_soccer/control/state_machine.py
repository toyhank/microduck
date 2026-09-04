"""
Finite State Machine for Autonomous Soccer Play.
Supports both STRICT mode (100% pure vision, no ground truth) and DEMO mode.
"""

import time
import math
import numpy as np

class SoccerState:
    SEARCH_BALL = "SEARCH_BALL"
    APPROACH_BALL = "APPROACH_BALL"
    TERMINAL_APPROACH = "TERMINAL_APPROACH"
    ALIGN_KICK = "ALIGN_KICK"
    KICK = "KICK"
    GOAL_CHECK = "GOAL_CHECK"
    CELEBRATE = "CELEBRATE"

class SoccerStateMachine:
    def __init__(self, mode="strict", kick_duration_sec=0.5, terminal_duration_sec=1.52, settle_duration_sec=0.30):
        self.mode = mode  # "strict" or "demo"
        self.state = SoccerState.SEARCH_BALL
        self.kick_duration_sec = kick_duration_sec
        self.terminal_duration_sec = terminal_duration_sec  # ~76 steps at 50Hz for 0.265m advance
        self.settle_duration_sec = settle_duration_sec      # ~15 steps at 50Hz to damp momentum

        self.last_ball_time = 0.0
        self.last_seen_cy = 0
        self.last_goal_bearing = 0.0
        self.has_seen_goal = False

        self.terminal_timer = 0.0
        self.kick_timer = 0.0
        self.stabilize_timer = 0.0
        self.celebrate_timer = 0.0
        self.goal_scored = False

    def update(self, ball_det, goal_det, current_time, imu_yaw=0.0):
        """
        Update the state machine based purely on perception detections.
        Returns:
            state: current SoccerState
            cmd_vx: desired forward velocity
            cmd_vy: desired lateral strafing velocity
            cmd_vyaw: desired yaw rate
            active_policy_mode: "walk", "stand", "kick_right"
            trigger_kick_now: bool
        """
        cmd_vx = 0.0
        cmd_vy = 0.0
        cmd_vyaw = 0.0
        active_policy_mode = "walk"
        trigger_kick_now = False

        # Update perception memory
        if ball_det.visible:
            self.last_ball_time = current_time
            self.last_seen_cy = ball_det.cy

        if goal_det.visible:
            self.last_goal_bearing = goal_det.bearing
            self.has_seen_goal = True

        # State dispatch
        if self.state == SoccerState.SEARCH_BALL:
            active_policy_mode = "walk"
            if ball_det.visible:
                self.state = SoccerState.APPROACH_BALL
            else:
                # Rotate slowly in place to scan field
                cmd_vx = 0.0
                cmd_vy = 0.0
                cmd_vyaw = 0.40

        elif self.state == SoccerState.APPROACH_BALL:
            active_policy_mode = "walk"
            if ball_det.visible:
                # Target bearing: align ball with right foot line-of-sight (+0.08 rad to the right)
                target_bearing = 0.08
                err_bearing = ball_det.bearing - target_bearing
                cmd_vyaw = -float(np.clip(2.0 * err_bearing, -0.45, 0.45))
                cmd_vx = 0.35
                # Lateral drift compensation and fine alignment
                cmd_vy = float(np.clip(0.05 - 1.2 * err_bearing, -0.15, 0.15))

                # Terminal approach trigger: ball reaches the lower boundary of the camera frame
                if ball_det.cy >= 230:
                    self.state = SoccerState.TERMINAL_APPROACH
                    self.terminal_timer = current_time + self.terminal_duration_sec
            else:
                # Ball slipped into blind spot under beak while at bottom of frame
                if self.last_seen_cy >= 210 and (current_time - self.last_ball_time) < 0.8:
                    self.state = SoccerState.TERMINAL_APPROACH
                    self.terminal_timer = current_time + (self.terminal_duration_sec - 0.08)
                elif (current_time - self.last_ball_time) < 0.6:
                    # Brief occlusion: keep forward momentum
                    cmd_vx = 0.30
                    cmd_vy = 0.0
                    cmd_vyaw = 0.0
                else:
                    self.state = SoccerState.SEARCH_BALL

        elif self.state == SoccerState.TERMINAL_APPROACH:
            # Calibrated dead-reckoning advance directly into the 0.09m foot strike zone
            active_policy_mode = "walk"
            cmd_vx = 0.35
            cmd_vy = 0.03
            cmd_vyaw = 0.0

            if current_time >= self.terminal_timer:
                self.state = SoccerState.ALIGN_KICK
                self.stabilize_timer = current_time + self.settle_duration_sec

        elif self.state == SoccerState.ALIGN_KICK:
            # Settle stance into firm standing pose to eliminate momentum
            active_policy_mode = "stand"
            cmd_vx = 0.0
            cmd_vy = 0.0
            cmd_vyaw = 0.0

            if current_time >= self.stabilize_timer:
                self.state = SoccerState.KICK
                self.kick_timer = current_time + self.kick_duration_sec
                trigger_kick_now = True

        elif self.state == SoccerState.KICK:
            active_policy_mode = "kick_right"
            cmd_vx = 0.0
            cmd_vy = 0.0
            cmd_vyaw = 0.0

            if current_time >= self.kick_timer:
                self.state = SoccerState.GOAL_CHECK
                self.stabilize_timer = current_time + 1.5

        elif self.state == SoccerState.GOAL_CHECK:
            active_policy_mode = "stand"
            cmd_vx = 0.0
            cmd_vy = 0.0
            cmd_vyaw = 0.0

            if self.goal_scored:
                self.state = SoccerState.CELEBRATE
                self.celebrate_timer = current_time + 4.0
            elif current_time >= self.stabilize_timer:
                # Ball didn't score: re-engage to pursue ball for follow-up strike
                self.state = SoccerState.SEARCH_BALL

        elif self.state == SoccerState.CELEBRATE:
            active_policy_mode = "stand"
            cmd_vx = 0.0
            cmd_vy = 0.0
            cmd_vyaw = 0.0

            if current_time >= self.celebrate_timer:
                self.state = SoccerState.SEARCH_BALL
                self.goal_scored = False

        return self.state, cmd_vx, cmd_vy, cmd_vyaw, active_policy_mode, trigger_kick_now
