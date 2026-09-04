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
    ALIGN_KICK = "ALIGN_KICK"
    KICK = "KICK"
    GOAL_CHECK = "GOAL_CHECK"
    CELEBRATE = "CELEBRATE"

class SoccerStateMachine:
    def __init__(self, mode="strict", kick_duration_sec=0.5):
        self.mode = mode  # "strict" or "demo"
        self.state = SoccerState.SEARCH_BALL
        self.kick_duration_sec = kick_duration_sec

        self.last_ball_time = 0.0
        self.last_goal_bearing = 0.0
        self.has_seen_goal = False

        self.kick_timer = 0.0
        self.stabilize_timer = 0.0
        self.celebrate_timer = 0.0
        self.goal_scored = False

        # Thresholds
        self.kick_distance_threshold = 0.18  # meters (estimated from vision)
        self.alignment_bearing_tolerance = 0.15  # rad (~8.5 deg)

    def update(self, ball_det, goal_det, current_time, imu_yaw=0.0):
        """
        Update the state machine based purely on perception detections.
        Returns:
            state: current SoccerState
            cmd_vx: desired forward velocity
            cmd_vyaw: desired yaw rate
            active_policy_mode: "walk", "stand", "kick_right"
            trigger_kick_now: bool
        """
        cmd_vx = 0.0
        cmd_vyaw = 0.0
        active_policy_mode = "walk"
        trigger_kick_now = False

        # Update perception memory
        if ball_det.visible:
            self.last_ball_time = current_time

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
                cmd_vyaw = 0.40

        elif self.state == SoccerState.APPROACH_BALL:
            active_policy_mode = "walk"
            if ball_det.visible:
                # Check if arrived at kick preparation distance
                if ball_det.distance <= self.kick_distance_threshold:
                    self.state = SoccerState.ALIGN_KICK
                    self.stabilize_timer = current_time + 0.4  # Settle stance before align/kick
                else:
                    # Visual servoing
                    bearing_error = ball_det.bearing + 0.05  # Bias slightly right for right foot
                    cmd_vyaw = -np.clip(1.8 * bearing_error, -0.5, 0.5)

                    dist_err = ball_det.distance - 0.14
                    cmd_vx = float(np.clip(0.6 * dist_err + 0.10, 0.12, 0.35))
            else:
                # Ball temporarily occluded or out of view
                time_lost = current_time - self.last_ball_time
                if time_lost < 0.6:
                    # Continue forward slowly hoping to catch sight
                    cmd_vx = 0.12
                    cmd_vyaw = 0.0
                else:
                    self.state = SoccerState.SEARCH_BALL

        elif self.state == SoccerState.ALIGN_KICK:
            # Stand firmly and orient toward goal
            if current_time < self.stabilize_timer:
                active_policy_mode = "stand"
                cmd_vx = 0.0
                cmd_vyaw = 0.0
            else:
                # Check goal alignment
                target_bearing = goal_det.bearing if goal_det.visible else self.last_goal_bearing

                if abs(target_bearing) > self.alignment_bearing_tolerance and self.has_seen_goal:
                    active_policy_mode = "walk"
                    cmd_vx = 0.0
                    cmd_vyaw = -np.clip(1.5 * target_bearing, -0.35, 0.35)
                else:
                    # Aligned and ready to kick!
                    self.state = SoccerState.KICK
                    self.kick_timer = current_time + self.kick_duration_sec
                    trigger_kick_now = True

        elif self.state == SoccerState.KICK:
            active_policy_mode = "kick_right"
            cmd_vx = 0.0
            cmd_vyaw = 0.0

            if current_time >= self.kick_timer:
                self.state = SoccerState.GOAL_CHECK
                self.stabilize_timer = current_time + 1.5

        elif self.state == SoccerState.GOAL_CHECK:
            active_policy_mode = "stand"
            cmd_vx = 0.0
            cmd_vyaw = 0.0

            if self.goal_scored:
                self.state = SoccerState.CELEBRATE
                self.celebrate_timer = current_time + 4.0
            elif current_time >= self.stabilize_timer:
                # Ball didn't score, search again
                self.state = SoccerState.SEARCH_BALL

        elif self.state == SoccerState.CELEBRATE:
            active_policy_mode = "stand"
            cmd_vx = 0.0
            cmd_vyaw = 0.0

            if current_time >= self.celebrate_timer:
                self.state = SoccerState.SEARCH_BALL
                self.goal_scored = False

        return self.state, cmd_vx, cmd_vyaw, active_policy_mode, trigger_kick_now
