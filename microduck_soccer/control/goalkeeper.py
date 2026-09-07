"""
Microduck Goalkeeper Controller
Autonomous goalkeeping state machine and motion planner for the defending Microduck robot.
"""

import math
from enum import Enum
import numpy as np
from ..perception.ball_detector import BallDetection


class GoalkeeperState(Enum):
    GUARD_LINE = "GUARD_LINE"
    SAVE_DIVE = "SAVE_DIVE"
    CLEAR_BALL = "CLEAR_BALL"


class GoalkeeperController:
    """
    Autonomous Goalkeeper for Microduck.
    Stationed at the goalmouth (x ~ 2.65m, y in [-0.35, 0.35]m), facing -X towards the pitch.
    """

    def __init__(self, goal_x: float = 2.65, post_y: float = 0.38):
        self.goal_x = goal_x
        self.post_y = post_y
        self.state = GoalkeeperState.GUARD_LINE
        self.active_mode = "stand"  # 'stand', 'walk', 'kick'
        self.kick_start_time = 0.0
        self.kick_duration = 0.5
        self.saves_count = 0
        self.blocks_count = 0
        self.has_touched_ball = False

    def update(
        self,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        gk_pos: np.ndarray,
        current_time: float,
    ):
        """
        Update goalkeeper decision based on ball and robot state.

        Args:
            ball_pos: [x, y, z] in world coordinates.
            ball_vel: [vx, vy, vz] in world coordinates.
            gk_pos: [x, y, z] goalkeeper trunk position.
            current_time: simulation time in seconds.

        Returns:
            active_mode: 'stand', 'walk', or 'kick'
            cmd_vx: local forward velocity (towards -X in world)
            cmd_vy: local lateral velocity (towards -Y in world)
            cmd_vyaw: local yaw rate
        """
        cmd_vx = 0.0
        cmd_vy = 0.0
        cmd_vyaw = 0.0

        b_x, b_y = float(ball_pos[0]), float(ball_pos[1])
        b_vx, b_vy = float(ball_vel[0]), float(ball_vel[1])
        gk_x, gk_y = float(gk_pos[0]), float(gk_pos[1])

        # If currently in kick motion (clearing ball), hold kick until window finishes
        if self.state == GoalkeeperState.CLEAR_BALL:
            if current_time - self.kick_start_time < self.kick_duration:
                return "kick", 0.0, 0.0, 0.0
            else:
                self.state = GoalkeeperState.GUARD_LINE

        # 1. Incoming Shot Detection -> SAVE_DIVE / KICK CLEAR
        is_shot_incoming = b_vx > 0.20 and b_x > 1.2
        if is_shot_incoming:
            self.state = GoalkeeperState.SAVE_DIVE
            # Predict intercept y-coordinate at goal line
            t_reach = max(0.01, (self.goal_x - b_x) / b_vx)
            y_intercept = b_y + b_vy * t_reach
            y_target = float(np.clip(y_intercept, -self.post_y + 0.06, self.post_y - 0.06))

            err_y = y_target - gk_y

            # If ball is within 0.55m of goalkeeper, execute clearance kick save!
            if (self.goal_x - b_x) < 0.55:
                self.state = GoalkeeperState.CLEAR_BALL
                self.active_mode = "kick_right" if err_y <= 0.03 else "kick_left"
                self.kick_start_time = current_time
                return self.active_mode, 0.0, 0.0, 0.0

            # Rapid lateral movement and forward step
            self.active_mode = "walk"
            world_vy = float(np.clip(err_y * 4.0, -0.40, 0.40))
            cmd_vy = -world_vy
            cmd_vx = 0.12

        # 2. Loose Ball in front of Goalkeeper -> CLEAR_BALL
        elif 2.40 < b_x < 2.65 and abs(b_y - gk_y) < 0.12 and np.linalg.norm(ball_vel[:2]) < 0.20:
            self.state = GoalkeeperState.CLEAR_BALL
            self.active_mode = "kick"
            self.kick_start_time = current_time

        # 3. Ball far away -> GUARD_LINE (Positioning)
        else:
            self.state = GoalkeeperState.GUARD_LINE
            # Angle-bisecting positioning along the goal line
            y_target = float(np.clip(b_y * 0.70, -self.post_y + 0.08, self.post_y - 0.08))
            err_y = y_target - gk_y

            if abs(err_y) > 0.04:
                self.active_mode = "walk"
                world_vy = float(np.clip(err_y * 1.5, -0.22, 0.22))
                cmd_vy = -world_vy
                cmd_vx = 0.0
            else:
                self.active_mode = "stand"
                cmd_vx = 0.0
                cmd_vy = 0.0

        return self.active_mode, cmd_vx, cmd_vy, cmd_vyaw


class VisualGoalkeeperController:
    """
    Autonomous Pure-Vision Goalkeeper for Microduck.
    Uses monocular camera observations (BallDetection from gk_egocentric) to:
    1. Estimate relative ball position (x_rel, y_rel) in robot frame.
    2. Filter ball velocities (vx_rel, vy_rel) using an exponential smoothing tracker.
    3. Predict ball arrival time and intercept lateral offset during incoming shots.
    4. Issue lateral dive/intercept, goal line guarding, or clearance kick.
    """

    def __init__(self, post_y: float = 0.38):
        self.post_y = post_y
        self.state = GoalkeeperState.GUARD_LINE
        self.active_mode = "stand"
        self.kick_start_time = 0.0
        self.kick_duration = 0.5

        # Tracking state
        self.last_update_time = None
        self.last_x_rel = None
        self.last_y_rel = None
        self.vx_rel = 0.0
        self.vy_rel = 0.0
        self.ball_seen_time = -float("inf")
        self.est_gk_y = 0.0
        self.last_cmd_vy = 0.0
        self.predicted_intercept_y = 0.0

        # Smoothing factor for 10Hz visual velocity estimation
        self.alpha = 0.60

    def update(self, ball_det: BallDetection, current_time: float, new_frame: bool = True):
        """
        Update goalkeeper decision purely based on monocular camera detection.

        Args:
            ball_det: BallDetection object from goalkeeper camera.
            current_time: current simulation or real clock time.
            new_frame: whether this update corresponds to a new camera frame.

        Returns:
            active_mode: 'stand', 'walk', or 'kick'
            cmd_vx: local forward velocity (towards -X in world)
            cmd_vy: local lateral velocity (towards -Y in world)
            cmd_vyaw: local yaw rate
        """
        # If currently in kick motion (clearing ball), hold kick until window finishes
        if self.state == GoalkeeperState.CLEAR_BALL:
            if current_time - self.kick_start_time < self.kick_duration:
                return "kick", 0.0, 0.0, 0.0
            else:
                self.state = GoalkeeperState.GUARD_LINE
                self.last_x_rel = None

        cmd_vx = 0.0
        cmd_vy = 0.0
        cmd_vyaw = 0.0

        # Update lateral odometry from last step's command
        if self.last_update_time is not None and current_time > self.last_update_time:
            dt = current_time - self.last_update_time
            if dt > 0.0005:
                # cmd_vy is local lateral velocity: cmd_vy < 0 moves right (world +Y)
                self.est_gk_y += (-self.last_cmd_vy) * dt
                self.est_gk_y = float(np.clip(self.est_gk_y, -self.post_y + 0.04, self.post_y - 0.04))

        if ball_det is not None and ball_det.visible:
            self.ball_seen_time = current_time
            # Camera frame to local goalkeeper frame:
            # optical axis (Z in camera) = forward distance in front of robot
            x_meas = float(ball_det.distance * math.cos(ball_det.bearing))
            # lateral offset: positive bearing is screen right (world -Y in goalkeeper orientation)
            y_meas = float(ball_det.distance * math.sin(ball_det.bearing))

            # Velocity estimation via filtered finite differences on new frames
            if new_frame:
                if self.last_update_time is not None and current_time > self.last_update_time:
                    dt = current_time - self.last_update_time
                    if dt > 0.001 and self.last_x_rel is not None:
                        raw_vx = (x_meas - self.last_x_rel) / dt
                        raw_vy = (y_meas - self.last_y_rel) / dt
                        self.vx_rel = self.alpha * self.vx_rel + (1.0 - self.alpha) * raw_vx
                        self.vy_rel = self.alpha * self.vy_rel + (1.0 - self.alpha) * raw_vy

                self.last_x_rel = x_meas
                self.last_y_rel = y_meas
                self.last_update_time = current_time

            # 1. Shot Incoming Detection -> SAVE_DIVE / KICK SAVE
            # Ball approaching towards goalkeeper (vx_rel < -0.15 m/s)
            is_shot_incoming = (self.vx_rel < -0.15) and (x_meas < 2.2)
            if is_shot_incoming:
                self.state = GoalkeeperState.SAVE_DIVE
                approach_speed = max(0.20, -self.vx_rel)
                t_reach = max(0.01, min(1.5, x_meas / approach_speed))
                y_intercept_rel = y_meas + self.vy_rel * t_reach
                self.predicted_intercept_y = y_intercept_rel

                # If ball is close (x < 0.55m), execute dynamic kick clearance save
                if x_meas < 0.55:
                    self.state = GoalkeeperState.CLEAR_BALL
                    self.active_mode = "kick_right" if y_intercept_rel <= 0.03 else "kick_left"
                    self.kick_start_time = current_time
                    self.last_cmd_vy = 0.0
                    self.last_update_time = current_time
                    return self.active_mode, 0.0, 0.0, 0.0

                # Otherwise rapidly step laterally to align with intercept and step forward to cut angle
                self.active_mode = "walk"
                world_vy = float(np.clip(y_intercept_rel * 4.0, -0.40, 0.40))
                cmd_vy = -world_vy
                cmd_vx = 0.12  # Step forward to close down angle!

            # 2. Loose ball right in front of goalkeeper -> CLEAR_BALL
            elif x_meas < 0.35 and abs(y_meas) < 0.20:
                self.state = GoalkeeperState.CLEAR_BALL
                self.active_mode = "kick_right" if y_meas <= 0.0 else "kick_left"
                self.kick_start_time = current_time
                self.last_cmd_vy = 0.0
                self.last_update_time = current_time
                return self.active_mode, 0.0, 0.0, 0.0

            # 3. Ball Far / Slow -> GUARD_LINE
            else:
                self.state = GoalkeeperState.GUARD_LINE
                # Visual angle bisecting: track bearing to center ball in field of view
                bearing_err = ball_det.bearing
                if abs(bearing_err) > 0.035:
                    self.active_mode = "walk"
                    world_vy = float(np.clip(bearing_err * 1.2, -0.25, 0.25))
                    cmd_vy = -world_vy
                    cmd_vx = 0.0
                else:
                    self.active_mode = "stand"
                    cmd_vx = 0.0
                    cmd_vy = 0.0

        else:
            # Ball in near-ground BLIND ZONE (< 0.4m) during shot arrival!
            # Extrapolate ballistic trajectory rather than freezing or standing still.
            dt_blind = current_time - self.ball_seen_time
            if dt_blind < 0.70 and self.vx_rel < -0.15 and self.last_x_rel is not None and self.last_x_rel < 1.0:
                x_extrap = self.last_x_rel + self.vx_rel * dt_blind
                y_extrap = self.last_y_rel + self.vy_rel * dt_blind
                if x_extrap < 0.50:
                    # Execute blind-spot kick save!
                    self.state = GoalkeeperState.CLEAR_BALL
                    self.active_mode = "kick_right" if y_extrap <= 0.03 else "kick_left"
                    self.kick_start_time = current_time
                    self.last_cmd_vy = 0.0
                    self.last_update_time = current_time
                    return self.active_mode, 0.0, 0.0, 0.0
                else:
                    self.state = GoalkeeperState.SAVE_DIVE
                    self.active_mode = "walk"
                    world_vy = float(np.clip(y_extrap * 4.0, -0.40, 0.40))
                    cmd_vy = -world_vy
                    cmd_vx = 0.12
            else:
                # Ball lost or far away: return towards center of goal line
                self.state = GoalkeeperState.GUARD_LINE
                if abs(self.est_gk_y) > 0.05:
                    self.active_mode = "walk"
                    world_vy = float(np.clip(-self.est_gk_y * 1.2, -0.20, 0.20))
                    cmd_vy = -world_vy
                else:
                    self.active_mode = "stand"
                    cmd_vx = 0.0
                    cmd_vy = 0.0

        # Respect goal post limits
        if self.est_gk_y >= self.post_y - 0.05 and cmd_vy < 0:
            cmd_vy = 0.0
        elif self.est_gk_y <= -self.post_y + 0.05 and cmd_vy > 0:
            cmd_vy = 0.0

        self.last_cmd_vy = cmd_vy
        self.last_update_time = current_time
        return self.active_mode, cmd_vx, cmd_vy, cmd_vyaw
