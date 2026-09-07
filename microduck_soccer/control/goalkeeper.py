"""
Microduck Goalkeeper Controller
Autonomous goalkeeping state machine and motion planner for the defending Microduck robot.
"""

from enum import Enum
import numpy as np


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

        # 1. Incoming Shot Detection -> SAVE_DIVE
        is_shot_incoming = b_vx > 0.25 and b_x > 1.2
        if is_shot_incoming:
            self.state = GoalkeeperState.SAVE_DIVE
            # Predict intercept y-coordinate at goal line
            t_reach = max(0.01, (self.goal_x - b_x) / b_vx)
            y_intercept = b_y + b_vy * t_reach
            y_target = float(np.clip(y_intercept, -self.post_y + 0.06, self.post_y - 0.06))

            err_y = y_target - gk_y
            # Goalkeeper faces -X, so world +Y is local -Y:
            # cmd_vy_local = -world_vy
            if abs(err_y) > 0.04:
                self.active_mode = "walk"
                world_vy = float(np.clip(err_y * 2.8, -0.35, 0.35))
                cmd_vy = -world_vy
                # Small forward lean / advance to close down angle
                cmd_vx = 0.08
            else:
                # Square up and brace for impact
                self.active_mode = "stand"
                cmd_vx = 0.0
                cmd_vy = 0.0

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
