"""
Visual Servoing Controller.
Computes smooth linear and yaw velocity commands from perception observations.
"""

import math
import numpy as np

class VisualServoController:
    def __init__(self, kp_yaw=1.8, kp_dist=0.6, max_vx=0.35, min_vx=0.10, max_vyaw=0.55):
        self.kp_yaw = kp_yaw
        self.kp_dist = kp_dist
        self.max_vx = max_vx
        self.min_vx = min_vx
        self.max_vyaw = max_vyaw

        # Target kick offset: Right foot is at y = -0.042m
        # When approaching the ball for right-foot kick, ball should be slightly right of center
        self.target_bearing_offset = -0.06  # rad

    def compute_approach_velocity(self, ball_bearing, ball_distance):
        """
        Compute (vx, vyaw) to approach the ball using visual servoing.
        Smoothly decelerates as the robot approaches the ball.
        """
        # Steering error relative to target foot bearing
        bearing_error = ball_bearing - self.target_bearing_offset
        vyaw = -np.clip(self.kp_yaw * bearing_error, -self.max_vyaw, self.max_vyaw)

        # Forward velocity: decelerate as distance shrinks
        # Stopping target distance is ~0.16m
        dist_error = ball_distance - 0.15
        if dist_error > 0.4:
            vx = self.max_vx
        else:
            vx = np.clip(self.kp_dist * dist_error + self.min_vx, self.min_vx, self.max_vx)

        return float(vx), float(vyaw)

    def compute_alignment_velocity(self, goal_bearing):
        """
        Compute yaw rate to align the robot's heading with the observed goal.
        """
        vyaw = -np.clip(self.kp_yaw * goal_bearing, -0.4, 0.4)
        return 0.0, float(vyaw)
