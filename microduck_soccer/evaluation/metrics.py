"""
Evaluation and Benchmark Metrics Module.
Strictly isolated from the robot controller to track ground truth metrics.
"""

import numpy as np

class EpisodeMetrics:
    def __init__(self, episode_id=0):
        self.episode_id = episode_id
        self.ball_detected = False
        self.approach_success = False
        self.kick_contact = False
        self.goal_scored = False
        self.fallen = False
        self.time_to_kick = 0.0
        self.total_time = 0.0
        self.initial_ball_dist = 0.0
        self.final_ball_dist_to_goal = 0.0

class SoccerEvaluator:
    def __init__(self, goal_x=2.8, goal_y=0.0, goal_width=0.8):
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.goal_width = goal_width

    def evaluate_step(self, d, trunk_body_id, ball_body_id, right_foot_id, metrics: EpisodeMetrics, elapsed_time):
        """Monitor physical ground truth purely for logging."""
        trunk_pos = d.xpos[trunk_body_id]
        ball_pos = d.xpos[ball_body_id]
        foot_pos = d.xpos[right_foot_id]

        # Check fall (trunk z < 0.06 or orientation upside down)
        if trunk_pos[2] < 0.06:
            metrics.fallen = True

        # Check kick contact (ball speed spike or foot distance < 0.08)
        foot_dist = np.linalg.norm(foot_pos - ball_pos)
        if foot_dist < 0.08:
            metrics.kick_contact = True

        # Check goal scored (ball crosses goal x >= 2.75 within post width)
        if ball_pos[0] >= 2.75 and abs(ball_pos[1] - self.goal_y) < (self.goal_width / 2.0):
            metrics.goal_scored = True

        metrics.total_time = elapsed_time
        metrics.final_ball_dist_to_goal = np.linalg.norm(ball_pos[0:2] - np.array([self.goal_x, self.goal_y]))
