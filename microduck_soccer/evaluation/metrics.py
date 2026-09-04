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
    def __init__(self, goal_x=2.8, goal_y=0.0, goal_width=0.8, foot_geom_id=None, ball_geom_id=None, foot_site_id=None):
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.goal_width = goal_width
        self.foot_geom_id = foot_geom_id
        self.ball_geom_id = ball_geom_id
        self.foot_site_id = foot_site_id

    def evaluate_step(self, d, trunk_body_id, ball_body_id, right_foot_id, metrics: EpisodeMetrics, elapsed_time, ball_qvel_adr=None):
        """Monitor physical ground truth purely for logging and benchmark validation."""
        trunk_pos = d.xpos[trunk_body_id]
        ball_pos = d.xpos[ball_body_id]

        # Resolve accurate foot position (prefer site_xpos over body_xpos)
        if self.foot_site_id is not None and 0 <= self.foot_site_id < len(d.site_xpos):
            foot_pos = d.site_xpos[self.foot_site_id]
        elif 0 <= right_foot_id < len(d.site_xpos):
            foot_pos = d.site_xpos[right_foot_id]
        elif 0 <= right_foot_id < len(d.xpos):
            foot_pos = d.xpos[right_foot_id]
        else:
            foot_pos = trunk_pos + np.array([0.0, -0.042, -0.11])

        # 1. Check fall (trunk z < 0.06)
        if trunk_pos[2] < 0.06:
            metrics.fallen = True

        # 2. Check kick contact via multiple ground-truth physical channels:
        # Channel A: Geometric Euclidean foot-to-ball distance < 8 cm
        foot_dist = float(np.linalg.norm(foot_pos - ball_pos))
        if foot_dist < 0.08:
            metrics.kick_contact = True

        # Channel B: Physical MuJoCo contact pairs
        if self.foot_geom_id is not None and self.ball_geom_id is not None and hasattr(d, "ncon"):
            for c_idx in range(d.ncon):
                c = d.contact[c_idx]
                if (c.geom1 == self.foot_geom_id and c.geom2 == self.ball_geom_id) or \
                   (c.geom1 == self.ball_geom_id and c.geom2 == self.foot_geom_id):
                    metrics.kick_contact = True
                    break

        # Channel C: Momentum transfer / ball velocity spike (> 0.20 m/s)
        if ball_qvel_adr is not None:
            b_vel = float(np.linalg.norm(d.qvel[ball_qvel_adr:ball_qvel_adr + 3]))
            if b_vel > 0.20 and foot_dist < 0.18:
                metrics.kick_contact = True

        # 3. Check goal scored (ball crosses goal x >= 2.75 within post width)
        if ball_pos[0] >= 2.75 and abs(ball_pos[1] - self.goal_y) < (self.goal_width / 2.0):
            metrics.goal_scored = True

        metrics.total_time = elapsed_time
        metrics.final_ball_dist_to_goal = float(np.linalg.norm(ball_pos[0:2] - np.array([self.goal_x, self.goal_y])))
