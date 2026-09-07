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
        self.foot_contact = False
        self.min_foot_ball_distance = float("inf")
        self.goal_scored = False
        self.gk_contact = False
        self.shot_saved = False
        self.fallen = False
        self.time_to_kick = 0.0
        self.total_time = 0.0
        self.initial_ball_dist = 0.0
        self.final_ball_dist_to_goal = 0.0

class SoccerEvaluator:
    def __init__(self, goal_x=2.8, goal_y=0.0, goal_width=0.8, foot_geom_id=None, ball_geom_id=None, foot_site_id=None, gk_geom_ids=None):
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.goal_width = goal_width
        self.foot_geom_id = foot_geom_id
        self.ball_geom_id = ball_geom_id
        self.foot_site_id = foot_site_id
        self.gk_geom_ids = set(gk_geom_ids) if gk_geom_ids else set()
        self.previous_ball_pos = None

    def evaluate_step(self, d, trunk_body_id, ball_body_id, right_foot_id, metrics: EpisodeMetrics, elapsed_time, ball_qvel_adr=None, active_mode=None):
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

        # Proximity is diagnostic only. A kick requires an actual foot/ball
        # contact during the kick policy; walking into the ball is separate.
        foot_dist = float(np.linalg.norm(foot_pos - ball_pos))
        metrics.min_foot_ball_distance = min(metrics.min_foot_ball_distance, foot_dist)
        if (self.foot_geom_id is not None and self.foot_geom_id >= 0
                and self.ball_geom_id is not None and self.ball_geom_id >= 0):
            for c in d.contact[:d.ncon]:
                if {int(c.geom1), int(c.geom2)} == {self.foot_geom_id, self.ball_geom_id} and c.dist <= 0:
                    metrics.foot_contact = True
                    if active_mode in ("kick_right", "kick_left"):
                        metrics.kick_contact = True

        # Check goalkeeper contacts
        if self.gk_geom_ids and self.ball_geom_id is not None and self.ball_geom_id >= 0:
            for c in d.contact[:d.ncon]:
                if (int(c.geom1) == self.ball_geom_id and int(c.geom2) in self.gk_geom_ids) or \
                   (int(c.geom2) == self.ball_geom_id and int(c.geom1) in self.gk_geom_ids):
                    if c.dist <= 0:
                        metrics.gk_contact = True
                        if metrics.kick_contact and not metrics.goal_scored:
                            metrics.shot_saved = True

        # Require a forward crossing of the goal plane, within the opening.
        # Interpolate the crossing so substep speed cannot skip a post check.
        previous = self.previous_ball_pos
        if previous is not None and previous[0] < self.goal_x <= ball_pos[0]:
            fraction = (self.goal_x - previous[0]) / (ball_pos[0] - previous[0])
            crossing = previous + fraction * (ball_pos - previous)
            if abs(crossing[1] - self.goal_y) < self.goal_width / 2 and 0 <= crossing[2] < 0.35:
                metrics.goal_scored = True
        self.previous_ball_pos = ball_pos.copy()

        metrics.total_time = elapsed_time
        metrics.final_ball_dist_to_goal = float(np.linalg.norm(ball_pos[0:2] - np.array([self.goal_x, self.goal_y])))
