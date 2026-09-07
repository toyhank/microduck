"""
Evaluation and Benchmark Metrics Module.
Strictly isolated from the robot controller to track ground truth metrics.
"""

import numpy as np

from ..field import (
    check_ball_field_status, BallFieldStatus,
    FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX,
    GOAL_INNER_HALF_WIDTH, GOAL_CROSSBAR_UNDERSIDE,
)

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
        self.out_of_bounds = False
        self.in_bounds = True
        self.total_kicks = 0
        self.rebound_shots = 0
        self.rebound_goal = False
        self.time_to_kick = 0.0
        self.total_time = 0.0
        self.initial_ball_dist = 0.0
        self.final_ball_dist_to_goal = 0.0

class SoccerEvaluator:
    def __init__(self, goal_x=2.8, goal_y=0.0, goal_width=0.8, foot_geom_id=None, ball_geom_id=None, foot_site_id=None, gk_geom_ids=None):
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.goal_width = goal_width
        if goal_width != 0.8:
            self.inner_half_width = max(0.1, (goal_width - 0.04) / 2.0)
        else:
            self.inner_half_width = GOAL_INNER_HALF_WIDTH
        self.crossbar_underside = GOAL_CROSSBAR_UNDERSIDE
        self.foot_geom_id = foot_geom_id
        self.ball_geom_id = ball_geom_id
        self.foot_site_id = foot_site_id
        self.gk_geom_ids = set(gk_geom_ids) if gk_geom_ids else set()
        self.previous_ball_pos = None
        self.previous_active_mode = None

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

        # Track kicks and rebounds
        is_kick_mode = active_mode in ("kick_right", "kick_left")
        was_kick_mode = self.previous_active_mode in ("kick_right", "kick_left")
        if is_kick_mode and not was_kick_mode:
            metrics.total_kicks += 1
            if metrics.total_kicks > 1:
                metrics.rebound_shots += 1
        self.previous_active_mode = active_mode

        # Check field boundary status (In-Bounds vs Out-of-Bounds vs Goal)
        status = check_ball_field_status(ball_pos)
        if metrics.out_of_bounds:
            # Once out-of-bounds, the ball is dead; cannot revert to in_bounds
            metrics.in_bounds = False
        elif status == BallFieldStatus.OUT_OF_BOUNDS:
            metrics.out_of_bounds = True
            metrics.in_bounds = False
        elif status == BallFieldStatus.IN_BOUNDS:
            metrics.in_bounds = True

        # Require a forward crossing of the goal plane, strictly within the inner opening.
        # A ball that was already out of bounds, or that crosses outside the opening / over the bar,
        # CAN NEVER score a goal and is strictly OUT_OF_BOUNDS.
        previous = self.previous_ball_pos
        if previous is not None:
            # 1. Sideline crossing interpolation: y crosses +1.0 or -1.0
            if not metrics.out_of_bounds:
                if (previous[1] <= FIELD_Y_MAX < ball_pos[1]) or (previous[1] >= FIELD_Y_MIN > ball_pos[1]):
                    metrics.out_of_bounds = True
                    metrics.in_bounds = False
                # 2. Backline crossing interpolation: x crosses FIELD_X_MIN (-0.4)
                elif previous[0] >= FIELD_X_MIN > ball_pos[0]:
                    metrics.out_of_bounds = True
                    metrics.in_bounds = False

            # 3. Goal line crossing at x = self.goal_x (2.8m)
            if previous[0] < self.goal_x <= ball_pos[0]:
                fraction = (self.goal_x - previous[0]) / (ball_pos[0] - previous[0])
                crossing = previous + fraction * (ball_pos - previous)
                is_inside_posts = abs(crossing[1] - self.goal_y) < self.inner_half_width
                is_under_bar = 0.0 <= crossing[2] <= self.crossbar_underside

                if not metrics.out_of_bounds and not metrics.goal_scored and is_inside_posts and is_under_bar:
                    metrics.goal_scored = True
                    if metrics.total_kicks > 1:
                        metrics.rebound_goal = True
                elif not is_inside_posts or not is_under_bar:
                    # Crossed the endline outside the posts (wide shot) or over crossbar -> OUT OF BOUNDS!
                    metrics.out_of_bounds = True
                    metrics.in_bounds = False

        self.previous_ball_pos = ball_pos.copy()

        metrics.total_time = elapsed_time
        metrics.final_ball_dist_to_goal = float(np.linalg.norm(ball_pos[0:2] - np.array([self.goal_x, self.goal_y])))
