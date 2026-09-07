"""
Soccer Field Geometry and Boundary Evaluation.

Defines standard pitch dimensions, boundaries, and spatial state checking.
Coordinates are in meters relative to the pitch origin.
"""

from enum import Enum
import numpy as np

# Field Boundary Dimensions (in meters)
FIELD_X_MIN = -0.4
FIELD_X_MAX = 2.8
FIELD_Y_MIN = -1.0
FIELD_Y_MAX = 1.0

# Goal Dimensions
GOAL_X = 2.8
GOAL_Y = 0.0
GOAL_WIDTH = 0.8
GOAL_POST_RADIUS = 0.02
GOAL_INNER_WIDTH = GOAL_WIDTH - 2 * GOAL_POST_RADIUS  # 0.76m (y in [-0.38, 0.38])
GOAL_INNER_HALF_WIDTH = GOAL_INNER_WIDTH / 2.0         # 0.38m
GOAL_POST_Y_LEFT = 0.4
GOAL_POST_Y_RIGHT = -0.4
GOAL_NET_DEPTH = 0.30  # Net extends to x = 3.10m
GOAL_CROSSBAR_HEIGHT = 0.35
GOAL_CROSSBAR_RADIUS = 0.02
GOAL_CROSSBAR_UNDERSIDE = GOAL_CROSSBAR_HEIGHT - GOAL_CROSSBAR_RADIUS  # 0.33m
BALL_RADIUS = 0.035

# Outer Curb Bounds (Buffer zone before barrier)
OUTER_CURB_X_MIN = -0.6
OUTER_CURB_X_MAX = 3.25
OUTER_CURB_Y_MIN = -1.25
OUTER_CURB_Y_MAX = 1.25


class BallFieldStatus(str, Enum):
    GOAL = "GOAL"
    IN_BOUNDS = "IN_BOUNDS"
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"


def check_ball_field_status(ball_xy, z=0.035):
    """
    Evaluate whether the soccer ball is currently in-bounds, scored in goal, or out-of-bounds.
    
    Args:
        ball_xy: [x, y] or [x, y, z] position in world coordinates.
        z: vertical height of the ball (optional).
        
      Returns:
        BallFieldStatus: GOAL, IN_BOUNDS, or OUT_OF_BOUNDS.
    """
    x = float(ball_xy[0])
    y = float(ball_xy[1])
    if len(ball_xy) >= 3:
        z = float(ball_xy[2])

    # 1. Goal net volume check: strictly inside inner posts, under crossbar, within net depth
    if GOAL_X < x <= (GOAL_X + GOAL_NET_DEPTH):
        if abs(y - GOAL_Y) < GOAL_INNER_HALF_WIDTH and (z <= GOAL_CROSSBAR_UNDERSIDE + 0.01):
            return BallFieldStatus.GOAL

    # 2. Field bounds check (strictly within pitch boundaries)
    if (FIELD_X_MIN <= x <= FIELD_X_MAX) and (FIELD_Y_MIN <= y <= FIELD_Y_MAX):
        return BallFieldStatus.IN_BOUNDS

    # 3. Otherwise out of bounds (passed touchline, backline, or endline outside goal)
    return BallFieldStatus.OUT_OF_BOUNDS


def is_in_bounds(ball_xy):
    """Check if ball is within pitch boundary lines."""
    return check_ball_field_status(ball_xy) == BallFieldStatus.IN_BOUNDS


def is_goal(ball_xy):
    """Check if ball is inside the goal net."""
    return check_ball_field_status(ball_xy) == BallFieldStatus.GOAL


def is_out_of_bounds(ball_xy):
    """Check if ball has crossed outside the field boundary (and not in goal)."""
    return check_ball_field_status(ball_xy) == BallFieldStatus.OUT_OF_BOUNDS
