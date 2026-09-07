"""
Microduck Soccer: Closed-loop Visual Servoing and Autonomous Soccer System.
"""

__version__ = "1.0.0"

from .assets import get_scene_xml_path, get_policy_path
from .field import (
    FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX,
    GOAL_X, GOAL_Y, GOAL_WIDTH,
    BallFieldStatus, check_ball_field_status, is_in_bounds, is_goal, is_out_of_bounds,
)
