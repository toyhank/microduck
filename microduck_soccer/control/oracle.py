"""Ground-truth adapter name retained for the simulation-only oracle runner."""
from .pose_controller import PoseSoccerController, wrap


class OracleSoccerController(PoseSoccerController):
    """Caller supplies ground truth; not a vision-only controller."""
