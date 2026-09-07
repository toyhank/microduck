"""
Microduck Soccer Asset Management
Centralized path resolution for MuJoCo MJCF scenes, robot meshes, and ONNX policies.
"""

from pathlib import Path

# Root directory of the repository / project
ROOT_DIR = Path(__file__).resolve().parent.parent

# Default standalone assets directory
ASSETS_DIR = ROOT_DIR / "assets"
POLICIES_DIR = ASSETS_DIR / "policies"
SCENE_XML = ASSETS_DIR / "scene_soccer.xml"
SCENE_GK_XML = ASSETS_DIR / "scene_soccer_goalkeeper.xml"

# Legacy paths for backward compatibility fallback
LEGACY_SCENE_XML = ROOT_DIR / "microduck_rl" / "src" / "mjlab_microduck" / "robot" / "microduck" / "scene_soccer.xml"
LEGACY_POLICIES_DIR = ROOT_DIR / "microduck" / "policies"


def get_scene_xml_path() -> str:
    """Return path to the soccer scene XML file. Prefers standalone assets/."""
    if SCENE_XML.exists():
        return str(SCENE_XML)
    if LEGACY_SCENE_XML.exists():
        return str(LEGACY_SCENE_XML)
    raise FileNotFoundError(f"Soccer scene XML not found at {SCENE_XML} or {LEGACY_SCENE_XML}")


def get_goalkeeper_scene_xml_path() -> str:
    """Return path to the dual-duck soccer scene XML file (striker + goalkeeper)."""
    if SCENE_GK_XML.exists():
        return str(SCENE_GK_XML)
    raise FileNotFoundError(f"Goalkeeper soccer scene XML not found at {SCENE_GK_XML}")


def get_policy_path(policy_name: str) -> str:
    """Return path to an ONNX policy file. Prefers standalone assets/policies/."""
    candidate = POLICIES_DIR / policy_name
    if candidate.exists():
        return str(candidate)
    legacy_candidate = LEGACY_POLICIES_DIR / policy_name
    if legacy_candidate.exists():
        return str(legacy_candidate)
    raise FileNotFoundError(f"Policy '{policy_name}' not found at {candidate} or {legacy_candidate}")
