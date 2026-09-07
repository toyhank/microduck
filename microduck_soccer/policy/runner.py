"""
RL Policy Runner adhering to official Microduck robotd control chain.
Features:
- Action scaling matching official robotd (walking=0.9, standing=1.0, kick=1.0).
- First-order low-pass filter (mjlab ACTION_LOW_PASS_{HEAD,LEG}_ALPHA: legs=0.7, head=0.5).
- Kick duration standard: 0.5 seconds (25 steps at 50Hz).
"""

import numpy as np
import onnxruntime as ort

# Local robot inference does not need platform telemetry. Disable it before
# creating any sessions (also applies in each benchmark worker process).
ort.disable_telemetry_events()

DEFAULT_POSE = np.array([
    0.0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0.0, 0.0,
    0.0, 0.0873, 0.4579, 0.0049, -0.4530
], dtype=np.float32)

# Joint groups for low-pass filtering:
# Joints 0..4: left leg, 5..8: neck/head, 9..13: right leg
LEG_INDICES = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
HEAD_INDICES = [5, 6, 7, 8]

# Official robotd timing and scaling defaults
WALK_ACTION_SCALE = 0.9
STAND_ACTION_SCALE = 1.0
KICK_ACTION_SCALE = 1.0
LEGS_LOWPASS_ALPHA = 0.7
HEAD_LOWPASS_ALPHA = 0.5
KICK_DURATION_SEC = 0.5  # Official 0.5s kick window

def quat_rotate_inverse(quat, vec):
    w, x, y, z = quat
    xyz = np.array([x, y, z])
    t = np.cross(xyz, vec) * 2
    return vec - w * t + np.cross(xyz, t)

class PolicyRunner:
    def __init__(self, walk_path, kick_r_path, kick_l_path=None, stand_path=None):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session_walk = ort.InferenceSession(walk_path, sess_options=options)
        self.session_kick_r = ort.InferenceSession(kick_r_path, sess_options=options)
        self.session_kick_l = ort.InferenceSession(kick_l_path, sess_options=options) if kick_l_path else None
        self.session_stand = ort.InferenceSession(stand_path, sess_options=options) if stand_path else None

        self.last_action = np.zeros(14, dtype=np.float32)
        self.filtered_target = DEFAULT_POSE.copy()
        self.world_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    def step(self, active_mode, sensor_ang_vel, trunk_quat, current_qpos, current_qvel, command_13d):
        """
        Step one 50Hz control cycle.
        Returns the target 14 joint positions with low-pass filtering applied.
        """
        proj_grav = quat_rotate_inverse(trunk_quat, self.world_gravity).astype(np.float32)
        rel_qpos = (current_qpos - DEFAULT_POSE).astype(np.float32)

        # Select session and action scale
        if active_mode == "walk":
            session = self.session_walk
            scale = WALK_ACTION_SCALE
        elif active_mode in ("kick", "kick_right"):
            session = self.session_kick_r
            scale = KICK_ACTION_SCALE
        elif active_mode == "kick_left":
            session = self.session_kick_l or self.session_kick_r
            scale = KICK_ACTION_SCALE
        elif active_mode == "stand":
            session = self.session_stand or self.session_walk
            scale = STAND_ACTION_SCALE
        else:
            session = self.session_walk
            scale = WALK_ACTION_SCALE

        obs = np.concatenate([
            sensor_ang_vel,
            proj_grav,
            rel_qpos,
            current_qvel,
            self.last_action,
            command_13d
        ]).astype(np.float32).reshape(1, -1)

        raw_action = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: obs})[0].squeeze(0).astype(np.float32)
        self.last_action = raw_action.copy()

        # Compute raw target
        raw_target = DEFAULT_POSE + raw_action * scale

        # First-order low-pass filter: target = (1 - alpha) * prev + alpha * raw
        new_filtered = self.filtered_target.copy()
        for idx in LEG_INDICES:
            new_filtered[idx] = (1.0 - LEGS_LOWPASS_ALPHA) * self.filtered_target[idx] + LEGS_LOWPASS_ALPHA * raw_target[idx]
        for idx in HEAD_INDICES:
            new_filtered[idx] = (1.0 - HEAD_LOWPASS_ALPHA) * self.filtered_target[idx] + HEAD_LOWPASS_ALPHA * raw_target[idx]

        self.filtered_target = new_filtered
        return np.clip(self.filtered_target, DEFAULT_POSE - 1.5, DEFAULT_POSE + 1.5)
