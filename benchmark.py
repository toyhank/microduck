#!/usr/bin/env python3
"""
Automated Benchmarking Suite for Microduck Soccer.
Evaluates closed-loop visual servoing and kicking across randomized trials.

Metrics tracked:
- Ball Detection Rate (%)
- Approach Success Rate (%)
- Kick Contact Rate (%)
- Goal Scoring Rate (%)
- Mean Time to Kick (s)
- Fall Rate (%)
"""

import argparse
import math
import random
import sys
import time
import cv2
import mujoco
import numpy as np

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from microduck_soccer.perception import BallDetector, GoalDetector
from microduck_soccer.control import SoccerStateMachine, SoccerState
from microduck_soccer.policy import PolicyRunner, DEFAULT_POSE, KICK_DURATION_SEC
from microduck_soccer.evaluation import SoccerEvaluator, EpisodeMetrics

XML_PATH = "microduck_rl/src/mjlab_microduck/robot/microduck/scene_soccer.xml"
POLICY_WALK = "microduck/policies/alpha_walking.onnx"
POLICY_KICK_R = "microduck/policies/ball_kick_right.onnx"
POLICY_KICK_L = "microduck/policies/ball_kick_left.onnx"
POLICY_STAND = "microduck/policies/alpha_stand.onnx"

def run_trial(trial_id, max_duration=12.0):
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)

    cam_width, cam_height = 320, 240
    renderer = mujoco.Renderer(m, cam_height, cam_width)
    ball_detector = BallDetector(cam_width, cam_height, fovy_deg=90.0)
    goal_detector = GoalDetector(cam_width, cam_height, fovy_deg=90.0)

    state_machine = SoccerStateMachine(mode="strict", kick_duration_sec=KICK_DURATION_SEC)
    policy_runner = PolicyRunner(POLICY_WALK, POLICY_KICK_R, POLICY_KICK_L, POLICY_STAND)
    evaluator = SoccerEvaluator(goal_x=2.8, goal_y=0.0, goal_width=0.8)
    metrics = EpisodeMetrics(episode_id=trial_id)

    imu_ang_vel_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
    trunk_base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    ball_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball")
    right_foot_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "sole_right")

    ball_jnt_id = m.body_jntadr[ball_body_id]
    ball_qpos_adr = m.jnt_qposadr[ball_jnt_id]
    ball_qvel_adr = m.jnt_dofadr[ball_jnt_id]

    joint_qpos_indices = [int(m.jnt_qposadr[m.actuator_trnid[i, 0]]) for i in range(14)]
    joint_qvel_indices = [int(m.jnt_dofadr[m.actuator_trnid[i, 0]]) for i in range(14)]

    # Randomize initial duck yaw (+- 20 deg)
    rand_yaw = random.uniform(-math.radians(20), math.radians(20))
    d.qpos[joint_qpos_indices] = DEFAULT_POSE
    d.qpos[2] = 0.12
    d.qpos[3] = math.cos(rand_yaw / 2.0)
    d.qpos[6] = math.sin(rand_yaw / 2.0)

    # Randomize ball position (x in [1.0, 1.3], y in [-0.2, 0.2])
    rand_bx = random.uniform(1.0, 1.3)
    rand_by = random.uniform(-0.2, 0.2)
    d.qpos[ball_qpos_adr:ball_qpos_adr + 7] = [rand_bx, rand_by, 0.035, 1, 0, 0, 0]
    d.qvel[ball_qvel_adr:ball_qvel_adr + 6] = 0.0
    mujoco.mj_forward(m, d)

    metrics.initial_ball_dist = math.sqrt(rand_bx**2 + rand_by**2)

    step_counter = 0
    sim_time = 0.0
    dt_step = 0.02

    ball_det = ball_detector.detect(np.zeros((cam_height, cam_width, 3), dtype=np.uint8))
    goal_det = goal_detector.detect(np.zeros((cam_height, cam_width, 3), dtype=np.uint8))

    while sim_time < max_duration:
        # Vision at 10Hz
        if step_counter % 5 == 0:
            renderer.update_scene(d, camera="egocentric")
            img_rgb = renderer.render()
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            ball_det = ball_detector.detect(img_bgr)
            goal_det = goal_detector.detect(img_bgr)

            if ball_det.visible:
                metrics.ball_detected = True

        # Pure vision FSM
        state, cmd_vx, cmd_vyaw, active_mode, trigger_kick = state_machine.update(
            ball_det, goal_det, sim_time
        )

        if state == SoccerState.ALIGN_KICK or state == SoccerState.KICK:
            metrics.approach_success = True
            if metrics.time_to_kick == 0.0:
                metrics.time_to_kick = sim_time

        # Policy step
        adr = m.sensor_adr[imu_ang_vel_id]
        sensor_ang_vel = d.sensordata[adr:adr+3].copy().astype(np.float32)
        trunk_quat = d.xquat[trunk_base_id].copy().astype(np.float32)
        current_qpos = d.qpos[joint_qpos_indices].copy().astype(np.float32)
        current_qvel = d.qvel[joint_qvel_indices].copy().astype(np.float32)

        command_13d = np.zeros(13, dtype=np.float32)
        if active_mode == "walk":
            command_13d[0] = cmd_vx
            command_13d[2] = cmd_vyaw

        target_qpos = policy_runner.step(
            active_mode, sensor_ang_vel, trunk_quat, current_qpos, current_qvel, command_13d
        )
        d.ctrl[:14] = target_qpos

        for _ in range(10):
            mujoco.mj_step(m, d)

        # Ground truth evaluation
        evaluator.evaluate_step(d, trunk_base_id, ball_body_id, right_foot_id, metrics, sim_time)

        if metrics.goal_scored:
            break

        step_counter += 1
        sim_time += dt_step

    return metrics

def main():
    parser = argparse.ArgumentParser(description="Microduck Soccer Benchmark Suite")
    parser.add_argument("--trials", type=int, default=10, help="Number of benchmark trials to run")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  🏁 Running Microduck Soccer Benchmark ({args.trials} Trials)")
    print("=" * 60)

    results = []
    for i in range(1, args.trials + 1):
        print(f"Trial {i:2d}/{args.trials}... ", end="", flush=True)
        m = run_trial(i)
        results.append(m)
        status = "GOAL ⚽" if m.goal_scored else ("KICKED 👟" if m.kick_contact else ("APPROACHED 🚶" if m.approach_success else "MISSED ❌"))
        print(f"[{status}] Time: {m.total_time:.2f}s, Kick contact: {m.kick_contact}, Goal: {m.goal_scored}")

    n = len(results)
    det_rate = sum(1 for r in results if r.ball_detected) / n * 100.0
    app_rate = sum(1 for r in results if r.approach_success) / n * 100.0
    kick_rate = sum(1 for r in results if r.kick_contact) / n * 100.0
    goal_rate = sum(1 for r in results if r.goal_scored) / n * 100.0
    fall_rate = sum(1 for r in results if r.fallen) / n * 100.0

    kick_times = [r.time_to_kick for r in results if r.time_to_kick > 0]
    mean_time_kick = np.mean(kick_times) if kick_times else 0.0

    print("\n" + "=" * 60)
    print("                  BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Total Trials:               {n}")
    print(f"  Ball Detection Rate:        {det_rate:.1f}%")
    print(f"  Approach Success Rate:      {app_rate:.1f}%")
    print(f"  Kick Contact Rate:          {kick_rate:.1f}%")
    print(f"  Goal Scoring Rate:          {goal_rate:.1f}%")
    print(f"  Mean Time to Kick:          {mean_time_kick:.2f} s")
    print(f"  Fall Rate:                  {fall_rate:.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()
