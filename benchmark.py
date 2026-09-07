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
import json
from pathlib import Path
import math
import random
import sys
import cv2
import mujoco
import numpy as np

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from microduck_soccer.perception import BallDetector, GoalDetector
from microduck_soccer.control import SoccerStateMachine, SoccerState
from microduck_soccer.control.calibrated_visual import CalibratedVisualSoccerController
from microduck_soccer.policy import PolicyRunner, DEFAULT_POSE, KICK_DURATION_SEC
from microduck_soccer.evaluation import SoccerEvaluator, EpisodeMetrics

from microduck_soccer.assets import get_scene_xml_path, get_policy_path

XML_PATH = get_scene_xml_path()
POLICY_WALK = get_policy_path("alpha_walking.onnx")
POLICY_KICK_R = get_policy_path("ball_kick_right.onnx")
POLICY_KICK_L = get_policy_path("ball_kick_left.onnx")
POLICY_STAND = get_policy_path("alpha_stand.onnx")

def run_trial(trial_id, max_duration=12.0, seed=None, terminal_duration=1.52,
              controller='calibrated', trace=False, default_spawn=False, look_before_kick=False):
    rng = random.Random(seed)
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)

    cam_width, cam_height = 320, 240
    renderer = mujoco.Renderer(m, cam_height, cam_width)
    try:
        ball_detector = BallDetector(cam_width, cam_height, fovy_deg=90.0)
        goal_detector = GoalDetector(cam_width, cam_height, fovy_deg=90.0)

        foot_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
        foot_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_collision")
        ball_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")

        state_machine = SoccerStateMachine(mode="strict", kick_duration_sec=KICK_DURATION_SEC,
                                           terminal_duration_sec=terminal_duration)
        if controller == 'calibrated':
            state_machine = CalibratedVisualSoccerController(m, look_before_kick=look_before_kick)
        policy_runner = PolicyRunner(POLICY_WALK, POLICY_KICK_R, POLICY_KICK_L, POLICY_STAND)
        evaluator = SoccerEvaluator(
            goal_x=2.8, goal_y=0.0, goal_width=0.8,
            foot_geom_id=foot_geom_id, ball_geom_id=ball_geom_id, foot_site_id=foot_site_id
        )
        metrics = EpisodeMetrics(episode_id=trial_id)

        imu_ang_vel_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
        trunk_base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        ball_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball")

        ball_jnt_id = m.body_jntadr[ball_body_id]
        ball_qpos_adr = m.jnt_qposadr[ball_jnt_id]
        ball_qvel_adr = m.jnt_dofadr[ball_jnt_id]

        joint_qpos_indices = [int(m.jnt_qposadr[m.actuator_trnid[i, 0]]) for i in range(14)]
        joint_qvel_indices = [int(m.jnt_dofadr[m.actuator_trnid[i, 0]]) for i in range(14)]

        # Randomize initial duck yaw (+- 15 deg)
        rand_yaw = rng.uniform(-math.radians(15), math.radians(15))
        d.qpos[joint_qpos_indices] = DEFAULT_POSE
        d.qpos[0] = 0.0
        d.qpos[1] = 0.0
        d.qpos[2] = 0.12
        d.qpos[3] = math.cos(rand_yaw / 2.0)
        d.qpos[4] = 0.0
        d.qpos[5] = 0.0
        d.qpos[6] = math.sin(rand_yaw / 2.0)
        d.qvel[:] = 0.0

        # Randomize ball position (x in [1.0, 1.25], y in [-0.12, 0.12])
        rand_bx = rng.uniform(1.0, 1.25)
        rand_by = rng.uniform(-0.12, 0.12)
        if default_spawn:
            d.qpos[3:7] = [1., 0., 0., 0.]
            rand_bx, rand_by = 1.2, .1
        d.qpos[ball_qpos_adr:ball_qpos_adr + 7] = [rand_bx, rand_by, 0.035, 1, 0, 0, 0]
        d.qvel[ball_qvel_adr:ball_qvel_adr + 6] = 0.0
        mujoco.mj_forward(m, d)

        metrics.initial_ball_dist = math.sqrt(rand_bx**2 + rand_by**2)

        step_counter = 0
        sim_time = 0.0
        last_trace = -1.
        last_state = None

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

            # Encoders and IMU orientation are the only non-image control inputs.
            adr = m.sensor_adr[imu_ang_vel_id]
            sensor_ang_vel = d.sensordata[adr:adr+3].copy().astype(np.float32)
            trunk_quat = d.sensor('orientation').data.copy().astype(np.float32)
            current_qpos = d.qpos[joint_qpos_indices].copy().astype(np.float32)
            current_qvel = d.qvel[joint_qvel_indices].copy().astype(np.float32)
            sensor_inputs = dict(joint_positions=current_qpos, orientation=trunk_quat,
                                 angular_velocity=sensor_ang_vel, new_frame=step_counter % 5 == 0)
            state, cmd_vx, cmd_vy, cmd_vyaw, active_mode, trigger_kick = state_machine.update(
                ball_det, goal_det, sim_time, **(sensor_inputs if controller == 'calibrated' else {})
            )
            if trace and (state != last_state or sim_time-last_trace >= 1.):
                if controller == 'calibrated':
                    loc = state_machine.localizer
                    rel = None if loc.ball_xy is None else (loc.ball_xy-loc.xy).round(3).tolist()
                    print(f'{sim_time:.2f} {state} estimated_ball={rel} yaw={loc.yaw:.3f} ball_age={sim_time-loc.ball_seen:.2f} cmd={(cmd_vx,cmd_vy,cmd_vyaw)}', flush=True)
                last_trace, last_state = sim_time, state

            reached = (state_machine.motion.positioning_settle or trigger_kick) if controller == 'calibrated' else (
                state in [SoccerState.TERMINAL_APPROACH, SoccerState.ALIGN_KICK, SoccerState.KICK])
            if reached:
                metrics.approach_success = True
                if metrics.time_to_kick == 0.0 and state == SoccerState.KICK:
                    metrics.time_to_kick = sim_time

            # Policy step
            command_13d = np.zeros(13, dtype=np.float32)
            if controller == 'calibrated':
                command_13d[3:5] = state_machine.head_command
            if active_mode == "walk":
                command_13d[0] = cmd_vx
                command_13d[1] = cmd_vy
                command_13d[2] = cmd_vyaw

            target_qpos = policy_runner.step(
                active_mode, sensor_ang_vel, trunk_quat, current_qpos, current_qvel, command_13d
            )
            d.ctrl[:14] = target_qpos

            for _ in range(10):
                mujoco.mj_step(m, d)
                evaluator.evaluate_step(d, trunk_base_id, ball_body_id, foot_site_id, metrics,
                                        float(d.time), ball_qvel_adr=ball_qvel_adr,
                                        active_mode=active_mode)

            step_counter += 1
            sim_time = float(d.time)

        metrics.goal_with_kick_contact = metrics.goal_scored and metrics.kick_contact
        if controller == 'calibrated':
            metrics.look_attempts = state_machine.look_count
            metrics.look_confirmations = state_machine.look_confirmations
            metrics.final_state = state_machine.state
        return metrics
    finally:
        renderer.close()


def main():
    parser = argparse.ArgumentParser(description="Microduck Soccer Benchmark Suite")
    parser.add_argument("--trials", type=int, default=10, help="Number of benchmark trials to run")
    parser.add_argument("--duration", type=float, default=12.0, help="Max duration per trial in seconds")
    parser.add_argument("--output", type=Path, help="Write per-trial metrics as JSON")
    parser.add_argument("--terminal-duration", type=float, default=1.52, help="Legacy controller blind advance duration")
    parser.add_argument("--seed", type=int, default=0, help="Reproducible initial conditions")
    parser.add_argument('--controller', choices=['calibrated', 'legacy'], default='calibrated')
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--look-before-kick', action='store_true')
    parser.add_argument('--default-spawn', action='store_true', help='Use the GUI initial pose and ball placement')
    args = parser.parse_args()
    if args.trials <= 0 or args.duration <= 0 or args.terminal_duration <= 0:
        parser.error("trials and durations must be positive")

    print("=" * 60)
    print(f"  🏁 Running Microduck Soccer Benchmark ({args.trials} Trials, {args.duration:.1f}s Max)")
    print("=" * 60)

    results = []
    for i in range(1, args.trials + 1):
        print(f"Trial {i:2d}/{args.trials}... ", end="", flush=True)
        m = run_trial(i, max_duration=args.duration, seed=args.seed + i - 1, terminal_duration=args.terminal_duration,
                      controller=args.controller, trace=args.trace, default_spawn=args.default_spawn,
                      look_before_kick=args.look_before_kick)
        results.append(m)
        status = "GOAL ⚽" if m.goal_scored else ("KICKED 👟" if m.kick_contact else ("APPROACHED 🚶" if m.approach_success else "MISSED ❌"))
        print(f"[{status}] Time: {m.total_time:.2f}s, Kick contact: {m.kick_contact}, Goal: {m.goal_scored}")

    n = len(results)
    det_rate = sum(1 for r in results if r.ball_detected) / n * 100.0
    app_rate = sum(1 for r in results if r.approach_success) / n * 100.0
    foot_rate = sum(1 for r in results if r.foot_contact) / n * 100.0
    kick_rate = sum(1 for r in results if r.kick_contact) / n * 100.0
    goal_rate = sum(1 for r in results if r.goal_scored) / n * 100.0
    kick_goal_rate = sum(r.goal_with_kick_contact for r in results) / n * 100.0
    fall_rate = sum(1 for r in results if r.fallen) / n * 100.0

    kick_times = [r.time_to_kick for r in results if r.time_to_kick > 0]
    mean_time_kick = np.mean(kick_times) if kick_times else 0.0

    print("\n" + "=" * 60)
    print("                  BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Total Trials:               {n}")
    print(f"  Ball Detection Rate:        {det_rate:.1f}%")
    print(f"  Strike Positioning Entry:    {app_rate:.1f}%")
    print(f"  Any Right-Foot Contact Rate: {foot_rate:.1f}%")
    print(f"  Kick Contact Rate:          {kick_rate:.1f}%")
    print(f"  Goal Scoring Rate:          {goal_rate:.1f}%")
    print(f"  Goals with Kick Contact:   {kick_goal_rate:.1f}%")
    print(f"  Mean Time to Kick:          {mean_time_kick:.2f} s")
    print(f"  Fall Rate:                  {fall_rate:.1f}%")
    print("=" * 60)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"seed": args.seed, "duration": args.duration, 'controller': args.controller,
                   'look_before_kick': args.look_before_kick,
                   "terminal_duration": args.terminal_duration,
                   "trials": [vars(r) for r in results]}
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
