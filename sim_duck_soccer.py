#!/usr/bin/env python3
"""
Microduck Soccer Simulation (MuJoCo)
Simulator-first closed-loop visual servoing and dynamic bipedal kicking system.

Modes:
  --mode strict (default): Monocular vision + encoder/IMU odometry, no world-state inputs.
                           No ball teleportation, physical dynamic kick, official 0.5s kick duration.
  --mode demo:             Demonstration mode with oracle alignment assistance.
"""

import argparse
import math
import sys
import time
import cv2
import mujoco
import mujoco.viewer
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from microduck_soccer.perception import BallDetector, GoalDetector
from microduck_soccer.control import SoccerStateMachine, SoccerState
from microduck_soccer.control.calibrated_visual import CalibratedVisualSoccerController
from microduck_soccer.policy import PolicyRunner, DEFAULT_POSE, KICK_DURATION_SEC
from microduck_soccer.evaluation import SoccerEvaluator, EpisodeMetrics

from microduck_soccer.assets import get_scene_xml_path, get_goalkeeper_scene_xml_path, get_policy_path
from microduck_soccer.control.goalkeeper import GoalkeeperController

XML_PATH = get_scene_xml_path()
POLICY_WALK = get_policy_path("alpha_walking.onnx")
POLICY_KICK_R = get_policy_path("ball_kick_right.onnx")
POLICY_KICK_L = get_policy_path("ball_kick_left.onnx")
POLICY_STAND = get_policy_path("alpha_stand.onnx")

def parse_args():
    parser = argparse.ArgumentParser(description="Microduck Autonomous Soccer Simulation")
    parser.add_argument("--mode", type=str, default="strict", choices=["strict", "demo"],
                        help="Operation mode: 'strict' (pure vision, no teleport) or 'demo'")
    parser.add_argument("--headless", action="store_true", help="Run without graphical GUI windows")
    parser.add_argument('--look-before-kick', action='store_true', help='Experimental: look down, confirm ball, restore head, then recheck kick readiness')
    parser.add_argument("--duration", type=float, help="Stop after this many simulation seconds")
    parser.add_argument("--terminal-duration", type=float, default=1.52, help="Legacy demo blind advance duration; unused in strict mode")
    parser.add_argument("--goalkeeper", action="store_true", help="Add Goalkeeper Microduck to defend the goal")
    args = parser.parse_args()
    if args.terminal_duration <= 0 or (args.duration is not None and args.duration <= 0):
        parser.error("durations must be positive")
    return args

def main():
    args = parse_args()

    print("=" * 65)
    print("   🦆 Microduck Soccer: Closed-Loop Visual Servoing ⚽")
    description = 'Camera + encoders/IMU' if args.mode == 'strict' else 'Assisted demo'
    print(f"   Mode: {args.mode.upper()} ({description})")
    print("=" * 65)

    xml_path = get_goalkeeper_scene_xml_path() if args.goalkeeper else XML_PATH
    try:
        m = mujoco.MjModel.from_xml_path(xml_path)
        d = mujoco.MjData(m)
    except Exception as e:
        print(f"Failed to load MuJoCo model: {e}")
        return

    # 1. Perception modules
    cam_width, cam_height = 320, 240
    renderer = mujoco.Renderer(m, cam_height, cam_width)
    ball_detector = BallDetector(cam_width, cam_height, fovy_deg=90.0)
    goal_detector = GoalDetector(cam_width, cam_height, fovy_deg=90.0)

    # 2. Control and Policy modules
    state_machine = SoccerStateMachine(mode=args.mode, kick_duration_sec=KICK_DURATION_SEC,
                                       terminal_duration_sec=args.terminal_duration)
    if args.mode == 'strict':
        state_machine = CalibratedVisualSoccerController(m, look_before_kick=args.look_before_kick)
    policy_runner = PolicyRunner(POLICY_WALK, POLICY_KICK_R, POLICY_KICK_L, POLICY_STAND)

    # 3. Independent Evaluator (ground truth only used for evaluation logging, NOT control)
    foot_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
    foot_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_collision")
    ball_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    gk_geom_ids = [i for i in range(m.ngeom) if m.geom(i).name.startswith("gk_")] if args.goalkeeper else None
    evaluator = SoccerEvaluator(
        goal_x=2.8, goal_y=0.0, goal_width=0.8,
        foot_geom_id=foot_geom_id, ball_geom_id=ball_geom_id, foot_site_id=foot_site_id,
        gk_geom_ids=gk_geom_ids
    )
    metrics = EpisodeMetrics()

    # Hardware sensor indices
    imu_ang_vel_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
    trunk_base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    ball_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball")

    ball_jnt_id = m.body_jntadr[ball_body_id]
    ball_qpos_adr = m.jnt_qposadr[ball_jnt_id]
    ball_qvel_adr = m.jnt_dofadr[ball_jnt_id]

    joint_qpos_indices = [int(m.jnt_qposadr[m.actuator_trnid[i, 0]]) for i in range(14)]
    joint_qvel_indices = [int(m.jnt_dofadr[m.actuator_trnid[i, 0]]) for i in range(14)]

    # Initial Duck state
    d.qpos[joint_qpos_indices] = DEFAULT_POSE
    d.qpos[2] = 0.12  # trunk height
    d.qpos[3:7] = [1, 0, 0, 0]  # identity quaternion

    # Goalkeeper setup
    if args.goalkeeper:
        gk_policy_runner = PolicyRunner(POLICY_WALK, POLICY_KICK_R, POLICY_KICK_L, POLICY_STAND)
        gk_controller = GoalkeeperController()
        gk_trunk_base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gk_trunk_base")
        gk_jnt_id = m.body_jntadr[gk_trunk_base_id]
        gk_qpos_adr = m.jnt_qposadr[gk_jnt_id]
        gk_imu_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "gk_imu_ang_vel")
        gk_imu_adr = m.sensor_adr[gk_imu_id]

        gk_joint_qpos_indices = [int(m.jnt_qposadr[m.actuator_trnid[i, 0]]) for i in range(15, 29)]
        gk_joint_qvel_indices = [int(m.jnt_dofadr[m.actuator_trnid[i, 0]]) for i in range(15, 29)]

        d.qpos[gk_joint_qpos_indices] = DEFAULT_POSE
        d.qpos[gk_qpos_adr:gk_qpos_adr+3] = [2.65, 0.0, 0.12]
        d.qpos[gk_qpos_adr+3:gk_qpos_adr+7] = [0, 0, 0, 1]

    mujoco.mj_forward(m, d)

    if not args.headless:
        cv2.namedWindow("Microduck Egocentric Vision", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Microduck Egocentric Vision", 640, 480)

    step_counter = 0
    reported_goal = False

    ball_det = ball_detector.detect(np.zeros((cam_height, cam_width, 3), dtype=np.uint8))
    goal_det = goal_detector.detect(np.zeros((cam_height, cam_width, 3), dtype=np.uint8))

    viewer_ctx = mujoco.viewer.launch_passive(m, d) if not args.headless else None

    print(f"\n[Running] Policy: 50Hz, Vision: 10Hz, Kick window: {KICK_DURATION_SEC:.1f}s")

    try:
        while args.duration is None or d.time < args.duration:
            if viewer_ctx and not viewer_ctx.is_running():
                break

            step_start = time.monotonic()
            elapsed_time = float(d.time)

            # ====================================================
            # 1. Perception Layer (Runs at 10Hz = every 5 steps)
            # ====================================================
            if step_counter % 5 == 0:
                renderer.update_scene(d, camera="egocentric")
                img_rgb = renderer.render()
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                ball_det = ball_detector.detect(img_bgr)
                goal_det = goal_detector.detect(img_bgr)

                if ball_det.visible:
                    metrics.ball_detected = True

                # Draw OpenCV visual telemetry overlay
                if not args.headless:
                    if ball_det.visible:
                        cv2.circle(img_bgr, (ball_det.cx, ball_det.cy), int(ball_det.radius), (0, 165, 255), 2)
                        cv2.circle(img_bgr, (ball_det.cx, ball_det.cy), 4, (0, 0, 255), -1)
                        cv2.putText(img_bgr, f"BALL ({ball_det.distance:.2f}m, {math.degrees(ball_det.bearing):.1f}deg)",
                                    (ball_det.cx - 50, ball_det.cy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

                    if goal_det.visible:
                        cv2.rectangle(img_bgr, (goal_det.cx - int(goal_det.width/2), goal_det.cy - int(goal_det.height/2)),
                                      (goal_det.cx + int(goal_det.width/2), goal_det.cy + int(goal_det.height/2)), (255, 100, 0), 2)
                        cv2.putText(img_bgr, f"GOAL ({math.degrees(goal_det.bearing):.1f}deg)",
                                    (goal_det.cx - 30, goal_det.cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 200, 0), 1)

                    # HUD status
                    cv2.putText(img_bgr, f"STATE: {state_machine.state}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(img_bgr, f"Mode: {args.mode.upper()}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                    if ball_det.visible:
                        cv2.putText(img_bgr, f"Est Ball Dist: {ball_det.distance:.2f}m", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    if args.mode == 'strict':
                        loc = state_machine.localizer
                        source = 'CAMERA' if ball_det.visible else (
                            f'TRACKED {elapsed_time-loc.ball_seen:.1f}s' if loc.ball_xy is not None else 'SEARCHING')
                        cv2.putText(img_bgr, f'Ball: {source}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
                    if args.goalkeeper:
                        cv2.putText(img_bgr, f"GK: {gk_controller.state.value}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 1)

                    cv2.drawMarker(img_bgr, (160, 120), (180, 180, 180), cv2.MARKER_CROSS, 12, 1)

                    if state_machine.state == SoccerState.CELEBRATE:
                        overlay = img_bgr.copy()
                        cv2.rectangle(overlay, (30, 80), (290, 160), (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.6, img_bgr, 0.4, 0, img_bgr)
                        cv2.putText(img_bgr, "GOOOOAL! ⚽🦆", (55, 120), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)
                        cv2.putText(img_bgr, "Microduck Scored!", (70, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

                    cv2.imshow("Microduck Egocentric Vision", img_bgr)
                    cv2.waitKey(1)

            # ====================================================
            # 2. Control & State Machine (Strict Vision Only)
            # ====================================================
            adr = m.sensor_adr[imu_ang_vel_id]
            sensor_ang_vel = d.sensordata[adr:adr+3].copy().astype(np.float32)
            trunk_quat = d.sensor('orientation').data.copy().astype(np.float32)
            current_qpos = d.qpos[joint_qpos_indices].copy().astype(np.float32)
            current_qvel = d.qvel[joint_qvel_indices].copy().astype(np.float32)
            sensor_inputs = dict(joint_positions=current_qpos, orientation=trunk_quat,
                                 angular_velocity=sensor_ang_vel, new_frame=step_counter % 5 == 0)
            state, cmd_vx, cmd_vy, cmd_vyaw, active_mode, trigger_kick = state_machine.update(
                ball_det, goal_det, elapsed_time, **(sensor_inputs if args.mode == 'strict' else {})
            )
            if trigger_kick:
                metrics.approach_success = True
                if metrics.time_to_kick == 0:
                    metrics.time_to_kick = elapsed_time
                print(f'[Kick] t={elapsed_time:.2f}s: stance settled and shot aligned', flush=True)

            # In DEMO mode only: allow optional teleport if user specifically requested demo mode
            if args.mode == "demo" and trigger_kick:
                # Oracle positioning in demo mode only
                trunk_pos = d.xpos[trunk_base_id]
                qw, qx, qy, qz = d.xquat[trunk_base_id]
                yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                bx = trunk_pos[0] + math.cos(yaw) * 0.09 - math.sin(yaw) * (-0.042)
                by = trunk_pos[1] + math.sin(yaw) * 0.09 + math.cos(yaw) * (-0.042)
                d.qpos[ball_qpos_adr:ball_qpos_adr + 7] = [bx, by, 0.035, 1, 0, 0, 0]
                d.qvel[ball_qvel_adr:ball_qvel_adr + 6] = 0.0
                mujoco.mj_forward(m, d)

            # ====================================================
            # 3. Policy Execution (50Hz = every step)
            # ====================================================
            command_13d = np.zeros(13, dtype=np.float32)
            if args.mode == 'strict':
                command_13d[3:5] = state_machine.head_command
            if active_mode == "walk":
                command_13d[0] = cmd_vx
                command_13d[1] = cmd_vy
                command_13d[2] = cmd_vyaw
            elif state == SoccerState.CELEBRATE:
                command_13d[4] = np.sin(elapsed_time * 6.0) * 0.2  # Head nod

            target_qpos = policy_runner.step(
                active_mode, sensor_ang_vel, trunk_quat, current_qpos, current_qvel, command_13d
            )
            d.ctrl[:14] = target_qpos

            if args.goalkeeper:
                gk_ball_pos = d.xpos[ball_body_id].copy()
                gk_ball_vel = d.qvel[ball_qvel_adr:ball_qvel_adr + 3].copy()
                gk_trunk_pos = d.xpos[gk_trunk_base_id].copy()

                gk_mode, gk_vx, gk_vy, gk_vyaw = gk_controller.update(
                    gk_ball_pos, gk_ball_vel, gk_trunk_pos, elapsed_time
                )

                gk_sensor_ang_vel = d.sensordata[gk_imu_adr:gk_imu_adr+3].copy().astype(np.float32)
                gk_trunk_quat = d.sensor('gk_orientation').data.copy().astype(np.float32)
                gk_current_qpos = d.qpos[gk_joint_qpos_indices].copy().astype(np.float32)
                gk_current_qvel = d.qvel[gk_joint_qvel_indices].copy().astype(np.float32)

                gk_cmd_13d = np.zeros(13, dtype=np.float32)
                if gk_mode == "walk":
                    gk_cmd_13d[0] = gk_vx
                    gk_cmd_13d[1] = gk_vy
                    gk_cmd_13d[2] = gk_vyaw

                gk_target_qpos = gk_policy_runner.step(
                    gk_mode, gk_sensor_ang_vel, gk_trunk_quat, gk_current_qpos, gk_current_qvel, gk_cmd_13d
                )
                d.ctrl[15:29] = gk_target_qpos

            # Physics decimation (10 steps of 0.002s = 0.02s / 50Hz)
            for _ in range(10):
                mujoco.mj_step(m, d)
                evaluator.evaluate_step(d, trunk_base_id, ball_body_id, foot_site_id, metrics,
                                        float(d.time), ball_qvel_adr=ball_qvel_adr,
                                        active_mode=active_mode)

            # ====================================================
            # 4. Independent Evaluation (Strictly outside controller)
            # ====================================================
            if metrics.goal_scored and not reported_goal:
                reported_goal = True
                if args.mode == "demo":
                    state_machine.goal_scored = True
                print("\n⚽ [EVALUATOR] Goal confirmed! Ball crossed goal line!")

            if viewer_ctx:
                viewer_ctx.sync()

            step_counter += 1
            elapsed = time.monotonic() - step_start
            if not args.headless and 0.02 - elapsed > 0:
                time.sleep(0.02 - elapsed)

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    finally:
        renderer.close()
        if viewer_ctx:
            viewer_ctx.close()
        if not args.headless:
            cv2.destroyAllWindows()
        summary_str = (f"\n[Summary] Sim time: {d.time:.2f}s, Goal: {metrics.goal_scored}, "
                       f"Kick contact: {metrics.kick_contact}, Any foot contact: {metrics.foot_contact}, "
                       f"Minimum foot-site distance: {metrics.min_foot_ball_distance:.3f}m")
        if args.goalkeeper:
            summary_str += f", GK Saved: {metrics.shot_saved}, GK Contact: {metrics.gk_contact}"
        print(summary_str)

if __name__ == "__main__":
    main()
