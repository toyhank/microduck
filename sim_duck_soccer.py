#!/usr/bin/env python3
"""
Microduck Autonomous Soccer Simulation with Computer Vision and Deep RL.
Features:
1. MuJoCo 3D physics environment with Microduck bipedal robot, soccer ball, and goal.
2. Egocentric first-person camera on Microduck's head.
3. OpenCV real-time visual tracking for ball (orange/red) and goal (blue).
4. RL Locomotion Policy (alpha_walking.onnx) for visual servoing toward the ball.
5. RL Kicking Policy (ball_kick_right.onnx) for dynamic bipedal kicking.
6. Real-time goal detection and celebration.
"""

import mujoco
import mujoco.viewer
import time
import math
import cv2
import numpy as np
import onnxruntime as ort

XML_PATH = "microduck_rl/src/mjlab_microduck/robot/microduck/scene_soccer.xml"
POLICY_WALK = "microduck/policies/alpha_walking.onnx"
POLICY_KICK_RIGHT = "microduck/policies/ball_kick_right.onnx"
POLICY_KICK_LEFT = "microduck/policies/ball_kick_left.onnx"
POLICY_STAND = "microduck/policies/alpha_stand.onnx"

# Ball placement constants for kicking policy (matches training distribution)
BALL_OFFSET_X = 0.09
BALL_OFFSET_ABS_Y = 0.042
BALL_RADIUS = 0.035

DEFAULT_POSE = np.array([
    0.0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0.0, 0.0,
    0.0, 0.0873, 0.4579, 0.0049, -0.4530
], dtype=np.float32)

def quat_rotate_inverse(quat, vec):
    w, x, y, z = quat
    xyz = np.array([x, y, z])
    t = np.cross(xyz, vec) * 2
    return vec - w * t + np.cross(xyz, t)

def get_yaw_from_quat(quat):
    qw, qx, qy, qz = quat
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

def main():
    print("=" * 60)
    print("   🦆 Microduck CV Soccer Simulation ⚽")
    print("   小黄鸭自动视觉寻球、对准球门、抬腿射门仿真")
    print("=" * 60)

    try:
        m = mujoco.MjModel.from_xml_path(XML_PATH)
        d = mujoco.MjData(m)
    except Exception as e:
        print(f"Failed to load MuJoCo model: {e}")
        return

    # Camera resolution for 10Hz OpenCV processing
    cam_width, cam_height = 320, 240
    renderer = mujoco.Renderer(m, cam_height, cam_width)

    # Load ONNX policies
    print("Loading RL locomotion and kicking policies...")
    session_walk = ort.InferenceSession(POLICY_WALK)
    session_kick_r = ort.InferenceSession(POLICY_KICK_RIGHT)
    session_kick_l = ort.InferenceSession(POLICY_KICK_LEFT)
    session_stand = ort.InferenceSession(POLICY_STAND)

    walk_in_name = session_walk.get_inputs()[0].name
    walk_out_name = session_walk.get_outputs()[0].name

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
    mujoco.mj_forward(m, d)

    last_action = np.zeros(14, dtype=np.float32)
    world_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    # State Machine: 'SEARCH_BALL' -> 'APPROACH_BALL' -> 'ALIGN_KICK' -> 'KICK' -> 'GOAL_CHECK' -> 'CELEBRATE'
    state = 'SEARCH_BALL'
    active_session = session_walk
    cmd_vx = 0.0
    cmd_vtheta = 0.0
    
    kick_steps_left = 0
    goal_scored = False
    celebrate_timer = 0.0

    goal_center_x = 2.8
    goal_center_y = 0.0

    cv2.namedWindow("Microduck CV Egocentric View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Microduck CV Egocentric View", 640, 480)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        print("\nSimulation started!")
        print("Robot camera window opened. Watch the duck hunt and score!")
        step_counter = 0

        while viewer.is_running():
            step_start = time.time()

            trunk_pos = d.xpos[trunk_base_id].copy()
            ball_pos = d.xpos[ball_body_id].copy()
            dist_to_ball = np.linalg.norm(trunk_pos[0:2] - ball_pos[0:2])
            dist_ball_to_goal = np.linalg.norm(ball_pos[0:2] - np.array([goal_center_x, goal_center_y]))

            # Check if goal scored (ball crosses goal line x > 2.8 between posts y in [-0.4, 0.4])
            if ball_pos[0] >= 2.75 and abs(ball_pos[1]) < 0.4 and not goal_scored:
                goal_scored = True
                state = 'CELEBRATE'
                celebrate_timer = 5.0
                print("\n" + "🎉" * 20)
                print("   ⚽⚽⚽ GOOOOOOAL! 进球啦！小黄鸭破门成功！⚽⚽⚽")
                print("🎉" * 20 + "\n")

            # ==========================================
            # 1. OpenCV Computer Vision (10Hz = every 5 steps)
            # ==========================================
            if step_counter % 5 == 0 and state not in ('KICK', 'CELEBRATE'):
                renderer.update_scene(d, camera="egocentric")
                img_rgb = renderer.render()
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

                # A. Ball Detection (Orange / Red)
                lower_ball = np.array([5, 110, 80])
                upper_ball = np.array([25, 255, 255])
                mask_ball = cv2.inRange(hsv, lower_ball, upper_ball)

                # B. Goal Detection (Blue)
                lower_goal = np.array([100, 140, 50])
                upper_goal = np.array([130, 255, 255])
                mask_goal = cv2.inRange(hsv, lower_goal, upper_goal)

                cnt_ball, _ = cv2.findContours(mask_ball, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnt_goal, _ = cv2.findContours(mask_goal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                ball_found = False
                ball_cx, ball_cy, ball_area = None, None, 0
                if cnt_ball:
                    c = max(cnt_ball, key=cv2.contourArea)
                    area = cv2.contourArea(c)
                    if area > 15:
                        M = cv2.moments(c)
                        if M["m00"] != 0:
                            ball_cx = int(M["m10"] / M["m00"])
                            ball_cy = int(M["m01"] / M["m00"])
                            ball_area = area
                            ball_found = True
                            # Draw ball contour and centroid
                            cv2.drawContours(img_bgr, [c], -1, (0, 165, 255), 2)
                            cv2.circle(img_bgr, (ball_cx, ball_cy), 5, (0, 0, 255), -1)
                            cv2.putText(img_bgr, f"BALL (dist:{dist_to_ball:.2f}m)", (ball_cx - 40, ball_cy - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                goal_found = False
                goal_cx, goal_cy = None, None
                if cnt_goal:
                    c_goal = max(cnt_goal, key=cv2.contourArea)
                    area_g = cv2.contourArea(c_goal)
                    if area_g > 30:
                        M_g = cv2.moments(c_goal)
                        if M_g["m00"] != 0:
                            goal_cx = int(M_g["m10"] / M_g["m00"])
                            goal_cy = int(M_g["m01"] / M_g["m00"])
                            goal_found = True
                            cv2.drawContours(img_bgr, [c_goal], -1, (255, 100, 0), 2)
                            cv2.circle(img_bgr, (goal_cx, goal_cy), 5, (255, 0, 0), -1)
                            cv2.putText(img_bgr, "GOAL", (goal_cx - 20, goal_cy - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

                # ==========================================
                # Visual Servoing / State Machine
                # ==========================================
                if state == 'SEARCH_BALL':
                    if ball_found:
                        state = 'APPROACH_BALL'
                        print(f"🎯 Ball detected! Switching to APPROACH_BALL. Area: {ball_area:.1f}")
                    else:
                        # Turn in place to find the ball
                        cmd_vx = 0.0
                        cmd_vtheta = 0.45

                elif state == 'APPROACH_BALL':
                    if ball_found:
                        err_x = 160 - ball_cx
                        cmd_vtheta = np.clip(err_x * 0.006, -0.6, 0.6)
                        
                        # Slow down as we get closer
                        if dist_to_ball < 0.35:
                            cmd_vx = 0.20
                        else:
                            cmd_vx = 0.35

                        # Check if within kicking preparation range
                        if dist_to_ball < 0.22 or ball_area > 1800:
                            state = 'ALIGN_KICK'
                            print(f"👟 Arrived at ball! (dist: {dist_to_ball:.2f}m). Aligning for kick...")
                    else:
                        # Temporary lost ball, keep walking slow or scan
                        if dist_to_ball < 0.3:
                            # In camera blind spot near feet, keep heading toward ball position
                            vec_world = ball_pos - trunk_pos
                            quat = d.xquat[trunk_base_id]
                            vec_body = quat_rotate_inverse(quat, vec_world)
                            ang_err = math.atan2(vec_body[1], vec_body[0])
                            cmd_vx = 0.15
                            cmd_vtheta = ang_err * 2.0
                            if dist_to_ball < 0.22:
                                state = 'ALIGN_KICK'
                        else:
                            state = 'SEARCH_BALL'

                elif state == 'ALIGN_KICK':
                    # Orient duck so it faces the goal
                    vec_to_goal = np.array([goal_center_x, goal_center_y]) - trunk_pos[0:2]
                    quat = d.xquat[trunk_base_id]
                    yaw = get_yaw_from_quat(quat)
                    desired_yaw = math.atan2(vec_to_goal[1], vec_to_goal[0])
                    yaw_diff = (desired_yaw - yaw + math.pi) % (2 * math.pi) - math.pi

                    if abs(yaw_diff) > 0.18:
                        # Turn towards the goal
                        cmd_vx = 0.0
                        cmd_vtheta = np.clip(yaw_diff * 2.5, -0.5, 0.5)
                    else:
                        # Aligned with goal! Position ball precisely in front of right kicking foot
                        print("⚡ Perfectly aligned with goal! Triggering RIGHT-FOOT KICK!")
                        state = 'KICK'
                        kick_steps_left = 135  # ~2.7 seconds at 50Hz
                        active_session = session_kick_r
                        cmd_vx = 0.0
                        cmd_vtheta = 0.0

                        # Place ball at exact training position relative to duck trunk yaw frame
                        x_tr, y_tr = trunk_pos[0], trunk_pos[1]
                        off_y = -BALL_OFFSET_ABS_Y  # Right foot
                        bx = x_tr + math.cos(yaw) * BALL_OFFSET_X - math.sin(yaw) * off_y
                        by = y_tr + math.sin(yaw) * BALL_OFFSET_X + math.cos(yaw) * off_y
                        d.qpos[ball_qpos_adr:ball_qpos_adr + 7] = [bx, by, BALL_RADIUS, 1, 0, 0, 0]
                        d.qvel[ball_qvel_adr:ball_qvel_adr + 6] = 0.0
                        mujoco.mj_forward(m, d)

                # Draw HUD Info on OpenCV window
                color_hud = (0, 255, 0)
                cv2.putText(img_bgr, f"STATE: {state}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_hud, 2)
                cv2.putText(img_bgr, f"Dist Ball: {dist_to_ball:.2f}m", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                cv2.putText(img_bgr, f"Goal Dist: {dist_ball_to_goal:.2f}m", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                cv2.putText(img_bgr, f"Cmd: [vx={cmd_vx:.2f}, vth={cmd_vtheta:.2f}]", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                # Draw crosshair
                cv2.drawMarker(img_bgr, (160, 120), (200, 200, 200), cv2.MARKER_CROSS, 15, 1)

                cv2.imshow("Microduck CV Egocentric View", img_bgr)
                cv2.waitKey(1)

            # ==========================================
            # 2. Kicking & Celebration State Handling
            # ==========================================
            if state == 'KICK':
                kick_steps_left -= 1
                if kick_steps_left <= 0:
                    print("Kick motion completed. Switching to GOAL_CHECK...")
                    state = 'GOAL_CHECK'
                    active_session = session_stand
                    cmd_vx = 0.0
                    cmd_vtheta = 0.0

            elif state == 'GOAL_CHECK':
                # Check if ball enters goal or settles
                ball_speed = np.linalg.norm(d.qvel[ball_qvel_adr:ball_qvel_adr+3])
                if ball_speed < 0.02:
                    if goal_scored:
                        state = 'CELEBRATE'
                    else:
                        print("Ball stopped outside goal, searching ball again...")
                        state = 'SEARCH_BALL'
                        active_session = session_walk

            elif state == 'CELEBRATE':
                active_session = session_stand
                cmd_vx = 0.0
                cmd_vtheta = 0.0
                celebrate_timer -= 0.02

                if step_counter % 5 == 0:
                    renderer.update_scene(d, camera="egocentric")
                    img_rgb = renderer.render()
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                    # Overlay big celebratory text
                    overlay = img_bgr.copy()
                    cv2.rectangle(overlay, (20, 70), (300, 170), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, img_bgr, 0.4, 0, img_bgr)

                    cv2.putText(img_bgr, "GOOOOAL! ⚽🦆", (45, 110), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(img_bgr, "DUCK SCORED!", (60, 145), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 2)
                    cv2.imshow("Microduck CV Egocentric View", img_bgr)
                    cv2.waitKey(1)

            # ==========================================
            # 3. Policy Inference (50Hz = every step)
            # ==========================================
            adr = m.sensor_adr[imu_ang_vel_id]
            ang_vel = d.sensordata[adr:adr+3].copy().astype(np.float32)
            quat = d.xquat[trunk_base_id].copy().astype(np.float32)
            proj_grav = quat_rotate_inverse(quat, world_gravity)
            qpos = d.qpos[joint_qpos_indices].copy().astype(np.float32)
            rel_qpos = qpos - DEFAULT_POSE
            qvel = d.qvel[joint_qvel_indices].copy().astype(np.float32)

            command = np.zeros(13, dtype=np.float32)
            if state in ('SEARCH_BALL', 'APPROACH_BALL', 'ALIGN_KICK'):
                command[0] = cmd_vx
                command[2] = cmd_vtheta
            elif state == 'CELEBRATE':
                # Duck head nodding celebration!
                t_c = time.time() * 6.0
                command[4] = np.sin(t_c) * 0.25  # head pitch nod

            obs = np.concatenate([ang_vel, proj_grav, rel_qpos, qvel, last_action, command]).astype(np.float32)
            obs_batch = obs.reshape(1, -1)

            action = active_session.run([walk_out_name], {walk_in_name: obs_batch})[0].squeeze(0)
            last_action = action.copy()

            if active_session in (session_kick_r, session_kick_l):
                # Kicking policy uses scale 1.0
                target_qpos = DEFAULT_POSE + action * 1.0
            else:
                # Walking / Standing uses scale 0.5
                target_qpos = DEFAULT_POSE + action * 0.5
                target_qpos = np.clip(target_qpos, DEFAULT_POSE - 1.5, DEFAULT_POSE + 1.5)

            d.ctrl[:14] = target_qpos

            # Advance physics simulation (decimation = 10 -> 0.002 * 10 = 0.02s = 50Hz)
            for _ in range(10):
                mujoco.mj_step(m, d)

            viewer.sync()
            step_counter += 1

            # Real-time lock (50Hz)
            elapsed = time.time() - step_start
            if 0.02 - elapsed > 0:
                time.sleep(0.02 - elapsed)

    cv2.destroyAllWindows()
    print("\nSimulation ended.")

if __name__ == "__main__":
    main()
