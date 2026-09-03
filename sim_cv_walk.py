import mujoco
import mujoco.viewer
import time
import cv2
import numpy as np
import onnxruntime as ort

XML_PATH = "microduck_rl/src/mjlab_microduck/robot/microduck/scene.xml"
POLICY_WALK = "microduck/policies/alpha_walking.onnx"
POLICY_PICK = "microduck/policies/alpha_ground_pick.onnx"

DEFAULT_POSE = np.array([
    0.0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0.0, 0.0,
    0.0, 0.0873, 0.4579, 0.0049, -0.4530
], dtype=np.float32)

def quat_rotate_inverse(quat, vec):
    w, x, y, z = quat
    xyz = np.array([x, y, z])
    t = np.cross(xyz, vec) * 2
    return vec - w * t + np.cross(xyz, t)

def main():
    try:
        m = mujoco.MjModel.from_xml_path(XML_PATH)
        d = mujoco.MjData(m)
    except Exception as e:
        print(f"Failed to load MuJoCo model: {e}")
        return

    # Use a smaller camera resolution for fast 10Hz OpenCV processing
    cam_width, cam_height = 320, 240
    renderer = mujoco.Renderer(m, cam_height, cam_width)
    
    session_walk = ort.InferenceSession(POLICY_WALK)
    session_pick = ort.InferenceSession(POLICY_PICK)
    input_name = session_walk.get_inputs()[0].name
    output_name = session_walk.get_outputs()[0].name
    
    imu_ang_vel_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
    trunk_base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    block_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target_block")
    lower_beak_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "lower_beak")
    
    joint_qpos_indices = [int(m.jnt_qposadr[m.actuator_trnid[i, 0]]) for i in range(14)]
    joint_qvel_indices = [int(m.jnt_dofadr[m.actuator_trnid[i, 0]]) for i in range(14)]
    
    last_action = np.zeros(14, dtype=np.float32)
    world_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    cmd_vx = 0.0
    gp_mode = False
    gp_phase = 0.0

    with mujoco.viewer.launch_passive(m, d) as viewer:
        print("Starting OpenCV Visual Servoing + RL Walking Control...")
        
        initial_qpos = d.qpos.copy()
        stand_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "STAND")
        if stand_id != -1:
            mujoco.mj_resetDataKeyframe(m, d, stand_id)
            d.qpos[21:] = initial_qpos[21:]
        mujoco.mj_forward(m, d)
        
        step_counter = 0
        
        while viewer.is_running():
            step_start = time.time()
            
            # ==========================================
            block_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target_block")
            
            dist_xy = 999.0
            if block_body_id != -1:
                trunk_pos = d.xpos[trunk_base_id]
                block_pos = d.xpos[block_body_id]
                dist_xy = np.linalg.norm(trunk_pos[0:2] - block_pos[0:2])

            # ==========================================
            # 1. OpenCV Vision (Runs every 5 steps = 10Hz)
            # ==========================================
            if step_counter % 5 == 0 and not gp_mode:
                renderer.update_scene(d, camera="egocentric")
                img_rgb = renderer.render()
                
                # RGB -> BGR for OpenCV
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                
                b, g, r = cv2.split(img_bgr)
                r = r.astype(int)
                g = g.astype(int)
                b = b.astype(int)
                
                # Robust red detection
                red_mask = (r > g + 40) & (r > b + 40) & (r > 100)
                mask = red_mask.astype(np.uint8) * 255
                
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(c)
                    
                    if area > 3:
                        M = cv2.moments(c)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            error_x = 160 - cx 
                            cmd_vtheta = error_x * 0.005
                            
                            # If it's huge, trigger pick (fallback visual trigger)
                            if area > (cam_width * cam_height * 0.1): 
                                if not gp_mode:
                                    print(f"ARRIVED! Target area: {area}. Triggering ground pick!")
                                    gp_mode = True
                                    gp_phase = 0.0
                                cmd_vx = 0.0
                                cmd_vtheta = 0.0
                            else:
                                cmd_vx = 0.3
                    else:
                        if dist_xy < 0.3:
                            # In blind spot, keep walking forward, but auto-correct heading using true state!
                            vec_world = block_pos - trunk_pos
                            quat = d.xquat[trunk_base_id]
                            vec_body = quat_rotate_inverse(quat, vec_world)
                            angle_err = np.arctan2(vec_body[1], vec_body[0])
                            
                            cmd_vx = 0.3
                            cmd_vtheta = angle_err * 3.0
                        else:
                            print("Red blob too small, scanning...")
                            cmd_vx = 0.0
                            cmd_vtheta = 0.5
                else:
                    if dist_xy < 0.3:
                        # In blind spot, keep walking forward, but auto-correct heading using true state!
                        vec_world = block_pos - trunk_pos
                        quat = d.xquat[trunk_base_id]
                        vec_body = quat_rotate_inverse(quat, vec_world)
                        angle_err = np.arctan2(vec_body[1], vec_body[0])
                        
                        cmd_vx = 0.3
                        cmd_vtheta = angle_err * 3.0
                    else:
                        print("No red block found, spinning to scan...")
                        cmd_vx = 0.0
                        cmd_vtheta = 0.5

            # ==========================================
            # ==========================================
            # Trigger pick purely based on strict physical distance AND alignment
            if dist_xy < 0.17:
                vec_world = block_pos - trunk_pos
                quat = d.xquat[trunk_base_id]
                vec_body = quat_rotate_inverse(quat, vec_world)
                angle_err = np.arctan2(vec_body[1], vec_body[0])
                
                if abs(angle_err) < 0.15: # Less than 8.5 degrees
                    if not gp_mode:
                        print(f"Perfectly aligned! (dist={dist_xy:.2f}, ang={angle_err:.2f}). Triggering ground pick!")
                        gp_mode = True
                        gp_phase = 0.0
                    cmd_vx = 0.0
                    cmd_vtheta = 0.0
                else:
                    # Too close but crooked, turn in place to align!
                    if not gp_mode:
                        # print(f"Close but crooked (ang={angle_err:.2f}). Aligning...")
                        cmd_vx = 0.0
                        cmd_vtheta = angle_err * 4.0

            # 2. RL Policy Inference (Runs every step = 50Hz)
            # ==========================================
            if gp_mode:
                gp_phase += 0.02 / 4.0
                if gp_phase >= 0.7:
                    gp_mode = False
                    print("Ground pick finished. Resuming scan...")
                    gp_phase = 0.0
            
            adr = m.sensor_adr[imu_ang_vel_id]
            ang_vel = d.sensordata[adr:adr+3].copy().astype(np.float32)
            quat = d.xquat[trunk_base_id].copy().astype(np.float32)
            proj_grav = quat_rotate_inverse(quat, world_gravity)
            qpos = d.qpos[joint_qpos_indices].copy().astype(np.float32)
            rel_qpos = qpos - DEFAULT_POSE
            qvel = d.qvel[joint_qvel_indices].copy().astype(np.float32)
            
            command = np.zeros(13, dtype=np.float32)
            if gp_mode:
                command[0] = np.cos(2 * np.pi * gp_phase)
                command[1] = np.sin(2 * np.pi * gp_phase)
                command[2] = 0.0
                active_session = session_pick
            else:
                command[0] = cmd_vx
                command[2] = cmd_vtheta
                active_session = session_walk
            
            obs = np.concatenate([ang_vel, proj_grav, rel_qpos, qvel, last_action, command]).astype(np.float32)
            obs_batch = obs.reshape(1, -1)
            
            action = active_session.run([output_name], {input_name: obs_batch})[0].squeeze(0)
            last_action = action.copy()
            
            action_scaled = action * 0.5
            target_qpos = DEFAULT_POSE + action_scaled
            target_qpos = np.clip(target_qpos, DEFAULT_POSE - 1.5, DEFAULT_POSE + 1.5)
            
            d.ctrl[:14] = target_qpos
            
            if gp_mode:
                if gp_phase < 0.38:
                    d.ctrl[14] = 0.8 # Open mouth
                else:
                    d.ctrl[14] = 0.0 # Snap shut!
                    # Magnetic teleport (Official video trick)
                    if block_body_id != -1 and lower_beak_id != -1:
                        beak_pos = d.xpos[lower_beak_id]
                        block_jnt_id = m.body_jntadr[block_body_id]
                        qpos_adr = m.jnt_qposadr[block_jnt_id]
                        d.qpos[qpos_adr:qpos_adr+3] = beak_pos + np.array([0.015, 0.0, -0.01])
                        d.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0] # Reset rotation
            else:
                d.ctrl[14] = 0.0 # Keep mouth closed
            
            for _ in range(10): 
                mujoco.mj_step(m, d)
            
            viewer.sync()
            step_counter += 1
            
            # Lock to real-time speed
            elapsed = time.time() - step_start
            if 0.02 - elapsed > 0:
                time.sleep(0.02 - elapsed)

if __name__ == "__main__":
    main()
