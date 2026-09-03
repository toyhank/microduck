import mujoco
import numpy as np
import onnxruntime as ort
import cv2

def quat_rotate_inverse(quat, vec):
    w, x, y, z = quat
    xyz = np.array([x, y, z])
    t = np.cross(xyz, vec) * 2
    return vec - w * t + np.cross(xyz, t)

def main():
    m = mujoco.MjModel.from_xml_path("microduck_rl/src/mjlab_microduck/robot/microduck/scene.xml")
    d = mujoco.MjData(m)
    
    renderer = mujoco.Renderer(m, 240, 320)
    
    session_walk = ort.InferenceSession("microduck/policies/alpha_walking.onnx")
    session_pick = ort.InferenceSession("microduck/policies/alpha_ground_pick.onnx")
    input_name = session_walk.get_inputs()[0].name
    output_name = session_walk.get_outputs()[0].name
    
    DEFAULT_POSE = np.array([
        0.0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0.0, 0.0,
        0.0, 0.0873, 0.4579, 0.0049, -0.4530
    ], dtype=np.float32)
    
    imu_ang_vel_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
    trunk_base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    block_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target_block")
    lower_beak_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "lower_beak")
    
    joint_qpos_indices = [int(m.jnt_qposadr[m.actuator_trnid[i, 0]]) for i in range(14)]
    joint_qvel_indices = [int(m.jnt_dofadr[m.actuator_trnid[i, 0]]) for i in range(14)]
    
    world_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    last_action = np.zeros(14, dtype=np.float32)
    
    stand_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    mujoco.mj_resetDataKeyframe(m, d, stand_id)
    mujoco.mj_forward(m, d)
    
    # Place towel at 1.0m
    block_jnt_id = m.body_jntadr[block_body_id]
    qpos_adr = m.jnt_qposadr[block_jnt_id]
    d.qpos[qpos_adr:qpos_adr+3] = [1.0, 0.0, 0.005]
    
    cmd_vx = 0.0
    cmd_vtheta = 0.0
    gp_mode = False
    gp_phase = 0.0
    
    print("Starting headless simulation...")
    
    for step in range(500):
        trunk_pos = d.xpos[trunk_base_id]
        block_pos = d.xpos[block_body_id]
        dist_xy = np.linalg.norm(trunk_pos[0:2] - block_pos[0:2])
        
        if step % 5 == 0 and not gp_mode:
            renderer.update_scene(d, camera="egocentric")
            img_rgb = renderer.render()
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            b, g, r = cv2.split(img_bgr)
            red_mask = (r.astype(int) > g.astype(int) + 40) & (r.astype(int) > b.astype(int) + 40) & (r.astype(int) > 100)
            mask = red_mask.astype(np.uint8) * 255
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 50:
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        err = cx - 160
                        cmd_vx = 0.3
                        cmd_vtheta = -err * 0.005
            else:
                cmd_vx = 0.0
                cmd_vtheta = 0.5
                
        if dist_xy < 0.17 and not gp_mode:
            vec_world = block_pos - trunk_pos
            quat = d.xquat[trunk_base_id]
            vec_body = quat_rotate_inverse(quat, vec_world)
            angle_err = np.arctan2(vec_body[1], vec_body[0])
            if abs(angle_err) < 0.15:
                print(f"Step {step}: Perfectly aligned! Triggering pick!")
                gp_mode = True
                gp_phase = 0.0
                cmd_vx = 0.0
                cmd_vtheta = 0.0
                
        if gp_mode:
            gp_phase += 0.02 / 4.0
            if gp_phase >= 0.7:
                print(f"Step {step}: Ground pick finished!")
                break
                
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
            active_session = session_pick
        else:
            command[0] = cmd_vx
            command[2] = cmd_vtheta
            active_session = session_walk
            
        obs = np.concatenate([ang_vel, proj_grav, rel_qpos, qvel, last_action, command]).astype(np.float32)
        action = active_session.run([output_name], {input_name: obs.reshape(1, -1)})[0].squeeze(0)
        last_action = action.copy()
        
        target_qpos = np.clip(DEFAULT_POSE + action * 0.5, DEFAULT_POSE - 1.5, DEFAULT_POSE + 1.5)
        d.ctrl[:14] = target_qpos
        
        if gp_mode:
            if gp_phase < 0.38:
                d.ctrl[14] = 0.8
            else:
                d.ctrl[14] = 0.0
                beak_pos = d.xpos[lower_beak_id]
                d.qpos[qpos_adr:qpos_adr+3] = beak_pos + np.array([0.015, 0.0, -0.01])
        else:
            d.ctrl[14] = 0.0
            
        for _ in range(10): mujoco.mj_step(m, d)
        
    print(f"Final towel Z position: {d.xpos[block_body_id][2]:.4f} (should be > 0.05 if picked successfully)")

if __name__ == "__main__":
    main()
