import mujoco
import numpy as np
import onnxruntime as ort

def quat_rotate_inverse(q, v):
    q_w, q_x, q_y, q_z = q
    q_vec = np.array([q_x, q_y, q_z])
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c

def main():
    m = mujoco.MjModel.from_xml_path("microduck_rl/src/mjlab_microduck/robot/microduck/scene.xml")
    d = mujoco.MjData(m)
    
    session_pick = ort.InferenceSession("microduck/policies/alpha_ground_pick.onnx")
    input_name = session_pick.get_inputs()[0].name
    output_name = session_pick.get_outputs()[0].name
    
    DEFAULT_POSE = np.array([
        0.0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0.0, 0.0,
        0.0, 0.0873, 0.4579, 0.0049, -0.4530
    ], dtype=np.float32)
    
    stand_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    if stand_id != -1:
        mujoco.mj_resetDataKeyframe(m, d, stand_id)
        mujoco.mj_forward(m, d)
        
    joint_qpos_indices = [int(m.jnt_qposadr[m.actuator_trnid[i, 0]]) for i in range(14)]
    standing_ctrl = d.qpos[joint_qpos_indices]
    print("Symmetric standing ctrl:")
    for i in range(14):
        print(f"ctrl[{i}] = {standing_ctrl[i]:.4f}")
    return
        
    trunk_base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    lower_beak_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "lower_beak")
    imu_ang_vel_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
    
    joint_qpos_indices = [int(m.jnt_qposadr[m.actuator_trnid[i, 0]]) for i in range(14)]
    joint_qvel_indices = [int(m.jnt_dofadr[m.actuator_trnid[i, 0]]) for i in range(14)]
        
    world_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    last_action = np.zeros(14, dtype=np.float32)
    
    min_beak_z = 999.0
    min_beak_x = 0.0
    trunk_z_at_min = 0.0
    
    session_walk = ort.InferenceSession("microduck/policies/alpha_walking.onnx")
    
    # Walk for 50 steps
    for _ in range(50):
        adr = m.sensor_adr[imu_ang_vel_id]
        ang_vel = d.sensordata[adr:adr+3].copy().astype(np.float32)
        quat = d.xquat[trunk_base_id].copy().astype(np.float32)
        proj_grav = quat_rotate_inverse(quat, world_gravity)
        qpos = d.qpos[joint_qpos_indices].copy().astype(np.float32)
        rel_qpos = qpos - DEFAULT_POSE
        qvel = d.qvel[joint_qvel_indices].copy().astype(np.float32)
        
        command = np.zeros(13, dtype=np.float32)
        command[0] = 0.0
        command[2] = 0.0
        
        obs = np.concatenate([ang_vel, proj_grav, rel_qpos, qvel, last_action, command]).astype(np.float32)
        obs_batch = obs.reshape(1, -1)
        
        action = session_walk.run([session_walk.get_outputs()[0].name], {session_walk.get_inputs()[0].name: obs_batch})[0].squeeze(0)
        last_action = action.copy()
        
        target_qpos = DEFAULT_POSE + action * 0.5
        d.ctrl[:14] = np.clip(target_qpos, DEFAULT_POSE - 1.5, DEFAULT_POSE + 1.5)
        
        for _ in range(10): mujoco.mj_step(m, d)
        
    print("Sweeping overrides...")
    best_min_z = 999.0
    best_h = 0
    best_n = 0
    
    for hp in [0.0, 0.5, 1.0, 1.5]:
        for np_override in [-0.5, 0.0, 0.5, 1.0, 1.5]:
            
            # Reset
            mujoco.mj_resetDataKeyframe(m, d, stand_id)
            
            # Fast walk to stabilize
            for _ in range(20):
                adr = m.sensor_adr[imu_ang_vel_id]
                ang_vel = d.sensordata[adr:adr+3].copy().astype(np.float32)
                quat = d.xquat[trunk_base_id].copy().astype(np.float32)
                proj_grav = quat_rotate_inverse(quat, world_gravity)
                qpos = d.qpos[joint_qpos_indices].copy().astype(np.float32)
                rel_qpos = qpos - DEFAULT_POSE
                qvel = d.qvel[joint_qvel_indices].copy().astype(np.float32)
                
                command = np.zeros(13, dtype=np.float32)
                
                obs = np.concatenate([ang_vel, proj_grav, rel_qpos, qvel, last_action, command]).astype(np.float32)
                obs_batch = obs.reshape(1, -1)
                
                action = session_walk.run([session_walk.get_outputs()[0].name], {session_walk.get_inputs()[0].name: obs_batch})[0].squeeze(0)
                last_action = action.copy()
                
                target_qpos = DEFAULT_POSE + action * 0.5
                d.ctrl[:14] = np.clip(target_qpos, DEFAULT_POSE - 1.5, DEFAULT_POSE + 1.5)
                
                for _ in range(10): mujoco.mj_step(m, d)
                
            gp_phase = 0.0
            min_beak_z = 999.0
            while gp_phase <= 0.38:
                gp_phase += 0.02 / 4.0
                
                adr = m.sensor_adr[imu_ang_vel_id]
                ang_vel = d.sensordata[adr:adr+3].copy().astype(np.float32)
                quat = d.xquat[trunk_base_id].copy().astype(np.float32)
                proj_grav = quat_rotate_inverse(quat, world_gravity)
                
                qpos = d.qpos[joint_qpos_indices].copy().astype(np.float32)
                rel_qpos = qpos - DEFAULT_POSE
                qvel = d.qvel[joint_qvel_indices].copy().astype(np.float32)
                
                command = np.zeros(13, dtype=np.float32)
                command[0] = np.cos(2 * np.pi * gp_phase)
                command[1] = np.sin(2 * np.pi * gp_phase)
                
                obs = np.concatenate([ang_vel, proj_grav, rel_qpos, qvel, last_action, command]).astype(np.float32)
                obs_batch = obs.reshape(1, -1)
                
                action = session_pick.run([output_name], {input_name: obs_batch})[0].squeeze(0)
                last_action = action.copy()
                
                action_scaled = action * 0.5
                target_qpos = DEFAULT_POSE + action_scaled
                target_qpos = np.clip(target_qpos, DEFAULT_POSE - 1.5, DEFAULT_POSE + 1.5)
                
                d.ctrl[:14] = target_qpos
                
                # Apply override
                progress = gp_phase / 0.38
                d.ctrl[6] = progress * hp
                d.ctrl[5] = progress * np_override
                
                for _ in range(10):
                    mujoco.mj_step(m, d)
                    
                beak_z = d.xpos[lower_beak_id][2]
                if beak_z < min_beak_z:
                    min_beak_z = beak_z
            
            if min_beak_z < best_min_z:
                best_min_z = min_beak_z
                best_h = hp
                best_n = np_override
                print(f"New best! head={hp}, neck={np_override} -> min_z={min_beak_z:.4f}")

    print(f"Final best: head={best_h}, neck={best_n}, min_z={best_min_z:.4f}")

if __name__ == "__main__":
    main()
