import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
import time

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
    
    session_walk = ort.InferenceSession("microduck_rl/src/mjlab_microduck/policy/microduck_walk.onnx")
    input_name = session_walk.get_inputs()[0].name
    output_name = session_walk.get_outputs()[0].name
    
    DEFAULT_POSE = np.array([0, 0.4, -0.8,  0, 0.4, -0.8,  0, 0,  0, 0.4, -0.8,  0, 0.4, -0.8], dtype=np.float32)
    
    # Reset
    stand_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    if stand_id != -1:
        mujoco.mj_resetDataKeyframe(m, d, stand_id)
        
    trunk_base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    imu_ang_vel_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
    
    block_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target_block")
    pedestal_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pedestal")
    
    joint_qpos_indices = []
    joint_qvel_indices = []
    joint_names = [
        "left_hip_roll", "left_hip_pitch", "left_knee",
        "right_hip_roll", "right_hip_pitch", "right_knee",
        "head_pitch", "neck_pitch",
        "left_ankle", "right_ankle",
        "left_hip_yaw", "right_hip_yaw"
    ]
    
    for name in joint_names:
        jnt_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        joint_qpos_indices.append(m.jnt_qposadr[jnt_id])
        joint_qvel_indices.append(m.jnt_dofadr[jnt_id])
        
    world_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    last_action = np.zeros(14, dtype=np.float32)
    
    # Place pedestal and block in front
    if pedestal_id != -1:
        m.body_pos[pedestal_id][0] = 0.15
        m.body_pos[pedestal_id][1] = 0.0
    if block_body_id != -1:
        block_jnt_id = m.body_jntadr[block_body_id]
        qpos_adr = m.jnt_qposadr[block_jnt_id]
        d.qpos[qpos_adr:qpos_adr+3] = [0.15, 0.0, 0.16]
        
    mujoco.mj_forward(m, d)
    
    gp_phase = 0.0
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            gp_phase += 0.02 / 4.0
            if gp_phase > 0.7:
                gp_phase = 0.0
                if block_body_id != -1:
                    block_jnt_id = m.body_jntadr[block_body_id]
                    qpos_adr = m.jnt_qposadr[block_jnt_id]
                    d.qpos[qpos_adr:qpos_adr+3] = [0.15, 0.0, 0.16]
                    d.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0]

            adr = m.sensor_adr[imu_ang_vel_id]
            ang_vel = d.sensordata[adr:adr+3].copy().astype(np.float32)
            quat = d.xquat[trunk_base_id].copy().astype(np.float32)
            proj_grav = quat_rotate_inverse(quat, world_gravity)
            
            qpos = d.qpos[joint_qpos_indices].copy().astype(np.float32)
            rel_qpos = qpos - DEFAULT_POSE
            qvel = d.qvel[joint_qvel_indices].copy().astype(np.float32)
            
            command = np.zeros(13, dtype=np.float32)
            command[0] = 0.0 # vx = 0
            command[2] = 0.0 # vtheta = 0
            
            obs = np.concatenate([ang_vel, proj_grav, rel_qpos, qvel, last_action, command]).astype(np.float32)
            obs_batch = obs.reshape(1, -1)
            
            action = session_walk.run([output_name], {input_name: obs_batch})[0].squeeze(0)
            last_action = action.copy()
            
            action_scaled = action * 0.5
            target_qpos = DEFAULT_POSE + action_scaled
            target_qpos = np.clip(target_qpos, DEFAULT_POSE - 1.5, DEFAULT_POSE + 1.5)
            d.ctrl[:14] = target_qpos
            
            # STANDING PICK OVERRIDES
            # We want head_pitch=0.6, neck_pitch=-1.5 when gp_phase=0.38
            
            if gp_phase < 0.38:
                progress = gp_phase / 0.38
            else:
                progress = 1.0 - (gp_phase - 0.38) / (0.7 - 0.38)
            
            d.ctrl[6] = progress * 0.6  # head_pitch
            d.ctrl[7] = progress * -1.5 # neck_pitch
            
            # Jaw Control
            if gp_phase < 0.38:
                d.ctrl[14] = 0.8
            else:
                d.ctrl[14] = 0.0
            
            for _ in range(10): 
                mujoco.mj_step(m, d)
            
            viewer.sync()
            
            elapsed = time.time() - step_start
            if 0.02 - elapsed > 0:
                time.sleep(0.02 - elapsed)

if __name__ == "__main__":
    main()
