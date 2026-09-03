import mujoco
import numpy as np
import onnxruntime as ort

XML_PATH = "microduck_rl/src/mjlab_microduck/robot/microduck/scene.xml"
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

def simulate(block_x, neck_amp, head_amp, jaw_close_phase):
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)
    session_pick = ort.InferenceSession(POLICY_PICK)
    input_name = session_pick.get_inputs()[0].name
    output_name = session_pick.get_outputs()[0].name
    
    imu_ang_vel_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
    trunk_base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    block_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target_block")
    
    joint_qpos_indices = [int(m.jnt_qposadr[m.actuator_trnid[i, 0]]) for i in range(14)]
    joint_qvel_indices = [int(m.jnt_dofadr[m.actuator_trnid[i, 0]]) for i in range(14)]
    last_action = np.zeros(14, dtype=np.float32)
    world_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    initial_qpos = d.qpos.copy()
    stand_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    if stand_id != -1:
        mujoco.mj_resetDataKeyframe(m, d, stand_id)
        d.qpos[21:] = initial_qpos[21:]
        
    if block_body_id != -1:
        m.body_pos[block_body_id][0] = block_x
        m.body_pos[block_body_id][1] = 0.0
        
        block_jnt_id = m.body_jntadr[block_body_id]
        qpos_adr = m.jnt_qposadr[block_jnt_id]
        d.qpos[qpos_adr] = 0.05
        
    mujoco.mj_forward(m, d)

    gp_phase = 0.0
    min_beak_z = 999.0
    beak_x_at_min_z = 0.0
    lower_beak_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "lower_beak")

    for _ in range(200):
        if gp_phase <= 0.7:
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
        command[2] = 0.0
        
        obs = np.concatenate([ang_vel, proj_grav, rel_qpos, qvel, last_action, command]).astype(np.float32)
        obs_batch = obs.reshape(1, -1)
        
        action = session_pick.run([output_name], {input_name: obs_batch})[0].squeeze(0)
        last_action = action.copy()
        
        action_scaled = action * 0.5
        target_qpos = DEFAULT_POSE + action_scaled
        target_qpos = np.clip(target_qpos, DEFAULT_POSE - 1.5, DEFAULT_POSE + 1.5)
        d.ctrl[:14] = target_qpos
        
        # Override neck and head
        neck_override = np.sin(gp_phase * np.pi / 0.7) * neck_amp
        head_override = np.sin(gp_phase * np.pi / 0.7) * head_amp
        
        # We assume neck needs to bend down (positive) and head needs to look up (negative)
        if neck_override > 0:
            d.ctrl[5] += neck_override
        if head_override < 0:
            d.ctrl[6] += head_override
            
        if gp_phase < jaw_close_phase:
            d.ctrl[14] = 0.5 
        else:
            d.ctrl[14] = 0.0 
        
        for _ in range(10): 
            mujoco.mj_step(m, d)
            
        if lower_beak_id != -1:
            bz = d.xpos[lower_beak_id][2]
            if bz < min_beak_z:
                min_beak_z = bz
                beak_x_at_min_z = d.xpos[lower_beak_id][0]
            
    trunk_z = d.xpos[trunk_base_id][2]
    if trunk_z < 0.05:
        return False, 0.0, min_beak_z, beak_x_at_min_z, 0.0, 0.0
        
    block_z = d.xpos[block_body_id][2]
    block_x_end = d.xpos[block_body_id][0]
    block_y_end = d.xpos[block_body_id][1]
    return block_z > 0.07, block_z, min_beak_z, beak_x_at_min_z, block_x_end, trunk_z

def search():
    print("Starting Grid Search...")
    for block_x in np.arange(0.14, 0.22, 0.01):
        # We fix head_amp=-0.6 (look up slightly to point beak down) and neck_amp=0.0
        success, final_z, min_bz, bx, fx, tz = simulate(block_x, 0.0, -0.6, 0.38)
        status = "SUCCESS!" if success else "FAILED"
        print(f"block_x={block_x:.3f} -> {status}, block_z={final_z:.3f}, beak_min_z={min_bz:.3f}, beak_x={bx:.3f}, final_bx={fx:.3f}, trunk_z={tz:.3f}")

if __name__ == "__main__":
    search()
