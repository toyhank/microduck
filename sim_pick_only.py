import mujoco
import mujoco.viewer
import time
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

def main():
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

    gp_phase = 0.0

    with mujoco.viewer.launch_passive(m, d) as viewer:
        print("Starting isolated picking simulation...")
        initial_qpos = d.qpos.copy()
        stand_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "STAND")
        if stand_id != -1:
            mujoco.mj_resetDataKeyframe(m, d, stand_id)
            d.qpos[21:] = initial_qpos[21:]
            
        # Move block right in front of the duck!
        if block_body_id != -1:
            block_jnt_id = m.body_jntadr[block_body_id]
            qpos_adr = m.jnt_qposadr[block_jnt_id]
            d.qpos[qpos_adr:qpos_adr+3] = [0.18, 0, 0.05]
            
        mujoco.mj_forward(m, d)

        while viewer.is_running():
            step_start = time.time()
            
            gp_phase += 0.02 / 4.0
            if gp_phase > 0.7:
                gp_phase = 0.0 # loop the animation
                # Reset block position occasionally if it was knocked away
                if block_body_id != -1:
                    block_jnt_id = m.body_jntadr[block_body_id]
                    qpos_adr = m.jnt_qposadr[block_jnt_id]
                    d.qpos[qpos_adr:qpos_adr+3] = [0.18, 0, 0.05]
                    d.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0]

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
            
            # ===============================================
            # HEAD PITCH OVERRIDE: Point the beak downwards
            # We use a NEGATIVE pitch (looking up) to extend the beak towards the ground
            # ===============================================
            pitch_override = np.sin(gp_phase * np.pi / 0.7) * -0.6
            if pitch_override < 0:
                d.ctrl[6] += pitch_override
            
            # Jaw Control and Magnetic Grasp
            if gp_phase < 0.38:
                d.ctrl[14] = 0.5 
            else:
                d.ctrl[14] = 0.0 
                # Magnetic Grasp! 
                if block_body_id != -1 and lower_beak_id != -1:
                    block_pos = d.xpos[block_body_id]
                    trunk_pos = d.xpos[trunk_base_id]
                    if np.linalg.norm(block_pos[0:2] - trunk_pos[0:2]) < 0.25:
                        beak_pos = d.xpos[lower_beak_id]
                        block_jnt_id = m.body_jntadr[block_body_id]
                        qpos_adr = m.jnt_qposadr[block_jnt_id]
                        d.qpos[qpos_adr:qpos_adr+3] = beak_pos + np.array([0.015, 0.0, -0.01])
                        d.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0]
            
            for _ in range(10): 
                mujoco.mj_step(m, d)
            
            viewer.sync()
            
            time_until_next_step = m.opt.timestep * 10 - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
