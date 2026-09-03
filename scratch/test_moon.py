import mujoco
import mujoco.viewer
import time
import numpy as np
import onnxruntime as ort

def quat_rotate_inverse(quat, vec):
    w, x, y, z = quat
    xyz = np.array([x, y, z])
    t = np.cross(xyz, vec) * 2
    return vec - w * t + np.cross(xyz, t)

def main():
    XML_PATH = "microduck_rl/src/mjlab_microduck/robot/microduck/scene_walk.xml"
    POLICY_PATH = "microduck/policies/moon_walking.onnx"
    
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    
    # --- SET MOON GRAVITY ---
    m.opt.gravity = [0, 0, -1.62]
    
    d = mujoco.MjData(m)
    
    session = ort.InferenceSession(POLICY_PATH)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    DEFAULT_POSE = np.array([
        0.0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0.0, 0.0,
        0.0, 0.0873, 0.4579, 0.0049, -0.4530
    ], dtype=np.float32)
    
    imu_ang_vel_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
    trunk_base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    
    joint_qpos_indices = [int(m.jnt_qposadr[m.actuator_trnid[i, 0]]) for i in range(14)]
    joint_qvel_indices = [int(m.jnt_dofadr[m.actuator_trnid[i, 0]]) for i in range(14)]
    
    last_action = np.zeros(14, dtype=np.float32)
    world_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    # Initialize robot to STAND keyframe (essential for balance!)
    stand_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    if stand_id != -1:
        mujoco.mj_resetDataKeyframe(m, d, stand_id)
    mujoco.mj_forward(m, d)

    # Match training physics parameters: timestep=0.005s, 4 substeps (50Hz control)
    m.opt.timestep = 0.005
    substeps = 4
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        print("Starting walking simulation with RETRAINED LUNAR POLICY (-1.62 m/s^2)...")
        print("Commanded speed is 0.35 m/s straight ahead.")
        print("Policy: microduck/policies/moon_walking.onnx")
        
        step = 0
        while viewer.is_running():
            step_start = time.time()
            
            adr = m.sensor_adr[imu_ang_vel_id]
            ang_vel = d.sensordata[adr:adr+3].copy().astype(np.float32)
            
            quat = d.xquat[trunk_base_id].copy().astype(np.float32)
            proj_grav = quat_rotate_inverse(quat, world_gravity)
            
            qpos = d.qpos[joint_qpos_indices].copy().astype(np.float32)
            rel_qpos = qpos - DEFAULT_POSE
            
            qvel = d.qvel[joint_qvel_indices].copy().astype(np.float32)
            
            command = np.zeros(13, dtype=np.float32)
            command[0] = 0.35 # 0.35 m/s forward
            command[1] = 0.0  # lateral
            command[2] = 0.0  # straight, no rotation
            
            obs = np.concatenate([ang_vel, proj_grav, rel_qpos, qvel, last_action, command]).astype(np.float32)
            
            action = session.run([output_name], {input_name: obs.reshape(1, -1)})[0]
            action = action.squeeze(0).astype(np.float32)
            last_action = action.copy()
            
            target_positions = np.clip(DEFAULT_POSE + action * 1.0, DEFAULT_POSE - 1.5, DEFAULT_POSE + 1.5)
            d.ctrl[:14] = target_positions
            
            # Step physics (4 substeps of 0.005s = 0.02s per RL control cycle)
            for _ in range(substeps):
                mujoco.mj_step(m, d)
                
            viewer.sync()
            step += 1
            
            elapsed = time.time() - step_start
            if 0.02 - elapsed > 0:
                time.sleep(0.02 - elapsed)

if __name__ == "__main__":
    main()
