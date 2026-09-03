import mujoco
import mujoco.viewer
import time
import numpy as np
import math
import onnxruntime as ort

XML_PATH = "microduck_rl/src/mjlab_microduck/robot/microduck/scene_walk.xml"
POLICY_PATH = "microduck/policies/alpha_walking.onnx"

# Matches HOME_FRAME
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
    
    session = ort.InferenceSession(POLICY_PATH)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    imu_ang_vel_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
    trunk_base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    
    joint_qpos_indices = [int(m.jnt_qposadr[m.actuator_trnid[i, 0]]) for i in range(m.nu)]
    joint_qvel_indices = [int(m.jnt_dofadr[m.actuator_trnid[i, 0]]) for i in range(m.nu)]
    
    last_action = np.zeros(m.nu, dtype=np.float32)
    world_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        print("Starting walking simulation...")
        print("The robot will automatically walk forward and turn slightly.")
        
        step = 0
        while viewer.is_running():
            step_start = time.time()
            
            # --- Generate Observations ---
            # 1. Base ang vel
            adr = m.sensor_adr[imu_ang_vel_id]
            ang_vel = d.sensordata[adr:adr+3].copy().astype(np.float32)
            
            # 2. Projected gravity
            quat = d.xquat[trunk_base_id].copy().astype(np.float32)
            proj_grav = quat_rotate_inverse(quat, world_gravity)
            
            # 3. Joint pos (relative to default)
            qpos = d.qpos[joint_qpos_indices].copy().astype(np.float32)
            rel_qpos = qpos - DEFAULT_POSE
            
            # 4. Joint vel
            qvel = d.qvel[joint_qvel_indices].copy().astype(np.float32)
            
            # 5. Command (13D)
            # Make it walk in a figure 8
            cmd_vx = 0.25 # 0.25 m/s forward
            cmd_vtheta = 0.5 * math.sin(step * 0.01) # turning speed
            
            command = np.zeros(13, dtype=np.float32)
            command[0] = cmd_vx
            command[2] = cmd_vtheta
            
            # Concatenate
            obs = np.concatenate([ang_vel, proj_grav, rel_qpos, qvel, last_action, command]).astype(np.float32)
            
            # --- Inference ---
            obs_batch = obs.reshape(1, -1)
            action = session.run([output_name], {input_name: obs_batch})[0]
            action = action.squeeze(0).astype(np.float32)
            last_action = action.copy()
            
            # --- Apply Action ---
            target_positions = DEFAULT_POSE + action * 1.0 # action_scale
            d.ctrl[:] = target_positions
            
            # Step physics (decimation = 10, control_dt = 0.02)
            for _ in range(10):
                mujoco.mj_step(m, d)
                
            viewer.sync()
            
            step += 1
            
            elapsed = time.time() - step_start
            if 0.02 - elapsed > 0:
                time.sleep(0.02 - elapsed)

if __name__ == "__main__":
    main()
