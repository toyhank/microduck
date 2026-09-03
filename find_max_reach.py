import mujoco
import numpy as np

def find_best_pose():
    m = mujoco.MjModel.from_xml_path("microduck_rl/src/mjlab_microduck/robot/microduck/robot_allcollisions.xml")
    d = mujoco.MjData(m)
    
    # Reset to stand
    stand_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    mujoco.mj_resetDataKeyframe(m, d, stand_id)
    initial_qpos = d.qpos.copy()
    
    # Get IDs
    trunk_base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    lower_beak_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "lower_beak")
    
    head_joint_idx = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "head_pitch")
    neck_joint_idx = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "neck_pitch")
    
    head_qpos_adr = m.jnt_qposadr[head_joint_idx]
    neck_qpos_adr = m.jnt_qposadr[neck_joint_idx]
    
    max_diff = -999.0
    best_head = 0
    best_neck = 0
    best_beak_x = 0
    best_beak_z = 0
    
    print("Sweeping head and neck pitches...")
    for head_p in np.arange(-1.5, 1.5, 0.1):
        for neck_p in np.arange(-1.5, 1.5, 0.1):
            d.qpos[:] = initial_qpos[:]
            d.qpos[head_qpos_adr] = head_p
            d.qpos[neck_qpos_adr] = neck_p
            
            mujoco.mj_kinematics(m, d)
            
            beak_x = d.xpos[lower_beak_id][0]
            beak_z = d.xpos[lower_beak_id][2]
            trunk_x = d.xpos[trunk_base_id][0]
            
            # The trunk is a box. It extends forward from trunk_x by some amount.
            # Assuming trunk length is about 0.15m (so front is trunk_x + 0.075)
            trunk_front_x = trunk_x + 0.075
            
            diff = beak_x - trunk_front_x
            if diff > max_diff:
                max_diff = diff
                best_head = head_p
                best_neck = neck_p
                best_beak_x = beak_x
                best_beak_z = beak_z

    print(f"Best pose: head={best_head:.2f}, neck={best_neck:.2f}")
    print(f"Beak X: {best_beak_x:.3f}, Trunk Front X (approx): {d.xpos[trunk_base_id][0] + 0.075:.3f}")
    print(f"Difference: {max_diff:.3f} m (Positive means beak is IN FRONT of chest)")
    print(f"Beak Z at this pose: {best_beak_z:.3f} m")

if __name__ == "__main__":
    find_best_pose()
