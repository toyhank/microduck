import mujoco
import time
import numpy as np

def main():
    m = mujoco.MjModel.from_xml_path("microduck_rl/src/mjlab_microduck/robot/microduck/scene.xml")
    d = mujoco.MjData(m)
    
    stand_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    if stand_id != -1:
        mujoco.mj_resetDataKeyframe(m, d, stand_id)
        
    lower_beak_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "lower_beak")
    trunk_base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    
    DEFAULT_POSE = np.array([
        0.0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0.0, 0.0,
        0.0, 0.0873, 0.4579, 0.0049, -0.4530
    ], dtype=np.float32)
    
    d.ctrl[:14] = DEFAULT_POSE
    
    print("Simulating kneel down...")
    
    min_beak_z = 999.0
    
    # 0.5 second animation
    steps = 50
    for step in range(steps):
        progress = step / steps
        
        # Knee bend (extreme)
        d.ctrl[3] = DEFAULT_POSE[3] - progress * 2.5 # left knee
        d.ctrl[12] = DEFAULT_POSE[12] + progress * 2.5 # right knee
        
        # Hip pitch (extreme lean body down)
        d.ctrl[2] = DEFAULT_POSE[2] - progress * 1.5 # left hip pitch
        d.ctrl[11] = DEFAULT_POSE[11] + progress * 1.5 # right hip pitch
        
        # Neck and Head down (extreme)
        d.ctrl[5] = DEFAULT_POSE[5] + progress * 1.5 # neck pitch
        d.ctrl[6] = DEFAULT_POSE[6] + progress * 1.0 # head pitch
        
        for _ in range(10):
            mujoco.mj_step(m, d)
            
        beak_z = d.xpos[lower_beak_id][2]
        trunk_z = d.xpos[trunk_base_id][2]
        beak_x = d.xpos[lower_beak_id][0]
        
        if beak_z < min_beak_z:
            min_beak_z = beak_z
            
    print(f"Kneel final state: beak_z={beak_z:.4f}, trunk_z={trunk_z:.4f}, beak_x={beak_x:.4f}")
    if trunk_z < 0.05:
        print("WARNING: Trunk hit the floor! The duck fell over.")
    elif beak_z <= 0.01:
        print("SUCCESS! Beak reached the towel on the floor!")

if __name__ == "__main__":
    main()
