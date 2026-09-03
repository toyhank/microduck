import mujoco
import mujoco.viewer
import time
import json
import os
import cv2
import numpy as np
from PIL import Image
from google import genai
from google.genai import types
import onnxruntime as ort

# Enforce local proxy for httpx (which the new SDK uses)
os.environ["HTTP_PROXY"] = os.environ.get("HTTP_PROXY", "http://127.0.0.1:7890")
os.environ["HTTPS_PROXY"] = os.environ.get("HTTPS_PROXY", "http://127.0.0.1:7890")

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL_ID = 'gemini-robotics-er-2-preview'

XML_PATH = "microduck_rl/src/mjlab_microduck/robot/microduck/scene_walk.xml"
POLICY_WALK = "microduck/policies/alpha_walking.onnx"

# Reference standing pose for the walking policy
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

    renderer = mujoco.Renderer(m, 480, 640)
    
    # Load the Reinforcement Learning walking policy
    session_walk = ort.InferenceSession(POLICY_WALK)
    input_name = session_walk.get_inputs()[0].name
    output_name = session_walk.get_outputs()[0].name
    
    imu_ang_vel_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
    trunk_base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    
    joint_qpos_indices = [int(m.jnt_qposadr[m.actuator_trnid[i, 0]]) for i in range(m.nu)]
    joint_qvel_indices = [int(m.jnt_dofadr[m.actuator_trnid[i, 0]]) for i in range(m.nu)]
    
    last_action = np.zeros(m.nu, dtype=np.float32)
    world_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    system_prompt = """You are the high-level brain for a Microduck bipedal robot.
You control the robot by setting its forward walking speed and turning speed.
Your goal is to walk towards the RED BLOCK.

INSTRUCTIONS (CRITICAL):
- You MUST ALWAYS output `"forward": 0.3` to keep walking towards the block, unless you have reached it.
- To steer towards the block:
  - If the red block is on the LEFT side of the image, output `"turn": 0.5`.
  - If the red block is on the RIGHT side of the image, output `"turn": -0.5`.
  - If it is perfectly centered, output `"turn": 0.0`.
- ONLY if the red block is HUGE and taking up the entire bottom of the screen (meaning you have reached it), output `"forward": 0.0` and `"turn": 0.0` to stop.

Output ONLY JSON in this format:
```json
{
  "forward": 0.3,
  "turn": 0.5
}
```
"""
    print("Initializing Gemini Session...")
    chat = client.chats.create(model=MODEL_ID, config=types.GenerateContentConfig(system_instruction=system_prompt, temperature=0.0))

    cmd_vx = 0.0
    cmd_vtheta = 0.0

    with mujoco.viewer.launch_passive(m, d) as viewer:
        print("Starting Full Stack: Gemini Vision + RL Walking Control...")
        
        # Save initial qpos (which contains the block's correct position from XML)
        initial_qpos = d.qpos.copy()
        
        stand_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "STAND")
        if stand_id != -1:
            mujoco.mj_resetDataKeyframe(m, d, stand_id)
            
            # The keyframe only has 21 values, so MuJoCo zeros out the block's freejoint!
            # Restore the block's original position from the XML
            d.qpos[21:] = initial_qpos[21:]
            
        mujoco.mj_forward(m, d)
        
        while viewer.is_running():
            # ==========================================
            # 1. High-Level AI: Vision & Path Planning
            # ==========================================
            renderer.update_scene(d, camera="egocentric")
            img_rgb = renderer.render()
            pil_img = Image.fromarray(img_rgb)
            
            print("\nGemini is thinking... (Analyzing camera frame)")
            try:
                response = chat.send_message([pil_img, "Where is the red block? Give me the forward and turn speeds."])
                text = response.text.strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].strip()
                cmd = json.loads(text)
                cmd_vx = float(cmd.get("forward", 0.0))
                cmd_vtheta = float(cmd.get("turn", 0.0))
                print(f"-> Gemini Command: Forward {cmd_vx} m/s | Turn {cmd_vtheta} rad/s")
            except Exception as e:
                print(f"Gemini error: {e}, stopping robot.")
                cmd_vx = 0.0
                cmd_vtheta = 0.0

            # ==========================================
            # 2. Low-Level RL: Balance & Walking 
            # ==========================================
            print("-> RL Policy taking over (Walking for 1.5 seconds...)")
            # Walk for 1.5 seconds (75 steps of 0.02s control dt)
            for _ in range(75):
                if not viewer.is_running(): break
                
                # Build observation for the ONNX policy
                adr = m.sensor_adr[imu_ang_vel_id]
                ang_vel = d.sensordata[adr:adr+3].copy().astype(np.float32)
                quat = d.xquat[trunk_base_id].copy().astype(np.float32)
                proj_grav = quat_rotate_inverse(quat, world_gravity)
                qpos = d.qpos[joint_qpos_indices].copy().astype(np.float32)
                rel_qpos = qpos - DEFAULT_POSE
                qvel = d.qvel[joint_qvel_indices].copy().astype(np.float32)
                
                # 13D command vector
                command = np.zeros(13, dtype=np.float32)
                command[0] = cmd_vx
                command[2] = cmd_vtheta
                
                obs = np.concatenate([ang_vel, proj_grav, rel_qpos, qvel, last_action, command]).astype(np.float32)
                obs_batch = obs.reshape(1, -1)
                
                # Infer next motor targets (50 Hz)
                action = session_walk.run([output_name], {input_name: obs_batch})[0]
                action = action.squeeze(0).astype(np.float32)
                last_action = action.copy()
                
                # Apply targets
                target_positions = DEFAULT_POSE + action * 1.0
                d.ctrl[:] = target_positions
                
                # Step physics (decimation = 10 -> 0.02s elapsed per control step)
                for _ in range(10): 
                    mujoco.mj_step(m, d)
                
                viewer.sync()
                time.sleep(0.02) # Real-time playback sync
            
if __name__ == "__main__":
    main()
