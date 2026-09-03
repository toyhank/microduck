import mujoco
import mujoco.viewer
import time
import json
import os
import cv2
import numpy as np
from PIL import Image

# Enforce local proxy for httpx (which the new SDK uses)
os.environ["HTTP_PROXY"] = os.environ.get("HTTP_PROXY", "http://127.0.0.1:7890")
os.environ["HTTPS_PROXY"] = os.environ.get("HTTPS_PROXY", "http://127.0.0.1:7890")

from google import genai
from google.genai import types

# Configure new Gemini Client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL_ID = 'gemini-robotics-er-2-preview'

XML_PATH = "microduck_rl/src/mjlab_microduck/robot/microduck/scene_walk.xml"

def get_hinge_joints(m):
    hinge_joints = []
    for i in range(m.njnt):
        if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
            jnt_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
            if jnt_name:
                hinge_joints.append(i)
    return hinge_joints

def get_observation(m, d, hinge_joints):
    obs = {}
    for i in hinge_joints:
        jnt_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        obs[jnt_name] = d.qpos[m.jnt_qposadr[i]].item()
    return obs

def set_targets(m, d, targets_deltas, hinge_joints):
    for jnt_name, delta in targets_deltas.items():
        jnt_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
        if jnt_id in hinge_joints:
            limit_min, limit_max = m.jnt_range[jnt_id]
            current_angle = d.qpos[m.jnt_qposadr[jnt_id]]
            new_angle = max(limit_min, min(limit_max, current_angle + delta))
            d.qpos[m.jnt_qposadr[jnt_id]] = new_angle

def main():
    try:
        m = mujoco.MjModel.from_xml_path(XML_PATH)
        d = mujoco.MjData(m)
    except Exception as e:
        print(f"Failed to load MuJoCo model: {e}")
        return

    renderer = mujoco.Renderer(m, 480, 640)
    hinge_joints = get_hinge_joints(m)
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        print("Starting Microduck Vision-Language Servoing...")
        
        stand_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "STAND")
        if stand_id != -1:
            mujoco.mj_resetDataKeyframe(m, d, stand_id)
        
        base_pos = d.qpos[0:7].copy()
        
        for _ in range(200):
            d.qpos[0:7] = base_pos
            d.qvel[0:6] = 0
            mujoco.mj_step(m, d)
            viewer.sync()
            time.sleep(m.opt.timestep)

        system_prompt = """You are the vision-motor controller for a Microduck robot.
Your goal is to use Visual Servoing to move the robot's beak to touch the red block.
You will see the image from the camera in the robot's head.

INSTRUCTIONS:
1. If the red block is to the LEFT of the center, rotate head LEFT by INCREASING 'head_yaw'.
2. If the red block is to the RIGHT of the center, rotate head RIGHT by DECREASING 'head_yaw'.
3. If the red block is BELOW the center, look DOWN by DECREASING 'head_pitch' or 'neck_pitch'.
4. If the red block is ABOVE the center, look UP by INCREASING 'head_pitch' or 'neck_pitch'.

Output a JSON object with the DELTAS (relative changes) to apply to the joints. Use values between -0.15 and 0.15 radians.
Example:
```json
{
  "head_yaw": 0.05,
  "head_pitch": -0.1
}
```
"""
        print("Initializing Gemini Session (google.genai)...")
        chat = client.chats.create(
            model=MODEL_ID,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.0
            )
        )

        while viewer.is_running():
            d.qpos[0:7] = base_pos
            d.qvel[0:6] = 0
            renderer.update_scene(d, camera="egocentric")
            img_rgb = renderer.render()
            pil_img = Image.fromarray(img_rgb)
            
            obs = get_observation(m, d, hinge_joints)
            prompt = f"Current Joint States: {json.dumps(obs)}\nWhere is the red block in the image? Output the joint deltas to center it and move closer."
            
            print("Sending egocentric camera view and state to Gemini...")
            try:
                response = chat.send_message([pil_img, prompt])
                response_text = response.text.strip()
                
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].strip()
                
                deltas = json.loads(response_text)
                print(f"Gemini decided deltas: {deltas}")
                
                set_targets(m, d, deltas, hinge_joints)
            except Exception as e:
                print(f"Error calling Gemini: {e}")
            
            # Interpolate smoothly for visual effect
            for _ in range(30):
                d.qpos[0:7] = base_pos
                d.qvel[0:6] = 0
                mujoco.mj_forward(m, d)
                viewer.sync()
                time.sleep(0.03) 

if __name__ == "__main__":
    main()
