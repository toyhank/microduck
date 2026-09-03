import mujoco
import mujoco.viewer
import time
import json
import google.generativeai as genai
import os
import cv2
import numpy as np

# Configure Gemini
# NOTE: Set your API key in the environment variable GEMINI_API_KEY
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-robotics-er-2-preview')

XML_PATH = "microduck/kinematics/assets/alpha/robot_walk_visible.xml"

def get_observation(m, d):
    obs = {}
    for i in range(m.njnt):
        jnt_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        if jnt_name:
            obs[jnt_name] = d.qpos[m.jnt_qposadr[i]].item()
    return obs

def set_targets(m, d, targets):
    for jnt_name, angle in targets.items():
        jnt_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
        if jnt_id != -1:
            # Simple direct position assignment for demonstration
            # In a real setup, this would be the target for a PD controller
            d.qpos[m.jnt_qposadr[jnt_id]] = angle

def main():
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)

    # Initialize viewer
    with mujoco.viewer.launch_passive(m, d) as viewer:
        print("Starting Microduck Gemini Simulation Environment...")
        
        # Get joint names and limits for the prompt
        joint_info = {}
        for i in range(m.njnt):
            jnt_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
            if jnt_name and jnt_name != "trunk_base_freejoint":
                jnt_info[jnt_name] = {
                    "range": m.jnt_range[i].tolist()
                }

        system_prompt = f"""You are a controller for the Microduck bipedal robot.
The robot has the following joints and limits (in radians):
{json.dumps(joint_info, indent=2)}

At each step, you will receive the current joint angles.
You must output a JSON object containing the target joint angles you want to set.
Example output:
```json
{{
  "head_yaw": 0.5,
  "left_knee": -0.2
}}
```
Only output the JSON object, nothing else.
"""

        print("System prompt for Gemini:")
        print(system_prompt)
        
        chat = model.start_chat(history=[])
        
        # Let the simulation settle
        for _ in range(100):
            mujoco.mj_step(m, d)
            viewer.sync()
            time.sleep(m.opt.timestep)

        while viewer.is_running():
            obs = get_observation(m, d)
            
            # Send observation to Gemini
            prompt = f"Current State: {json.dumps(obs)}"
            print(f"Sending state to Gemini...")
            try:
                response = chat.send_message(system_prompt + "\n\n" + prompt)
                response_text = response.text.strip()
                
                # Extract JSON
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].strip()
                
                targets = json.loads(response_text)
                print(f"Gemini decided targets: {targets}")
                
                set_targets(m, d, targets)
            except Exception as e:
                print(f"Error calling Gemini or parsing response: {e}")
            
            # Step the simulation for 1 second (100 steps of 10ms)
            for _ in range(100):
                mujoco.mj_step(m, d)
                viewer.sync()
                time.sleep(m.opt.timestep)

if __name__ == "__main__":
    main()
