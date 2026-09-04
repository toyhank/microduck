# Microduck Gemini Simulation Environment

This project sets up a simulation environment for the [Microduck](https://github.com/pollen-robotics/microduck) robot and connects it to the Gemini API for agentic control.

## Setup Instructions

1. Ensure you have Python installed. The required packages have already been installed in the `venv` folder if you ran this setup via the agent. Otherwise, run:
   ```bash
   pip install mujoco google-generativeai opencv-python pillow
   ```
2. Set your Gemini API Key as an environment variable:
   - On Windows: `set GEMINI_API_KEY=your_api_key_here`
   - On PowerShell: `$env:GEMINI_API_KEY="your_api_key_here"`

## Available Scripts

### 1. Vision-Language Control (`sim_gemini_vision.py`) **[NEW]**
This script implements a **Visual-Language closed-loop**. It renders the robot's egocentric camera view and sends the raw image along with joint states to Gemini (Era 2 / 2.5 Pro). Gemini then decides how to move its head/neck to "grasp" or look at a target red block.
```bash
.\venv\Scripts\python sim_gemini_vision.py
```
*Note: This script dynamically adds a floor, lighting, a target red block, and cameras to the robot model.*

### 3. Reinforcement Learning Walking (`sim_walk.py`)
This script loads the pre-trained ONNX RL policy (PPO) to control the robot's locomotion. It demonstrates the duck automatically balancing and walking in a figure-8 pattern using the official `alpha_walking.onnx` policy.
```bash
.\venv\Scripts\python sim_walk.py
```

### 4. Traditional CV Visual Servoing (`sim_cv_walk.py`) **[NEW]**
Uses traditional OpenCV image processing (color thresholding and contour detection) combined with the RL walking policy. Because it doesn't need to wait for a cloud LLM, the visual feedback runs at 10Hz and the walking is perfectly fluid!
```bash
.\venv\Scripts\python sim_cv_walk.py
```

### 5. Lunar Gravity Locomotion (`scratch/test_moon.py`) **[NEW]**
MuJoCo simulation under Moon gravity (`-1.62 m/s²`) using a specialized retrained PPO RL policy (`microduck/policies/moon_walking.onnx`). It optimizes foot contact and stance under low normal forces to eliminate floating and slippage.
```bash
.\venv\Scripts\python scratch/test_moon.py
```

### 6. Autonomous CV Soccer Simulation (`sim_duck_soccer.py`) **[NEW]**
Autonomous Microduck robot soccer simulation with computer vision:
- Uses the egocentric head camera and OpenCV to find the soccer ball and goal.
- Uses the `alpha_walking.onnx` policy for visual servoing toward the ball.
- Aligns with the goal and activates `ball_kick_right.onnx` to dynamically kick the ball.
- Detects goals and triggers a celebration animation!
```bash
.\venv\Scripts\python sim_duck_soccer.py
```

To retrain the lunar walking policy from scratch using GPU in `microduck_rl`:
```bash
cd microduck_rl
.\.venv\Scripts\python.exe -u -m mjlab_microduck.train_cli Mjlab-Velocity-Flat-Moon-MicroDuck --agent.logger tensorboard --env.scene.num-envs 1024 --agent.max-iterations 3000
```

## How the Vision Script works

- The script uses `mujoco.Renderer` to capture a 480x640 RGB image from the `egocentric` camera mounted in the duck's head.
- The image is converted to a PIL Image and sent natively in the Prompt to Gemini's multi-modal API.
- Gemini is given a system prompt explaining the robot's capabilities (joint limits) and is asked to respond with target joint angles in JSON format to approach the red block.
- The script parses Gemini's response and updates the robot's joints.
