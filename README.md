# 🦆 Microduck Soccer: Closed-Loop Visual Servoing & Bipedal Kick System ⚽

<p align="center">
  <a href="README.md"><b>English</b></a> | <a href="README_zh.md"><b>中文文档</b></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Robot-Microduck-ffcc00?style=for-the-badge&logo=android" alt="Microduck">
  <img src="https://img.shields.io/badge/Physics-MuJoCo_3.x-blue?style=for-the-badge" alt="MuJoCo">
  <img src="https://img.shields.io/badge/Vision-OpenCV_4.x-green?style=for-the-badge&logo=opencv" alt="OpenCV">
  <img src="https://img.shields.io/badge/RL-ONNX_Runtime-purple?style=for-the-badge&logo=onnx" alt="ONNX">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/License-Apache_2.0-red?style=for-the-badge" alt="License">
</p>

<p align="center">
  <em>A simulator-first visual servoing layer for <b>Pollen Robotics Microduck</b> that closes the perception-action loop around the official ball-blind <code>BallKick</code> policy.</em><br>
  <em>Features strict vision-only navigation (no ground-truth state cheats), monocular metric depth estimation, physical bipedal kicking (no ball teleportation), and an automated benchmark suite.</em>
</p>

---

## 📖 Motivation & Technical Positioning

In the official [Pollen Robotics Microduck](https://github.com/pollen-robotics/microduck) reinforcement learning stack, the pre-trained `ball_kick_right.onnx` policy is intentionally **ball-blind**: it executes a dynamic kicking motion from a standing stance, assuming an operator or high-level behavior has already navigated the robot to the ball. Furthermore, official architecture documents designate ball play (`approach / line up / kick`) as a planned perception-driven autonomous behavior.

While projects like `quackd` (2D simulator) and recent community edge implementations (e.g. RDK X5) explore ball tracking, **Microduck Soccer** provides a clean, simulator-first, strict visual servoing layer designed directly against the official Microduck MJCF models, sensor conventions, and `robotd` runtime:

- **100% Strict Vision (`--mode strict`)**: The controller relies strictly on RGB head camera frames and robot proprioception (IMU, joint encoders). All ground-truth state variables (`d.xpos`, `d.qvel`) are strictly quarantined to an external evaluator for benchmarking.
- **Physical Dynamic Kick (No Teleportation)**: Navigates the physical robot until the foot strikes the real ball. No artificial ball teleportation or snapping.
- **Monocular Metric Depth Estimation**: Solves metric distance from known ball diameter ($D = 70\text{ mm}$) using pinhole geometry:
  $$Z \approx \frac{f \cdot D}{d}$$
- **Official `robotd` Control Chain Alignment**: Implements official action scaling ($0.9$ walk, $1.0$ kick/stand), first-order joint low-pass filters (legs $\alpha=0.7$, head $\alpha=0.5$), and official 0.5s kick duration windows.
- **Onboard RPC Protocol Compliance**: Real-robot script (`duck_soccer_onboard.py`) strictly adheres to `duck-ipc-proto`: continuous `robot.move` notifications with `vyaw` (not `vtheta`), discrete `robot.do` requests, and official `chirp` voice tags.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph SENSE ["1. Perception Layer (10Hz OpenCV)"]
        Cam["Egocentric Head Camera (320x240 @ 10Hz)"] --> BGR["Raw RGB Frame"]
        BGR --> BallDet["BallDetector: HSV + MinEnclosingCircle"]
        BGR --> GoalDet["GoalDetector: Blue Feature Extraction"]
        BallDet --> Depth["Metric Depth Z = (f * D) / d & Bearing"]
        GoalDet --> GoalBearing["Goal Bearing & Heading Angle"]
    end

    subgraph BRAIN ["2. Decision & Visual Servoing"]
        Depth & GoalBearing --> FSM{"SoccerStateMachine (Strict Mode)"}
        FSM -->|Ball not visible| S1["SEARCH_BALL: In-place scan rotation"]
        FSM -->|Ball detected| S2["APPROACH_BALL: Omnidirectional visual servoing with drift compensation"]
        FSM -->|cy >= 230 (blind spot)| S3["TERMINAL_APPROACH: Calibrated 0.26m strike advance"]
        FSM -->|Advance complete| S4["ALIGN_KICK: Settle stance in firm standing pose"]
        FSM -->|Stance settled| S5["KICK: Trigger ball_kick_right (0.5s window)"]
        FSM -->|Kick complete| S6["GOAL_CHECK: Observe trajectory & follow-up"]
        S6 -->|Goal confirmed| S7["CELEBRATE: Victory head nod"]
    end

    subgraph ACT ["3. Policy Execution Layer (50Hz Low-Pass Filtered)"]
        S1 & S2 & S3 --> WalkPol["alpha_walking.onnx (Scale 0.9, Lowpass 0.7/0.5)"]
        S4 & S6 & S7 --> StandPol["alpha_stand.onnx (Scale 1.0)"]
        S5 --> KickPol["ball_kick_right.onnx (Scale 1.0, 0.5s duration)"]
    end

    subgraph EVAL ["4. Isolated Evaluation & Benchmark"]
        MuJoCo["MuJoCo Physics Engine"] -. Ground Truth .-> Eval["SoccerEvaluator (Foot contact, velocity, goal, fall)"]
    end
```

---

## 🎯 State Machine Specification

Because the forward-facing egocentric camera cannot see objects directly beneath the beak ($Z \le 0.35\text{ m}$), the controller solves the terminal camera blind spot through a RoboCup-standard visual dead-reckoning strike sequence:

| State | Perception Trigger | Control Action | Transition Condition |
| :--- | :--- | :--- | :--- |
| **`SEARCH_BALL`** | `ball.visible == False` | In-place yaw spin ($v_{\text{yaw}} = 0.40\text{ rad/s}$) | `ball.visible == True` |
| **`APPROACH_BALL`** | Ball bearing $\theta$ and depth $Z$ | Visual servoing steering ($v_{\text{yaw}} = -2.0 \theta$), forward march ($v_x = 0.35$), and lateral drift compensation ($v_y$) | Ball touches bottom of frame ($c_y \ge 230$) |
| **`TERMINAL_APPROACH`** | Ball enters beak blind spot | Fixed $0.26\text{ m}$ forward dead-reckoning advance ($v_x = 0.35\text{ m/s}$, $t = 1.52\text{ s}$) directly into the foot strike volume | Timer expires ($t \ge 1.52\text{ s}$) |
| **`ALIGN_KICK`** | Ball in strike volume | Stand firmly in `alpha_stand` to eliminate forward inertia | Stance settled ($t \ge 0.30\text{ s}$) |
| **`KICK`** | Stance settled and aligned | Swap ONNX policy to `ball_kick_right.onnx` for dynamic single-leg strike | Kick duration expires ($0.5\text{ s}$) |
| **`GOAL_CHECK`** | Kick finished | Stand upright and observe ball trajectory | Ball scores or re-engages `SEARCH_BALL` |
| **`CELEBRATE`** | Evaluator confirms goal | Pitch head up and down rhythmically, victory celebration | Timer expires |

---

## 🚀 Simulation Quickstart

### 1. Installation
```bash
git clone https://github.com/toyhank/microduck.git
cd microduck
pip install -r requirements.txt
```

### 2. Run the 3D Soccer Simulation (Strict Vision Mode)
```bash
# Strict mode: 100% vision, physical contact kicking, no cheating
python sim_duck_soccer.py --mode strict

# Demo mode (for visual testing with oracle assistance)
python sim_duck_soccer.py --mode demo
```

### 3. Run the Automated Benchmark Suite
Run randomized trials (random ball distance, random initial yaw) to compute quantitative success metrics:
```bash
python benchmark.py --trials 10
```

Example benchmark report:
```text
============================================================
                  BENCHMARK RESULTS
============================================================
  Total Trials:               10
  Ball Detection Rate:        100.0%
  Approach Success Rate:      90.0%
  Kick Contact Rate:          80.0%
  Goal Scoring Rate:          70.0%
  Mean Time to Kick:          6.45 s
  Fall Rate:                  0.0%
============================================================
```

---

## 🤖 Real-Robot Onboard Deployment

The script [`duck_soccer_onboard.py`](duck_soccer_onboard.py) runs directly on the Microduck's onboard Rockchip RK3566 Linux SBC:

### 1. Protocol Architecture
- **Camera**: Captures from `/dev/video0` via OpenCV.
- **Local IPC Socket**: Connects directly to `/run/robotd.sock` over JSON-RPC 2.0.
  - Continuous velocity control: `notify("robot.move", {"vx": 0.3, "vy": 0.0, "vyaw": ...})`
  - Discrete skill triggering: `request("robot.do", {"skill": "kick_right"})`
  - Voice feedback: `notify("robot.sound", {"tag": "chirp"})`

### 2. Deployment Steps
```bash
# 1. Copy script to the robot
scp duck_soccer_onboard.py radxa@<DUCK_IP>:~/

# 2. Run on the robot
ssh radxa@<DUCK_IP>
python3 duck_soccer_onboard.py
```

---

## 📂 Repository Structure

```text
microduck/
├── sim_duck_soccer.py        # ⚽ Simulation entry point (--mode strict / demo)
├── benchmark.py              # 🏁 Automated benchmark suite
├── duck_soccer_onboard.py    # 🤖 Compliant real-robot onboard runner
├── requirements.txt          # 📦 Dependencies
├── LICENSE                   # 📄 Apache-2.0 License
├── NOTICE                    # 📄 Attribution & upstream notices
├── README.md                 # 📖 English documentation
├── README_zh.md              # 📖 Chinese documentation
│
├── microduck_soccer/         # 📦 Core Python package
│   ├── perception/           # Monocular depth and goal detection
│   ├── control/              # Visual servoing & strict state machine
│   ├── policy/               # Official robotd-aligned ONNX runner
│   └── evaluation/           # Isolated metric evaluator
│
├── microduck_rl/             # 🏟️ MuJoCo robot & pitch models (scene_soccer.xml)
└── microduck/                # 🧠 Trained ONNX locomotion & kicking policies
```

---

## 🤝 Acknowledgments & Prior Work

- Builds upon the [Microduck](https://github.com/pollen-robotics/microduck) bipedal platform by **Pollen Robotics / Hugging Face**.
- Context and inspiration from community works including `quackd` and D-Robotics RDK X5.
- Powered by [DeepMind MuJoCo](https://mujoco.org/).
