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
        FSM -->|Ball detected| S2["APPROACH_BALL: Smooth deceleration visual servoing"]
        FSM -->|Z <= 0.18m| S3["ALIGN_KICK: Orient toward goal & position right foot"]
        FSM -->|Aligned| S4["KICK: Trigger ball_kick_right (0.5s window)"]
        FSM -->|Kick complete| S5["GOAL_CHECK: Observe trajectory"]
        S5 -->|Goal confirmed| S6["CELEBRATE: Victory head nod"]
    end

    subgraph ACT ["3. Policy Execution Layer (50Hz Low-Pass Filtered)"]
        S1 & S2 --> WalkPol["alpha_walking.onnx (Scale 0.9, Lowpass 0.7/0.5)"]
        S4 --> KickPol["ball_kick_right.onnx (Scale 1.0, 0.5s duration)"]
        S6 --> StandPol["alpha_stand.onnx (Scale 1.0, Rhythmic nod)"]
    end

    subgraph EVAL ["4. Isolated Evaluation & Benchmark"]
        MuJoCo["MuJoCo Physics Engine"] -. Ground Truth .-> Eval["SoccerEvaluator (Ball distance, contact, goal, fall)"]
    end
```

---

## 🎯 State Machine Specification

| State | Perception Trigger | Control Action | Transition Condition |
| :--- | :--- | :--- | :--- |
| **`SEARCH_BALL`** | `ball.visible == False` | In-place yaw spin ($v_{\text{yaw}} = 0.40\text{ rad/s}$) | `ball.visible == True` |
| **`APPROACH_BALL`** | Ball bearing and estimated distance $Z$ | Visual servoing steering ($v_{\text{yaw}} = -k_p \theta$) and distance-adaptive forward speed ($v_x = \text{clip}(k_d (Z - Z_{\text{target}}), 0.12, 0.35)$) | $Z \le 0.18\text{ m}$ (kicking zone) |
| **`ALIGN_KICK`** | Ball in kicking zone | Stand firmly, orient heading toward detected goal bearing | Heading alignment within $\pm 8.5^\circ$ |
| **`KICK`** | Stance settled and aligned | Swap ONNX policy to `ball_kick_right.onnx` for exactly $0.5\text{ s}$ | $t \ge t_{\text{kick}} + 0.5\text{ s}$ |
| **`GOAL_CHECK`** | Kick finished | Stand upright and observe ball trajectory | Ball enters goal or velocity settles |
| **`CELEBRATE`** | Evaluator confirms goal | Pitch head up and down rhythmically, play victory sound | Timer expires |

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
