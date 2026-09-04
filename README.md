# 🦆 Microduck Soccer (小黄鸭自主视觉足球机器人) ⚽

<p align="center">
  <img src="https://img.shields.io/badge/Robot-Microduck-ffcc00?style=for-the-badge&logo=android" alt="Microduck">
  <img src="https://img.shields.io/badge/Physics-MuJoCo_3.x-blue?style=for-the-badge" alt="MuJoCo">
  <img src="https://img.shields.io/badge/Vision-OpenCV_4.x-green?style=for-the-badge&logo=opencv" alt="OpenCV">
  <img src="https://img.shields.io/badge/RL-ONNX_Runtime-purple?style=for-the-badge&logo=onnx" alt="ONNX">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/License-Apache_2.0-red?style=for-the-badge" alt="License">
</p>

<p align="center">
  <em>首个基于 Pollen Robotics <b>Microduck</b> 双足小黄鸭机器人的端到端<b>自主视觉寻球、逼近、瞄准球门与抬腿大力抽射</b>全闭环系统。</em><br>
  <em>同时支持 <b>MuJoCo 真实物理引擎三维仿真</b> 与 <b>真机（Rockchip RK3566）单机完全脱机运行</b>！</em>
</p>

---

## 📖 项目简介 (Overview)

**Microduck（小黄鸭）** 是一款高 25cm、重约 800g 的 15 自由度双足开源机器人。官方虽然开源了行走与踢球的基础强化学习策略，但截至目前官方路线图中尚未实现**全自主的视觉寻球与踢球闭环系统**（在官方文档中仍标记为待开发特性）。

本项目的核心目标是为 Microduck 赋予**“球场自主感知与竞技大脑”**：
1. **👁️ 自主视觉感知**：无需外部动捕或全局相机，仅依靠小黄鸭头部的第一人称广角摄像头，利用轻量级 OpenCV 实时识别足球与球门。
2. **🏃 双足强化学习行走 (RL Locomotion)**：运行 `alpha_walking.onnx` 策略网络，结合视觉偏角误差（Visual Servoing）平稳转向并快步逼近足球。
3. **🎯 球门对齐与位姿调整**：到达球前时根据球门朝向进行身体朝向对齐，并将足球精确定位在踢球脚的发力甜点位。
4. **⚡ 动态单脚平衡与抬腿抽射 (RL Kicking)**：动态切换至 `ball_kick_right.onnx` 策略网络，小黄鸭左腿单脚单点平衡，右腿向后蓄力后大幅前摆猛烈抽射，将足球强力射入球门！
5. **🎉 入网检测与胜利欢庆**：检测足球滚过球门线落入球网，弹出进球横幅并触发小黄鸭点头欢庆动作。
6. **🤖 1:1 无缝迁移真机**：提供独立的单机真机运行脚本，通过本地 Unix Socket (`/run/robotd.sock`) 与机载 `robotd` 守护进程通信，完全不需要外部笔记本电脑！

---

## 🏗️ 系统架构图 (Architecture)

```mermaid
flowchart TD
    subgraph SENSE ["1. 视觉感知层 (Sensing & CV)"]
        Cam["头戴第一人称摄像头 (320x240 @ 10Hz)"] --> BGR["BGR 原始视频帧"]
        BGR --> HSV["HSV 颜色空间转换"]
        HSV --> MaskBall["橙红高对比掩膜 (足球提取)"]
        HSV --> MaskGoal["蓝色特征掩膜 (球门提取)"]
        MaskBall --> CentroidBall["足球质心 (cx, cy) & 投影面积"]
        MaskGoal --> CentroidGoal["球门朝向 & 目标方位角"]
    end

    subgraph BRAIN ["2. 决策与有限状态机 (FSM State Machine)"]
        CentroidBall & CentroidGoal --> FSM{"有限状态机"}
        FSM -->|未发现足球| S1["SEARCH_BALL: 原地旋转扫描"]
        FSM -->|锁定足球| S2["APPROACH_BALL: 视觉伺服平稳逼近"]
        FSM -->|到达球前 (面积阈值)| S3["ALIGN_KICK: 对准球门方向"]
        FSM -->|对齐完毕| S4["KICK: 触发抬腿抽射"]
        FSM -->|踢球完成| S5["GOAL_CHECK: 判定进球"]
        S5 -->|进球成功| S6["CELEBRATE: 点头欢呼庆祝"]
    end

    subgraph ACT ["3. 策略控制层 (RL Motion Control @ 50Hz)"]
        S1 & S2 --> WalkPol["行走策略 alpha_walking.onnx (Twist 指令控制)"]
        S3 --> AlignPol["微调步态控制"]
        S4 --> KickPol["射门策略 ball_kick_right.onnx (全零命令爆发)"]
        S6 --> StandPol["站立策略 alpha_stand.onnx (头部韵律点头)"]
    end

    subgraph PLATFORM ["4. 执行终端 (Execution)"]
        WalkPol & KickPol & StandPol --> MuJoCoSim["MuJoCo 3D 物理仿真环境 (scene_soccer.xml)"]
        WalkPol & KickPol & StandPol --> RealDuck["Microduck 真机 (Rockchip RK3566 /run/robotd.sock)"]
    end
```

---

## 🎯 状态机详细生命周期 (FSM Details)

| 状态 (State) | 触发条件 | 控制行为 | 退出条件 |
| :--- | :--- | :--- | :--- |
| **`SEARCH_BALL`** | 视野内未检测到有效足球轮廓 | 身体原地慢速旋转 (`vtheta = 0.45 rad/s`)，头部保持前倾扫描 | 检测到有效足球轮廓 (`area > 15`) |
| **`APPROACH_BALL`** | 锁定足球重心 `(cx, cy)` | 计算中心偏角误差 `err_x = 160 - cx`，视觉伺服转向并以 `vx = 0.35 m/s` 快步前行 | 接近足球 (`dist < 0.22m` 或 `area > 1800`) |
| **`ALIGN_KICK`** | 到达球前准备区 | 计算小黄鸭朝向与球门中心连线夹角，原地慢速旋转使身体正对球门，微调球与右脚相对位置 | 偏航误差 `|yaw_diff| < 10°` |
| **`KICK`** | 对齐完成 | 切换为 `ball_kick_right.onnx`，左腿单脚支撑平衡，右腿大幅后摆后猛烈抽射（持续约 2.7s） | 踢球动作时钟周期耗尽 |
| **`GOAL_CHECK`** | 踢球动作完成 | 观察足球飞行轨迹与终点位置 | 球速收敛，或判定球进门线 |
| **`CELEBRATE`** | 足球越过球门线（`x >= 2.75m, |y| < 0.4m`） | 屏幕弹出金色进球特效，小黄鸭切换站立模式并有节奏地点头欢呼 | 庆祝倒计时结束，重置至搜球 |

---

## 🚀 仿真快速上手 (Simulation Quickstart)

### 1. 环境准备
确保已安装 Python 3.10 或更高版本，然后安装依赖库：

```bash
# 建议在项目虚拟环境中运行
pip install -r requirements.txt
```

### 2. 一键启动 3D 足球仿真
在项目根目录下运行：

```bash
python sim_duck_soccer.py
```

### 3. 运行窗口说明
启动后系统将同时渲染两个窗口：
1. **MuJoCo 3D 主物理视窗**：
   - 全局观察小黄鸭在绿茵场上的走位、单脚平衡以及抬腿踢球进网的真实物理过程。
   - 可通过鼠标右键旋转视角、滚轮缩放、左键拖拽视角。
2. **OpenCV 第一人称 HUD 视窗**：
   - 展示从小黄鸭头顶相机所看到的真实视角。
   - 实时绘制：足球检测包围盒（橙圈）、球门标靶（蓝框）、当前状态机阶段、距离与速度遥测信息、中央准星。

---

## 🤖 真机单机脱机部署指南 (Real Robot Onboard Deployment)

小黄鸭机身内置一颗 **Rockchip RK3566（四核 64 位 ARM Cortex-A55 Linux）** 主板。本项目提供的 [`duck_soccer_onboard.py`](duck_soccer_onboard.py) 可以**直接在小黄鸭本机独立运行，完全不需要连接电脑**！

### 1. 部署原理
- **视觉**：直接由 OpenCV 捕获机载头部摄像头 `/dev/video0`。
- **通信**：通过 Python 标准库直接连接小黄鸭本机 Unix 套接字 **`/run/robotd.sock`**，发送 JSON-RPC 2.0 指令：
  - 走路：`{"method": "robot.move", "params": {"vx": 0.35, "vy": 0.0, "vtheta": ...}}`
  - 踢球：`{"method": "robot.do", "params": {"skill": "kick_right"}}`
  - 叫声：`{"method": "robot.sound", "params": {"tag": "happy"}}`

### 2. 部署与运行步骤

#### 第一步：将脚本传入小黄鸭
电脑与小黄鸭连入同一 Wi-Fi 后，使用 scp 传输脚本：
```bash
scp duck_soccer_onboard.py radxa@<小黄鸭IP>:~/
```

#### 第二步：登录小黄鸭测试运行
```bash
ssh radxa@<小黄鸭IP>
python3 duck_soccer_onboard.py
```

#### 第三步：配置开机完全脱机自启（真正实现通电即踢）
在小黄鸭终端中创建一个自启服务：
```bash
sudo systemctl edit --force --full duck-soccer.service
```
粘贴以下内容并保存：
```ini
[Unit]
Description=Microduck Autonomous Soccer Onboard System
After=robotd.service
Wants=robotd.service

[Service]
Type=simple
User=radxa
WorkingDirectory=/home/radxa
ExecStart=/usr/bin/python3 /home/radxa/duck_soccer_onboard.py
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
```
启用该服务：
```bash
sudo systemctl enable --now duck-soccer.service
```
现在，断开所有 Wi-Fi 和 SSH 连接，只要给小黄鸭通电，把它放在地板上和一个橙色小球在一起，它就会**完全自主地环视四周、迈步奔向足球并大力抽射**！

### 3. 真机实操小贴士
1. **足球推荐**：请使用 70mm 直径、15g~30g 的**轻质空心地板球 (Floorball)、儿童发泡软球或塑料小球**，切勿使用成人真皮足球以保护舵机齿轮。
2. **射门前刹车缓冲**：实物地面摩擦力可能存在滑移，脚本已内置接近球后先下发 `robot.stop()` 稳立 0.6 秒以消除运动惯性，确保单脚踢球时机身不晃动。
3. **光照校准**：实际房间灯光若有偏色，可在 `duck_soccer_onboard.py` 中微调 `lower_ball` 与 `upper_ball` 的 HSV 范围。

---

## 📂 项目结构 (Repository Structure)

```text
microduck/
├── sim_duck_soccer.py        # ⚽ MuJoCo 仿真与 CV 视觉闭环主程序
├── duck_soccer_onboard.py    # 🤖 真机板载完全脱机独立运行脚本
├── requirements.txt          # 📦 项目核心依赖列表
├── README.md                 # 📖 项目详细说明文档
│
├── microduck_rl/             # 🏟️ MuJoCo 机器人与物理场景模型
│   └── src/mjlab_microduck/robot/microduck/
│       ├── scene_soccer.xml  # ⚽ 足球场、球门、球网与足球场景配置
│       ├── robot_allcollisions.xml # 鸭子 15-DOF 碰撞几何与相机定义
│       └── assets/           # STL 网格模型与贴图
│
└── microduck/                # 🧠 强化学习运控策略库
    └── policies/
        ├── alpha_walking.onnx    # 官方 PPO 双足平稳行走网络 (61D -> 14D)
        ├── ball_kick_right.onnx  # 官方 PPO 右腿凌空抽射网络 (61D -> 14D)
        ├── ball_kick_left.onnx   # 官方 PPO 左腿抽射网络 (61D -> 14D)
        └── alpha_stand.onnx      # 官方 PPO 站立平衡网络 (61D -> 14D)
```

---

## 🗺️ 未来展望 (Roadmap)

- [x] 基于第一人称相机的橙红色足球检测与追踪
- [x] 视觉伺服平稳行走逼近足球
- [x] 基于球门位置的射门对齐与发力点调整
- [x] 动态切换强化学习射门策略完成抽射
- [x] 真机 Rockchip RK3566 单机完全脱机运行脚本
- [ ] **守门鸭对战模式**：引入第二只小黄鸭部署在门前，利用视觉左右横跳扑救。
- [ ] **双鸭 2v2 足球赛**：实现传球协助与团队对抗策略。
- [ ] **NPU 模型加速**：将视觉检测网络转换为 RKNN 格式，直接在板载 NPU 上以 60FPS 运行。

---

## 🤝 致谢 (Acknowledgments)

- 感谢 **[Pollen Robotics](https://pollen-robotics.com)** 与 **[Hugging Face](https://huggingface.co)** 开源优秀的 Microduck 双足机器人平台及其训练策略。
- 物理仿真由 **[DeepMind MuJoCo](https://mujoco.org/)** 强力驱动。
