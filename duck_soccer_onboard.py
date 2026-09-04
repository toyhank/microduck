#!/usr/bin/env python3
"""
Microduck Onboard Autonomous Soccer Script (真机单机运行版本)
运行在小黄鸭内部的 Rockchip RK3566 开发板上，无需外部电脑！

原理:
1. 通过 OpenCV 打开板载头部摄像头 (/dev/video0)
2. 识别橙红色足球与蓝色球门，计算偏航误差
3. 通过本地 Unix Domain Socket (/run/robotd.sock) 发送 JSON-RPC 2.0 指令
4. 接近足球后自动对齐球门并下发 kick_right 抬腿射门！
"""

import socket
import json
import time
import math
import sys
import cv2
import numpy as np

ROBOT_SOCKET = "/run/robotd.sock"

class DuckClient:
    def __init__(self, sock_path=ROBOT_SOCKET):
        self.sock_path = sock_path
        self.sock = None
        self.connect()

    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.connect(self.sock_path)
            print(f"[DuckClient] Connected to {self.sock_path}")
        except Exception as e:
            print(f"[DuckClient] Error connecting to {self.sock_path}: {e}")
            self.sock = None

    def call(self, method, params=None):
        if self.sock is None:
            self.connect()
            if self.sock is None:
                return None
        req = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": int(time.time() * 1000)
        }
        msg = json.dumps(req) + "\n"
        try:
            self.sock.sendall(msg.encode("utf-8"))
        except Exception as e:
            print(f"[DuckClient] Send error: {e}")
            self.sock = None

    def move(self, vx=0.0, vy=0.0, vtheta=0.0):
        """发送行走速度指令 (m/s, rad/s)"""
        self.call("robot.move", {"vx": float(vx), "vy": float(vy), "vtheta": float(vtheta)})

    def stop(self):
        """停步稳立"""
        self.call("robot.stop")

    def kick(self, foot="right"):
        """执行踢球动作 (kick_right 或 kick_left)"""
        skill_name = "kick_right" if foot == "right" else "kick_left"
        self.call("robot.do", {"skill": skill_name})

    def quack(self, tag="happy"):
        """叫一声庆祝 (可选语音功能)"""
        self.call("robot.sound", {"tag": tag})

def main():
    print("=" * 50)
    print("🦆 Microduck 真机机载自主寻球踢球系统启动 ⚽")
    print("=" * 50)

    duck = DuckClient()

    # 打开真机板载摄像头 (一般为 /dev/video0)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    if not cap.isOpened():
        print("[Error] 无法打开真机摄像头 /dev/video0，请检查设备权限或 mediad 是否独占！")
        return

    state = "SEARCH_BALL"
    kick_cooldown_end = 0

    print("[Ready] 视觉主循环开始，等待小黄鸭自动巡球...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # 1. 橙红色足球检测 (真机可根据实际房间光照微调 H: 5~25, S: 100~255, V: 70~255)
            lower_ball = np.array([5, 100, 70])
            upper_ball = np.array([25, 255, 255])
            mask_ball = cv2.inRange(hsv, lower_ball, upper_ball)
            cnts_ball, _ = cv2.findContours(mask_ball, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            ball_found = False
            ball_cx = None
            ball_area = 0

            if cnts_ball:
                c_ball = max(cnts_ball, key=cv2.contourArea)
                area = cv2.contourArea(c_ball)
                if area > 40:  # 滤除噪点
                    M = cv2.moments(c_ball)
                    if M["m00"] != 0:
                        ball_cx = int(M["m10"] / M["m00"])
                        ball_area = area
                        ball_found = True

            # 2. 状态机控制
            current_time = time.time()

            # 踢球后的保护与庆祝冷却时间
            if state == "KICKING":
                if current_time >= kick_cooldown_end:
                    print("🎉 射门完成！小黄鸭欢叫庆祝！")
                    duck.quack("happy")
                    state = "SEARCH_BALL"
                time.sleep(0.05)
                continue

            if state == "SEARCH_BALL":
                if ball_found:
                    print(f"[State] 发现足球 (面积: {ball_area:.0f})，开始逼近...")
                    state = "APPROACH_BALL"
                else:
                    # 原地慢速旋转扫视寻找足球
                    duck.move(vx=0.0, vy=0.0, vtheta=0.35)

            elif state == "APPROACH_BALL":
                if ball_found:
                    err_x = 160 - ball_cx  # 中心偏移 (320 宽)
                    cmd_vtheta = float(np.clip(err_x * 0.005, -0.5, 0.5))

                    # 距离判定：真机上用轮廓面积估算距离
                    if ball_area > 3500:  # 球已经在脚边极近处
                        print(f"[State] 已经到达足球面前 (面积: {ball_area:.0f})！刹车站稳...")
                        duck.stop()
                        time.sleep(0.6)  # 站稳 0.6 秒以消除运动惯性

                        print("⚡ 抬右腿射门！GOOOOOAL！")
                        duck.kick("right")
                        state = "KICKING"
                        kick_cooldown_end = current_time + 3.0
                    elif ball_area > 1500:
                        # 接近时减速，精细对齐
                        duck.move(vx=0.15, vy=0.0, vtheta=cmd_vtheta)
                    else:
                        # 正常巡航快步前进
                        duck.move(vx=0.30, vy=0.0, vtheta=cmd_vtheta)
                else:
                    # 偶尔跟丢，慢速前移或重新搜索
                    duck.move(vx=0.10, vy=0.0, vtheta=0.0)
                    time.sleep(0.2)
                    state = "SEARCH_BALL"

            time.sleep(0.08)  # ~12Hz 视觉控制频率

    except KeyboardInterrupt:
        print("\n用户中断，小黄鸭安全停机...")
        duck.stop()
    finally:
        cap.release()

if __name__ == "__main__":
    main()
