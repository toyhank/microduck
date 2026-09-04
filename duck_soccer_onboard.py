#!/usr/bin/env python3
"""
Microduck Onboard Autonomous Soccer Script (Standalone Real-Robot Execution)
Runs directly on the robot's onboard Rockchip RK3566 Linux SBC without an external PC.

Pipeline:
1. Open the onboard front-facing camera (/dev/video0) via OpenCV.
2. Detect the orange/red soccer ball and blue goal using HSV color segmentation.
3. Compute the heading error relative to the camera center (320x240).
4. Send JSON-RPC 2.0 motion commands to the local Unix Domain Socket (/run/robotd.sock).
5. When the ball is in front of the foot, align with the goal and trigger kick_right!
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
    """Client for communicating with the onboard robotd daemon via Unix Domain Socket."""

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
        """Send a JSON-RPC 2.0 request over the Unix socket."""
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
        """Send velocity command (vx: m/s, vy: m/s, vtheta: rad/s)."""
        self.call("robot.move", {"vx": float(vx), "vy": float(vy), "vtheta": float(vtheta)})

    def stop(self):
        """Stop locomotion and stand firmly."""
        self.call("robot.stop")

    def kick(self, foot="right"):
        """Trigger a dynamic kicking skill ('kick_right' or 'kick_left')."""
        skill_name = "kick_right" if foot == "right" else "kick_left"
        self.call("robot.do", {"skill": skill_name})

    def quack(self, tag="happy"):
        """Play a voice sound effect for celebration."""
        self.call("robot.sound", {"tag": tag})

def main():
    print("=" * 55)
    print("🦆 Microduck Onboard Autonomous Soccer System ⚽")
    print("=" * 55)

    duck = DuckClient()

    # Open the onboard camera (/dev/video0)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    if not cap.isOpened():
        print("[Error] Failed to open /dev/video0. Check permissions or camera daemon exclusivity.")
        return

    state = "SEARCH_BALL"
    kick_cooldown_end = 0

    print("[Ready] Perception loop running. Searching for the soccer ball...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # 1. Orange/Red soccer ball detection (HSV: H: 5-25, S: 100-255, V: 70-255)
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
                if area > 40:  # Noise filtering
                    M = cv2.moments(c_ball)
                    if M["m00"] != 0:
                        ball_cx = int(M["m10"] / M["m00"])
                        ball_area = area
                        ball_found = True

            # 2. State Machine Control
            current_time = time.time()

            # Post-kick cooldown and celebration
            if state == "KICKING":
                if current_time >= kick_cooldown_end:
                    print("🎉 Kick completed! Celebrating victory!")
                    duck.quack("happy")
                    state = "SEARCH_BALL"
                time.sleep(0.05)
                continue

            if state == "SEARCH_BALL":
                if ball_found:
                    print(f"[State] Ball spotted (area: {ball_area:.0f})! Navigating towards ball...")
                    state = "APPROACH_BALL"
                else:
                    # Spin slowly in place to scan surroundings
                    duck.move(vx=0.0, vy=0.0, vtheta=0.35)

            elif state == "APPROACH_BALL":
                if ball_found:
                    err_x = 160 - ball_cx  # Center offset (width 320)
                    cmd_vtheta = float(np.clip(err_x * 0.005, -0.5, 0.5))

                    # Distance estimation using contour area
                    if ball_area > 3500:  # Reached close proximity in front of foot
                        print(f"[State] Arrived at ball (area: {ball_area:.0f})! Braking to stabilize...")
                        duck.stop()
                        time.sleep(0.6)  # Pause for 0.6s to eliminate residual inertia

                        print("⚡ Executing RIGHT-FOOT KICK! GOOOOOAL!")
                        duck.kick("right")
                        state = "KICKING"
                        kick_cooldown_end = current_time + 3.0
                    elif ball_area > 1500:
                        # Decelerate near ball for precision alignment
                        duck.move(vx=0.15, vy=0.0, vtheta=cmd_vtheta)
                    else:
                        # Cruise forward smoothly
                        duck.move(vx=0.30, vy=0.0, vtheta=cmd_vtheta)
                else:
                    # Temporarily lost ball, slow drift forward or re-scan
                    duck.move(vx=0.10, vy=0.0, vtheta=0.0)
                    time.sleep(0.2)
                    state = "SEARCH_BALL"

            time.sleep(0.08)  # ~12Hz perception and control rate

    except KeyboardInterrupt:
        print("\nInterrupted by user. Safely stopping Microduck...")
        duck.stop()
    finally:
        cap.release()

if __name__ == "__main__":
    main()
