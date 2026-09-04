#!/usr/bin/env python3
"""
Microduck Onboard Autonomous Soccer Script (Standalone Real-Robot Execution)
Runs directly on the robot's onboard Rockchip RK3566 Linux SBC without an external PC.

Features:
- Compliant with official duck-ipc-proto RPC contract:
  * Uses 'vyaw' for angular velocity (denied if 'vtheta' is passed).
  * Uses JSON-RPC 2.0 notifications for high-frequency continuous control ('robot.move').
  * Uses JSON-RPC 2.0 requests for discrete actions ('robot.stop', 'robot.do').
  * Uses official SoundTag ('chirp') for voice feedback.
- Monocular depth estimation: Z = (f * D) / d based on known 70 mm ball diameter.
"""

import socket
import json
import time
import math
import sys
import cv2
import numpy as np

ROBOT_SOCKET = "/run/robotd.sock"
BALL_DIAMETER_METERS = 0.070  # 70 mm physical ball diameter

class DuckClient:
    """Client for communicating with the onboard robotd daemon via Unix Domain Socket."""

    def __init__(self, sock_path=ROBOT_SOCKET):
        self.sock_path = sock_path
        self.sock = None
        self._msg_id = 1
        self.connect()

    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.connect(self.sock_path)
            self.sock.settimeout(1.0)
            print(f"[DuckClient] Connected to {self.sock_path}")
        except Exception as e:
            print(f"[DuckClient] Error connecting to {self.sock_path}: {e}")
            self.sock = None

    def notify(self, method, params=None):
        """Send a JSON-RPC 2.0 notification (no 'id', no response expected). Used for robot.move."""
        if self.sock is None:
            self.connect()
            if self.sock is None:
                return
        req = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {}
        }
        msg = json.dumps(req) + "\n"
        try:
            self.sock.sendall(msg.encode("utf-8"))
        except Exception as e:
            print(f"[DuckClient] Notify send error: {e}")
            self.sock = None

    def request(self, method, params=None):
        """Send a JSON-RPC 2.0 request (has 'id', reads response). Used for discrete skills."""
        if self.sock is None:
            self.connect()
            if self.sock is None:
                return None
        current_id = self._msg_id
        self._msg_id += 1
        req = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": current_id
        }
        msg = json.dumps(req) + "\n"
        try:
            self.sock.sendall(msg.encode("utf-8"))
            # Read single-line JSON response
            raw_resp = b""
            while not raw_resp.endswith(b"\n"):
                chunk = self.sock.recv(1024)
                if not chunk:
                    break
                raw_resp += chunk
            if raw_resp:
                return json.loads(raw_resp.decode("utf-8").strip())
        except Exception as e:
            print(f"[DuckClient] Request error: {e}")
            self.sock = None
        return None

    def move(self, vx=0.0, vy=0.0, vyaw=0.0):
        """Send continuous velocity twist: vx (m/s), vy (m/s), vyaw (rad/s, + = left)."""
        self.notify("robot.move", {"vx": float(vx), "vy": float(vy), "vyaw": float(vyaw)})

    def stop(self):
        """Stop locomotion and stand firmly."""
        self.request("robot.stop")

    def kick(self, foot="right"):
        """Trigger dynamic kick skill: 'kick_right' or 'kick_left'."""
        skill_name = "kick_right" if foot == "right" else "kick_left"
        return self.request("robot.do", {"skill": skill_name})

    def quack(self, tag="chirp"):
        """Play sound tag. Valid tags: 'alarm', 'greet', 'inquire', 'peck', 'chirp', 'coo', 'wheee'."""
        self.notify("robot.sound", {"tag": tag})

def main():
    print("=" * 60)
    print("🦆 Microduck Onboard Autonomous Soccer System ⚽")
    print("   Compliant with official robotd IPC contract")
    print("=" * 60)

    duck = DuckClient()

    # Open the onboard camera (/dev/video0)
    cam_width, cam_height = 320, 240
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

    if not cap.isOpened():
        print("[Error] Failed to open /dev/video0. Check permissions or camera daemon exclusivity.")
        return

    # Focal length for 90-degree FOV: f = (height / 2) / tan(45 deg) = 120 px
    focal_length = 120.0

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
            estimated_dist = 999.0

            if cnts_ball:
                c_ball = max(cnts_ball, key=cv2.contourArea)
                area = cv2.contourArea(c_ball)
                if area > 35:  # Noise filtering
                    (cx_f, cy_f), radius = cv2.minEnclosingCircle(c_ball)
                    diameter_px = radius * 2.0
                    if diameter_px > 2.0:
                        ball_cx = int(cx_f)
                        # Monocular depth: Z = (f * D) / d
                        estimated_dist = (focal_length * BALL_DIAMETER_METERS) / diameter_px
                        ball_found = True

            current_time = time.time()

            # Post-kick cooldown and celebration
            if state == "KICKING":
                if current_time >= kick_cooldown_end:
                    print("🎉 Kick completed! Celebrating victory!")
                    duck.quack("chirp")
                    state = "SEARCH_BALL"
                time.sleep(0.05)
                continue

            if state == "SEARCH_BALL":
                if ball_found:
                    print(f"[State] Ball spotted ({estimated_dist:.2f}m)! Navigating towards ball...")
                    state = "APPROACH_BALL"
                else:
                    # Spin slowly in place to scan surroundings
                    duck.move(vx=0.0, vy=0.0, vyaw=0.35)

            elif state == "APPROACH_BALL":
                if ball_found:
                    # Target bearing: bias slightly right for right-foot kick
                    err_x = 160 - (ball_cx + 10)
                    cmd_vyaw = float(np.clip((err_x / focal_length) * 1.5, -0.45, 0.45))

                    # Monocular distance stopping threshold (~0.18m)
                    if estimated_dist <= 0.18:
                        print(f"[State] Reached kick position ({estimated_dist:.2f}m)! Braking to stabilize...")
                        duck.stop()
                        time.sleep(0.6)  # Settle stance before triggering kick

                        print("⚡ Executing RIGHT-FOOT KICK! GOOOOOAL!")
                        duck.kick("right")
                        state = "KICKING"
                        kick_cooldown_end = current_time + 3.0
                    else:
                        # Smooth deceleration
                        dist_err = estimated_dist - 0.15
                        cmd_vx = float(np.clip(0.5 * dist_err + 0.10, 0.12, 0.32))
                        duck.move(vx=cmd_vx, vy=0.0, vyaw=cmd_vyaw)
                else:
                    # Lost ball briefly, scan or drift
                    duck.move(vx=0.08, vy=0.0, vyaw=0.0)
                    time.sleep(0.2)
                    state = "SEARCH_BALL"

            time.sleep(0.08)  # ~12Hz control loop

    except KeyboardInterrupt:
        print("\nInterrupted by user. Safely stopping Microduck...")
        duck.stop()
    finally:
        cap.release()

if __name__ == "__main__":
    main()
