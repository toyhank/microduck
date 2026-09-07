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
import cv2

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
        if duck.sock is not None:
            duck.sock.close()
        return

    from microduck_soccer.perception import BallDetector, GoalDetector
    from microduck_soccer.control import SoccerStateMachine

    # Use the same perception and state machine as simulation. Camera FOV and
    # blind-advance duration still require calibration on the physical robot.
    ball_detector = BallDetector(cam_width, cam_height, fovy_deg=90.0)
    goal_detector = GoalDetector(cam_width, cam_height, fovy_deg=90.0)
    controller = SoccerStateMachine()
    previous_mode = None
    frame_failures = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                duck.stop()
                previous_mode = "stand"
                frame_failures += 1
                if frame_failures >= 5:
                    raise RuntimeError("Camera stream lost")
                controller = SoccerStateMachine()
                time.sleep(0.05)
                continue
            frame_failures = 0
            if frame.shape[:2] != (cam_height, cam_width):
                frame = cv2.resize(frame, (cam_width, cam_height))
            state, vx, vy, vyaw, mode, trigger = controller.update(
                ball_detector.detect(frame), goal_detector.detect(frame), time.monotonic()
            )
            if trigger:
                response = duck.kick("right")
                if response is None or "error" in response:
                    raise RuntimeError(f"Kick request failed: {response}")
                print("Right-foot kick requested; contact/goal not verified.")
            elif mode == "walk":
                duck.move(vx=vx, vy=vy, vyaw=vyaw)
            elif mode == "stand" and previous_mode != "stand":
                duck.stop()
            previous_mode = mode
            time.sleep(0.08)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        duck.stop()
        cap.release()
        if duck.sock is not None:
            duck.sock.close()


if __name__ == "__main__":
    main()
