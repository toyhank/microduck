"""
Goal Perception Module.
Detects goal posts and banner, extracting horizontal bearing for body alignment.
"""

import math
import cv2
import numpy as np

class GoalDetection:
    def __init__(self, cx=None, cy=None, width=0.0, height=0.0, area=0.0, bearing=0.0, visible=False, contour=None):
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        self.area = area
        self.bearing = bearing  # Bearing angle relative to camera axis (rad, positive = right)
        self.visible = visible
        self.contour = contour

class GoalDetector:
    def __init__(self, img_width=320, img_height=240, fovy_deg=90.0):
        self.width = img_width
        self.height = img_height
        self.fovy_rad = math.radians(fovy_deg)
        self.focal_length = (self.height / 2.0) / math.tan(self.fovy_rad / 2.0)

        # HSV color range for blue goal
        # The simulated sky is also blue (S ~= 146). Goal paint is much more
        # saturated; including the sky makes its full-frame box look centered.
        self.lower_hsv = np.array([100, 180, 50])
        self.upper_hsv = np.array([130, 255, 255])
        self.min_area = 25.0

    def detect(self, bgr_img) -> GoalDetection:
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return GoalDetection(visible=False)

        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < self.min_area:
            return GoalDetection(visible=False)

        x, y, w, h = cv2.boundingRect(c)
        if w >= self.width * .9 or h >= self.height * .9:
            return GoalDetection(visible=False)
        cx = int(x + w / 2.0)
        cy = int(y + h / 2.0)

        center_x = self.width / 2.0
        bearing = math.atan2(cx - center_x, self.focal_length)

        return GoalDetection(
            cx=cx,
            cy=cy,
            width=w,
            height=h,
            area=area,
            bearing=bearing,
            visible=True,
            contour=c
        )
