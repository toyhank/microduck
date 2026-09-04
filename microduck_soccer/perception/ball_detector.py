"""
Monocular Ball Detector with Metric Distance Estimation.
Estimates 3D bearing and depth from a known physical ball diameter (70 mm).
"""

import math
import cv2
import numpy as np

# Physical characteristics
BALL_DIAMETER_METERS = 0.070  # 70 mm hollow ball

class BallDetection:
    def __init__(self, cx=None, cy=None, radius=0.0, area=0.0, distance=0.0, bearing=0.0, visible=False, contour=None):
        self.cx = cx
        self.cy = cy
        self.radius = radius
        self.area = area
        self.distance = distance  # Estimated distance in meters
        self.bearing = bearing    # Bearing angle relative to camera axis (rad, positive = right)
        self.visible = visible
        self.contour = contour

class BallDetector:
    def __init__(self, img_width=320, img_height=240, fovy_deg=90.0):
        self.width = img_width
        self.height = img_height
        self.fovy_rad = math.radians(fovy_deg)
        # Focal length in pixels: f = (height / 2) / tan(fovy / 2)
        self.focal_length = (self.height / 2.0) / math.tan(self.fovy_rad / 2.0)

        # HSV color threshold range for orange/red ball
        self.lower_hsv = np.array([5, 110, 80])
        self.upper_hsv = np.array([25, 255, 255])

        # Filter thresholds
        self.min_area = 15.0

    def detect(self, bgr_img) -> BallDetection:
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)

        # Morphological opening to suppress noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return BallDetection(visible=False)

        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < self.min_area:
            return BallDetection(visible=False)

        # Minimum enclosing circle for robust diameter estimation
        (cx_f, cy_f), radius = cv2.minEnclosingCircle(c)
        cx, cy = int(cx_f), int(cy_f)
        diameter_px = radius * 2.0

        if diameter_px <= 1.0:
            return BallDetection(visible=False)

        # Monocular depth: Z = (f * D) / d
        distance = (self.focal_length * BALL_DIAMETER_METERS) / diameter_px

        # Horizontal bearing: bearing = atan2(cx - cx_center, f)
        center_x = self.width / 2.0
        bearing = math.atan2(cx - center_x, self.focal_length)

        return BallDetection(
            cx=cx,
            cy=cy,
            radius=radius,
            area=area,
            distance=distance,
            bearing=bearing,
            visible=True,
            contour=c
        )
