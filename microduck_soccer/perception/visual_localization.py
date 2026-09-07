"""Camera + encoder/IMU localization; no simulator pose or contact inputs.

The independent kinematic model places its root at the origin. A supporting
foot supplies short-range odometry while the ball passes below the camera.
Only measured pixels locate the ball and goal. Dimensions/extrinsics are
calibration, not scene positions. Flat ground and limited foot slip assumed.
"""
import math
import mujoco
import numpy as np


class VisualLocalization:
    def __init__(self, model, width=320, height=240):
        self.model = model
        self.kinematics = mujoco.MjData(model)
        self.joints = [int(model.jnt_qposadr[model.actuator_trnid[i, 0]]) for i in range(14)]
        self.camera = model.camera('egocentric').id
        self.feet = [model.geom(n).id for n in ('left_foot_collision', 'right_foot_collision')]
        self.soles = []
        for geom in self.feet:
            mesh = model.geom_dataid[geom]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            self.soles.append(model.mesh_vert[start:start+count].copy())
        self.ball_geom = model.geom('ball_geom').id
        self.width, self.height = width, height
        self.focal = height / (2 * math.tan(math.radians(model.cam_fovy[self.camera]) / 2))
        self.xy = np.zeros(2)
        self.velocity = np.zeros(2)
        self.previous_feet = None
        self.previous_rotations = None
        self.previous_time = None
        self.ball_xy = None
        self.goal_xy = None
        self.ball_velocity = np.zeros(2)
        self.drift_velocity = np.zeros(2)
        self.ball_seen = -float('inf')
        self.goal_seen = -float('inf')
        self.yaw = 0.

    def update_sensors(self, now, joint_positions, orientation):
        kin = self.kinematics
        kin.qpos[:3] = 0.
        kin.qpos[3:7] = orientation
        kin.qpos[self.joints] = joint_positions
        mujoco.mj_kinematics(self.model, kin)
        mujoco.mj_camlight(self.model, kin)
        feet = kin.geom_xpos[self.feet].copy()
        rotations = kin.geom_xmat[self.feet].reshape(2, 3, 3)
        # The soles are meshes, not their bounding boxes. Use their actual
        # supporting vertices for height and rolling-contact odometry.
        corners = [vertices[np.argmin(vertices @ rotations[i, 2])]
                   for i, vertices in enumerate(self.soles)]
        bottoms = [feet[i, 2] + rotations[i, 2] @ corners[i] for i in range(2)]
        support = int(np.argmin(bottoms))
        dt = now - self.previous_time if self.previous_time is not None else 0.
        if self.previous_feet is not None and dt > 0:
            # Track the same sole contact corner through foot roll, rather
            # than treating the moving box center as a fixed ground point.
            corner = corners[support]
            previous = self.previous_feet[support] + self.previous_rotations[support] @ corner
            current = feet[support] + rotations[support] @ corner
            delta = previous[:2] - current[:2]
            # Flat-floor stride calibration: support switching overestimates
            # forward travel during the last short approach with this policy.
            w, x, y, z = orientation
            yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
            forward = np.array([math.cos(yaw), math.sin(yaw)])
            delta -= .12 * np.dot(delta, forward) * forward
            self.xy += delta
            if .12 < now-self.ball_seen < 3. and np.dot(delta / dt, forward) > .06:
                # While the ball was stationary and visible, its apparent
                # world drift measured short-term odometry bias. Carry that
                # correction only through the brief camera blind interval.
                self.xy -= self.drift_velocity * dt
            self.velocity = .8 * self.velocity + .2 * delta / dt
        self.previous_feet = feet
        self.previous_rotations = rotations.copy()
        self.previous_time = now
        self.height_estimate = -float(bottoms[support])
        w, x, y, z = orientation
        self.yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))

    def observe(self, ball, goal, now):
        kin = self.kinematics
        camera = kin.cam_xpos[self.camera].copy()
        camera[:2] += self.xy
        camera[2] += self.height_estimate
        rotation = kin.cam_xmat[self.camera].reshape(3, 3)
        if ball.visible and ball.cy + ball.radius < self.height - 2 and ball.cx - ball.radius > 1 and ball.cx + ball.radius < self.width - 2:
            # A near, off-axis sphere projects to an ellipse. Fit its cone
            # axis on unit pixel rays; a bounding-circle center biases depth.
            pixels = ball.contour.reshape(-1, 2).astype(float)
            rays = np.column_stack(((pixels[:, 0]-self.width/2)/self.focal,
                                    (self.height/2-pixels[:, 1])/self.focal,
                                    -np.ones(len(pixels))))
            rays /= np.linalg.norm(rays, axis=1, keepdims=True)
            _, _, vt = np.linalg.svd(rays-rays.mean(axis=0), full_matrices=False)
            axis = vt[-1]
            if axis[2] > 0:
                axis = -axis
            ray = rotation @ axis
            if ray[2] < -.03:
                scale = (.035-camera[2]) / ray[2]
                if scale > 0:
                    estimate = (camera + scale * ray)[:2]
                    dt = now-self.ball_seen
                    if self.ball_xy is not None and 0 < dt < .3:
                        self.ball_velocity = .7*self.ball_velocity + .3*(estimate-self.ball_xy)/dt
                    else:
                        self.ball_velocity[:] = 0.
                    self.ball_xy = estimate
                    self.ball_seen = now
                    if np.linalg.norm(estimate-self.xy) > .22 and np.linalg.norm(self.ball_velocity) < .07:
                        self.drift_velocity = .7*self.drift_velocity + .3*self.ball_velocity
        if goal.visible and goal.width > 8:
            depth = .8 * self.focal / goal.width
            point = camera + rotation @ np.array([(goal.cx-self.width/2)*depth/self.focal,
                                                  (self.height/2-goal.cy)*depth/self.focal, -depth])
            if self.goal_xy is None or np.linalg.norm(point[:2]-self.goal_xy) < .35:
                self.goal_xy = point[:2] if self.goal_xy is None else .9*self.goal_xy + .1*point[:2]
                self.goal_seen = now

    def foot_clearance(self):
        if self.ball_xy is None:
            return float('inf')
        # Collision query on the PRIVATE kinematic estimate only. No live
        # simulator data is passed here or mutated; this is predicted clearance.
        self.kinematics.geom_xpos[self.ball_geom] = [*(self.ball_xy-self.xy), .035-self.height_estimate]
        return min(mujoco.mj_geomDistance(self.model, self.kinematics, geom,
                                         self.ball_geom, .1, None) for geom in self.feet)

    def observe_near_ball(self, ball, now):
        """Fit a visible sphere arc during a stationary look-down inspection.

        Discard the artificial contour along image edges. A cropped bounding
        circle is NOT a valid range measurement. Require a substantial curved
        silhouette and a small cone-fit residual before accepting its position.
        """
        if not ball.visible or ball.contour is None:
            return None
        pixels = ball.contour.reshape(-1, 2).astype(float)
        pixels = pixels[(pixels[:, 0] > 2) & (pixels[:, 0] < self.width-3)
                        & (pixels[:, 1] > 2) & (pixels[:, 1] < self.height-3)]
        if len(pixels) < 12 or np.linalg.norm(np.ptp(pixels, axis=0)) < 25:
            return None
        rays = np.column_stack(((pixels[:,0]-self.width/2)/self.focal,
                                (self.height/2-pixels[:,1])/self.focal, -np.ones(len(pixels))))
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        camera = self.kinematics.cam_xpos[self.camera].copy()
        camera += [*self.xy, self.height_estimate]
        rotation = self.kinematics.cam_xmat[self.camera].reshape(3,3)
        # Beak occlusion introduces non-spherical edges. Keep only a coherent
        # curved arc whose angular radius agrees with the known 35 mm sphere.
        rng = np.random.default_rng(0)
        best = None
        for _ in range(64):
            a, b, c = rays[rng.choice(len(rays),3,replace=False)]
            axis = np.cross(b-a,c-a)
            norm = np.linalg.norm(axis)
            if norm < 1e-6:
                continue
            axis /= norm
            if axis[2] > 0:
                axis = -axis
            plane = float(np.dot(axis,a))
            direction = rotation @ axis
            if direction[2] >= -.1 or not 0 < plane < 1:
                continue
            distance = (.035-camera[2])/direction[2]
            radius = distance*np.sqrt(1-plane*plane)
            if not .028 < radius < .042:
                continue
            inliers = np.abs(rays @ axis-plane) < .0025
            if best is None or inliers.sum() > best.sum():
                best = inliers
        if best is None or best.sum() < max(12,len(rays)*.6):
            return None
        arc = rays[best]
        _, singular, vt = np.linalg.svd(arc-arc.mean(axis=0),full_matrices=False)
        if singular[1] < .015 or singular[2]/singular[1] > .08:
            return None
        axis = vt[-1]
        if axis[2] > 0:
            axis = -axis
        ray = rotation @ axis
        if ray[2] >= -.1:
            return None
        distance = (.035-camera[2]) / ray[2]
        if distance <= 0:
            return None
        estimate = (camera + distance * ray)[:2]
        if np.linalg.norm(estimate-self.xy) > .25:
            return None
        return estimate
