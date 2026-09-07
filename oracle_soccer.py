#!/usr/bin/env python3
"""Physics-only oracle baseline. Ground truth navigation, no ball teleportation.

Run from repository root. No renderer is constructed unless --viewer is used.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path
import random
import time
import mujoco
import numpy as np
from microduck_soccer.control.oracle import OracleSoccerController
from microduck_soccer.policy import PolicyRunner, DEFAULT_POSE
from microduck_soccer.evaluation import SoccerEvaluator, EpisodeMetrics

from microduck_soccer.assets import get_scene_xml_path, POLICIES_DIR

ROOT = Path(__file__).resolve().parent
XML = Path(get_scene_xml_path())
POLICIES = POLICIES_DIR


def run_trial(seed=0, duration=60., viewer=False, trace=False):
    rng = random.Random(seed)
    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)
    runner = PolicyRunner(*(str(POLICIES / p) for p in
                           ['alpha_walking.onnx', 'ball_kick_right.onnx',
                            'ball_kick_left.onnx', 'alpha_stand.onnx']))
    controller = OracleSoccerController()
    qi = [int(m.jnt_qposadr[m.actuator_trnid[i, 0]]) for i in range(14)]
    vi = [int(m.jnt_dofadr[m.actuator_trnid[i, 0]]) for i in range(14)]
    tid, bid = m.body('trunk_base').id, m.body('ball').id
    ball_joint = m.body_jntadr[bid]
    bq, bv = m.jnt_qposadr[ball_joint], m.jnt_dofadr[ball_joint]
    foot = m.site('right_foot').id
    foot_geoms = [m.geom(name).id for name in ('left_foot_collision','right_foot_collision')]
    ball_geom = m.geom('ball_geom').id
    evaluator = SoccerEvaluator(goal_x=2.835, goal_width=.69, foot_geom_id=m.geom('right_foot_collision').id,
                                ball_geom_id=m.geom('ball_geom').id, foot_site_id=foot)
    metrics = EpisodeMetrics(seed)
    yaw0 = rng.uniform(-math.pi / 12, math.pi / 12)
    d.qpos[qi] = DEFAULT_POSE
    d.qpos[2] = .12
    d.qpos[3:7] = [math.cos(yaw0 / 2), 0, 0, math.sin(yaw0 / 2)]
    initial_ball = [rng.uniform(1., 1.25), rng.uniform(-.12, .12)]
    # Episode initialization only. Neither robot nor ball state is set again.
    d.qpos[bq:bq+7] = [*initial_ball, .035, 1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    metrics.initial_ball_dist = float(np.linalg.norm(initial_ball))
    handle = None
    if viewer:
        from mujoco import viewer as mjviewer
        handle = mjviewer.launch_passive(m, d)
    last_state = None
    try:
        while d.time < duration and not metrics.goal_scored and not metrics.fallen:
            start = time.monotonic()
            quat = d.xquat[tid]
            yaw = math.atan2(2*(quat[0]*quat[3]+quat[1]*quat[2]), 1-2*(quat[2]**2+quat[3]**2))
            gyro = d.sensor('imu_ang_vel').data.astype(np.float32)
            state, vx, vy, vyaw, mode, trigger = controller.update(
                float(d.time), d.xpos[tid,:2].copy(), yaw, d.qvel[:2].copy(),
                float(np.linalg.norm(gyro)), d.xpos[bid,:2].copy(), d.qvel[bv:bv+2].copy(),
                np.array([2.8, 0.]), foot_clearance=min(
                    mujoco.mj_geomDistance(m,d,g,ball_geom,.05,None) for g in foot_geoms))
            if trace and (state != last_state or round(d.time / .02) % 50 == 0):
                rotation = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])
                offset = rotation @ (d.xpos[bid,:2]-d.xpos[tid,:2])
                print(f't={d.time:.2f} {state} ball_local={offset.round(3)} yaw={yaw:.3f} cmd=({vx:.3f},{vy:.3f},{vyaw:.3f})', flush=True)
            last_state = state
            if trigger:
                metrics.approach_success = True
            if trigger and metrics.time_to_kick == 0:
                metrics.time_to_kick = float(d.time)
            command = np.zeros(13, dtype=np.float32)
            command[:3] = [vx, vy, vyaw]
            d.ctrl[:14] = runner.step(mode, gyro, quat.astype(np.float32),
                                      d.qpos[qi].astype(np.float32), d.qvel[vi].astype(np.float32), command)
            for _ in range(10):
                mujoco.mj_step(m, d)
                evaluator.evaluate_step(d, tid, bid, foot, metrics, float(d.time), active_mode=mode)
            if handle:
                if not handle.is_running():
                    break
                handle.sync()
                time.sleep(max(0., .02-(time.monotonic()-start)))
        report = dict(vars(metrics))
        report["goal_with_kick_contact"] = metrics.goal_scored and metrics.kick_contact
        report.pop('ball_detected')  # No camera is used in oracle mode.
        return dict(report, seed=seed, kicks=controller.kicks,
                    final_ball_xy=d.xpos[bid,:2].tolist(), state=controller.state)
    finally:
        if handle:
            handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--duration', type=float, default=60.)
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.trials <= 0 or args.duration <= 0:
        parser.error('trials and duration must be positive')
    if args.workers < 1 or (args.viewer and args.workers != 1):
        parser.error('workers must be positive; viewer requires workers=1')
    results=[]
    def save_result(result):
        results.append(result)
        results.sort(key=lambda r: r['seed'])
        print(json.dumps(result),flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({'mode':'oracle','duration':args.duration,
                                               'requested_trials':args.trials,
                                               'trials':results},indent=2),encoding='utf-8')
    if args.workers == 1:
        for i in range(args.trials):
            save_result(run_trial(args.seed+i,args.duration,args.viewer,args.trace))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = [pool.submit(run_trial,args.seed+i,args.duration,False,args.trace)
                    for i in range(args.trials)]
            for job in as_completed(jobs):
                save_result(job.result())
    print(f'Goals {sum(r["goal_scored"] for r in results)}/{len(results)}; '
          f'goals with kick contact {sum(r["goal_with_kick_contact"] for r in results)}/{len(results)}; '
          f'contact {sum(r["kick_contact"] for r in results)}/{len(results)}; '
          f'falls {sum(r["fallen"] for r in results)}/{len(results)}')
if __name__=='__main__':
    main()
