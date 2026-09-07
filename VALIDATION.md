# Soccer contact-control validation

This document records the earlier strict-vision contact fix. The separate, subsequently added oracle navigation baseline is documented in [ORACLE_VALIDATION.md](ORACLE_VALIDATION.md).

Base commit: `5823816c05b2cf42e99d761670ccaf41342d2221`.

## Changes

- Simulation FSM deadlines use `d.time`, not wall time. Rendering stalls no longer shorten the physical blind advance or kick. Wall time is only used for display pacing.
- State transitions choose their new policy immediately. The kick trigger and first kick action now occur together; a 0.5-second kick occupies 25 control ticks. Deadline comparisons tolerate floating-point accumulation.
- Approach uses the existing metric-distance servo, conservatively slowing from 0.35 to 0.30 m/s. Large bearing errors stop forward advance, unexplained occlusions do not command forward motion, and the blind advance requires approximate bearing alignment.
- Terminal lateral command changes from 0.03 to 0.06 m/s to place the ball further toward the right-foot strike region. Terminal duration remains configurable, defaulting to 1.52 simulation seconds. This is a calibrated open-loop command, not measured odometry.
- Evaluation samples every physics substep. Only physical right-foot/ball contact during the kick policy sets `kick_contact`. Walking contacts are separately recorded as `foot_contact`; proximity is diagnostic only.
- Goal evaluation requires a forward crossing of the configured goal plane within its width and below the crossbar. This is a ball-center crossing metric, not a full official football rule implementation.
- Strict mode never feeds the evaluator's goal signal into the controller. Goal bearing remains telemetry; goal aiming is not implemented.
- Onboard execution shares the simulator's detectors and FSM, removing the undefined `ball_cy` path and divergent control logic. Camera loss stops movement. A kick request is not announced as a confirmed goal.
- Benchmark supports reproducible seeds and per-trial JSON output. ONNX sessions use one inference thread each and their own input/output names.

## Tests actually run

Environment: Linux CPU simulation, MuJoCo 3.12.0, OpenCV 5.0.0, ONNX Runtime 1.29.0, NumPy 2.3.5. Offscreen EGL rendering used software Mesa. GUI windows and hardware were not exercised.

### Regression checks

```bash
python -m unittest discover -s tests -v
python -m compileall -q benchmark.py sim_duck_soccer.py duck_soccer_onboard.py microduck_soccer
```

Nine regression tests passed: synchronous kick timing, terminal-stop timing, distance slowdown, large-angle advance suppression, occlusion handling, terminal alignment gate, rejection of proximity/velocity false positives, separation of walk and kick contacts, and goal-plane/height conditions. Some tests cover multiple conditions.

### Physical trials

Each episode ran for 12 simulation seconds. Initial yaw is uniform in ±15 degrees; ball x is 1.0–1.25 m and y is ±0.12 m. Trial i uses seed `--seed + i - 1`.

| Controller | Seeds | Actual kick contacts | Goals | Falls |
| --- | --- | --- | --- | --- |
| Original FSM with corrected contact measurement and reproducible harness | 0–4 | 3/5 | 0/5 | 0/5 |
| Final controller | 0–4 | 4/5 | 0/5 | 0/5 |
| Final controller, separate seed group | 5–9 | 3/5 | 0/5 | 0/5 |

Final per-seed kick contact: `true, true, true, true, false, true, false, false, true, true`.

The 7/10 aggregate is a small-sample observation in this scene, not a guaranteed success rate. The separate seed group does not have an original-controller baseline. These are contact counts, not proof of a strong shot or goal.

```bash
python benchmark.py --trials 5 --seed 0 --output results-0-4.json
python benchmark.py --trials 5 --seed 5 --output results-5-9.json
python sim_duck_soccer.py --mode strict --headless --duration 12
```

On Linux without a display, set `MUJOCO_GL=egl` and install a working EGL/Mesa driver first. `--headless` still renders RGB frames for vision.

The simulation entry point completed its 12-second strict run with physical kick contact and no goal. This verifies the actual entry point's offscreen path, not the desktop viewer.

An isolated standing-kick probe placed balls at x=0.04–0.14 m and y=-0.08–0.02 m relative to the settled trunk. Good forward strikes were observed near x=0.06–0.08 m, y=-0.04 m; several centerline placements missed. These placements were test fixtures only, never used to position the ball in strict control.

An exploratory larger approach-bearing offset was rejected after a previously approachable seed failed to enter terminal approach within 12 seconds. The final approach offset remains 0.08 rad.

## Remaining limits

Blind advance still depends on head pose, gait and floor properties and can miss. No reliable ball-location estimate is maintained inside the blind zone. The next substantial improvement needs camera/body calibration and a validated estimate of the ball relative to the kicking foot, rather than further blind timer tuning. Goal aiming and real-robot camera/RPC operation remain unverified.

Onboard deployment now requires copying `microduck_soccer/` alongside `duck_soccer_onboard.py`, as shown in both READMEs.
