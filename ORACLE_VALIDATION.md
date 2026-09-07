# Oracle soccer baseline validation

This baseline diagnoses navigation, foot placement and physical kicking using
MuJoCo ground truth. It is separate from `sim_duck_soccer.py --mode strict`.
It is not a vision-only success claim or a hardware validation.

## What changed

- Navigate to a position behind the ball relative to the goal.
- Use measured minimum active walking commands: the bundled walking policy
  barely moves for small proportional commands. Stop in a position deadband.
- Watch physical foot/ball clearance to stop an approaching foot before an
  unintended walking collision where possible.
- Stand for at least 0.6 seconds before checking kick readiness; compensate
  residual translation and yaw from the gait-to-stand transition.
- Require the ball to be in the right-foot strike region, sufficiently still,
  with low robot speed and angular speed before triggering the 0.5-second kick.
- Compensate the existing kick policy's lateral shot bias using a measured
  contact-position table. Wait for the ball after kicking, then retry if needed.
- Disable ONNX Runtime platform telemetry before creating inference sessions.

The controller returns only velocity commands and policy selections. The runner
sets initial `qpos` before the episode; during the episode it only sets joint
targets and advances physics. No ball teleportation, forces or velocity injection
are used. Scene XML, friction and policy weights are unchanged.

## Calibration and development

Isolated standing-kick fixtures tested ball positions x = 0.075/0.085/0.095 m and
y = -0.065/-0.055/-0.045/-0.035/-0.025 m relative to the settled trunk.
All 15 placements were clear of initial foot overlap. The x = 0.085,
y = -0.055 m placement produced a measured outgoing heading of -0.321 rad
relative to the trunk. Adjacent lateral placements changed the angle strongly;
the controller therefore checks an interpolated shot angle after settling.
Those fixture placements are calibration only, not episode control actions.

Three-second walking probes after standing showed that commands vx = 0.15 or
vy = 0.15 produced only millimetres of displacement, while vy = 0.30 produced
about 0.14 m and vyaw = 1.2 produced about 1.55 rad. These measurements motivated
the minimum active commands; they do not establish hardware behaviour.

Development seeds 0–9 produced 9 goals, 8 goals with kick contact and no falls
in the final tuning batch. They are excluded from the following validation set.
Parameters were frozen before testing seeds 100–199.

## Independent validation

Linux CPU, Python 3.12, MuJoCo 3.12.0, ONNX Runtime 1.29.0, NumPy 2.3.5.
No rendering. Four workers, each ONNX session with one inference thread.
Each trial starts at robot xy = (0, 0), yaw uniform in +/-15 degrees;
ball x uniform in [1.0, 1.25] m and y in [-0.12, 0.12] m. Each seed has its own
Python `random.Random` generator. Episodes end on goal, fall or 40 simulated seconds.

| Measure | Result |
| --- | ---: |
| Completed independent trials | 100 |
| Goals | 98 |
| Goals with a recorded kick contact | 96 |
| Trials with kick contact | 97 |
| Goals without kick contact (walking pushes) | 2 |
| Falls by trunk-height criterion | 0 |
| Mean time of the 98 successful episodes | 12.34 s |

Raw results: [validation/oracle-seeds-100-199.json](validation/oracle-seeds-100-199.json).
Seeds 100–182 completed before automatic approval review stopped execution due
to an unexpected telemetry connection. Seeds 183–199 completed after adding
the supported `onnxruntime.disable_telemetry_events()` call; navigation parameters
and physics were unchanged. Every seed appears exactly once in the combined data.

`kick_contact` requires an actual right-foot/ball contact during the kick policy,
sampled each physics substep. `goal_with_kick_contact` is the conjunction of that
episode flag and goal success, not proof that a later walking contact contributed
nothing to the goal. Push-only goals occurred on seeds 162 and 197.

The oracle evaluator requires a forward crossing at ball-centre x = 2.835 m,
past the nominal x = 2.8 m goal plane by the 0.035 m ball radius, within
|y| < 0.345 m. It retains the existing centre-height condition 0 <= z < 0.35 m.
This is a conservative horizontal clearance metric, not complete football rules.
A fall means trunk z < 0.06 m; zero such events does not prove general stability.

Failed seed 106 ended at ball xy = (2.8257, 0.1854) m without a kick: the ball
had not crossed the configured full-ball goal threshold by the time limit.
Failed seed 138 had one kick but ended at (0.9319, 2.1358) m, outside the goal.
These failures were retained and no validation seeds were used to retune.

## Reproduce

```bash
pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 python oracle_soccer.py --seed 100 --trials 100 --duration 40 --workers 4 --output oracle-results.json
python -m unittest discover -s tests -v
python oracle_soccer.py --viewer --seed 100
```

All 13 regression tests passed: nine earlier visual-control/evaluation tests and
four oracle geometry, kick gating, single-trigger and stop tests. The GUI viewer
path and hardware were not exercised. Batch trials do not construct a renderer.

The 98/100 result applies only to this scene, bundled policies, dependency
versions and limited spawn range. There are no opponents or obstacles. The
existing scene lets the ball roll long distances; friction was not changed.
Wider spawn ranges, slopes, moving balls and visual localization remain untested.
The next visual-control step is to estimate the same foot-relative ball position
and goal direction from calibrated cameras and odometry, then validate separately.
