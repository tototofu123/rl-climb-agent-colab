# 3D Upgrade — Real HKUST Wall, Depth-Aware Climbing (Stage 6/7)

_Date: 2026-10-24_

## Goal

Replace the 2D stickman with a 3D, anthropometric climber on the **real HKUST
boulder wall**, where holds have depth, orientation and strength — so the
"3D angles" of climbing (side-pulls, underclings, lean-out balance, corner
terrain) actually determine what the agent can do.

Wall geometry facts (confirmed by vision analysis of the gym photos in the
climbing app):

- Every wall section is **vertical (0° overhang)**.
- The boulder is a 3D structure: a ~5.05 m × 4.58 m main wall plus two
  perpendicular **side walls** (inside corners) ~0.9 m / ~0.45 m deep.
- Holds protrude from the wall plane by 2–25 cm depending on type (volumes
  deepest, crimps shallowest) and open outward/upward.

## Data pipeline

```
hkust-climbing-app/assets          (read-only source, not modified)
  HOLD_SPEC_v12.json               417 holds: type, difficulty, direction, boxSize
  boulder_grid_config.json         grid cells -> wall sections (cols 0/5 = side walls)
  route_names_en.json
        │ copy
        ▼
data/hkust/boulder/*               + data/hkust/photos/* + hold_depth_priors.json
        │ wallimport/importer.py
        ▼
GymWall3D / Hold3D                 metric 3D holds: pos, normal, depth, radius,
                                  pull_axis, cone_cos, f_max_hand/foot
```

- **Coordinates**: x along wall, y up, z out of the wall toward the climber.
- **Scale**: 300 px/m (one grid panel ≈ 360 px = 1.2 m).
- **Side walls**: panorama columns 0 and 5 fold toward the climber; their
  holds protrude ±x. A few giant volume annotations have centres outside the
  panorama — clamped onto the wall.
- **Pull axis**: the app's `direction` field (compass, 180 = down) is mapped
  into each wall section's plane, then tilted out of the wall by a per-type
  opening angle. Depth/cone/strength priors live in
  `data/hkust/hold_depth_priors.json`, calibrated with a vision model against
  the front and side wall photos (see `_provenance` there).

## Routes

The app's `cat`/`num` is a **hold-set grouping, not a route** (routes live in
Supabase), so `wallimport/routeset.py` generates boulder problems the way a
routesetter would: start low, chain holds one move apart (0.3–0.85 m up,
bounded lateral drift), biasing hold choice by difficulty preset
(easy = jugs, hard = crimps/slopers). Target = top hold of the chain.
`TaskRandomizationWrapper` samples a new problem each episode.

## 3D climber (`physics/body3d.py`)

- Anthropometric body: pelvis, torso (lean_out + lean_side orientation),
  head, 2-bone arms/legs with analytic IK (elbows bow down/out, knees away
  from the wall), per-segment masses.
- Composite CoM integrated under gravity with per-limb 3D distance
  constraints → the body swings as a **3D pendulum** (lateral and
  out-of-plane).
- Torso orientation follows a damped spring toward the anchor-implied pose
  (hang from hands / stand on feet) → visible lean dynamics.
- **Grip feasibility** (per env step): solve the minimum-norm contact forces
  satisfying force + torque equilibrium, then check each limb's force against
  its hold's grip cone (pull_axis ± cone half-angle) and strength limit.
  Violations slip the limb. This is where 3D angles matter:
  - hanging from an up-facing hold is impossible (you'd have to pull up),
  - wide stances rotate foot forces out of their cones → slips,
  - feet on good holds reduce hand load (verified in tests).

## Environment (`env/climbing_env_3d.py`)

- Actions `Discrete(22)`: noop | grab slot 1–16 | release limb ×4 | dyno.
  Slot = fixed task hold (chain sorted bottom-up); grabbing auto-picks the
  limb by role (high hold → hand, low → foot).
- Observations (241-d): CoM pos/vel, torso vector, per-limb state + grip
  utilisation, 13 features per hold slot (relative position, depth, pull
  axis, cone, strengths, target/held/grabbable flags), target vector+dist.
- Rewards: progress (3D distance), hold bonus, stability bonus (grip
  margin), slip penalty, reach penalty, fall penalty, success bonus, dyno
  bonuses.
- Success: a hand parked on the target hold for 2 steps with ≥2 limbs
  attached. Fall: CoM below 0.3 m (the mat).
- Start: compact, grip-feasible stance auto-picked from the bottom of the
  chain (tries stances until `check_grips` passes).

## Renderer (`render/renderer3d.py`)

Perspective camera offset to the front-right, painter-sorted drawing, depth
shading; wall sections, mat, all 417 holds (task holds ring-highlighted,
target magenta), 3D climber skeleton with joints and lean indicator, HUD.
Headless PNG mode for CI (`render_frame_png`) — render output was verified
with a vision model during development.

## Usage

```bash
# random agent demo (window)
python3.12 run_stage6.py

# train on one easy problem
python3.12 training/train_3d.py --difficulty easy --route-seed 0 --timesteps 200000

# train on a random pool of problems (generalization)
python3.12 training/train_3d.py --task-random --pool-size 8

# replay a trained model
python3.12 run_stage6.py --model agents/ppo3d_easy_*.zip
```

## Verified by tests

`tests/test_stage6.py` (19) + `tests/test_stage7.py` (14): importer geometry
and depth ordering, pull-axis orientation, routesetter chaining, IK bone
lengths, hanging equilibrium, impossible-orientation slips, feet reduce hand
load, 3D out-of-plane swing, dyno flight, env API/spaces/reset determinism,
success/fall termination, renderer PNG, PPO smoke.
