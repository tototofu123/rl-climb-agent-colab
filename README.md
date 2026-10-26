# RL Climb Agent

A reinforcement learning climbing agent. Stages 0–5: a 2D stick-figure
climber with physics, dynos and curriculum learning. **Stage 6/7 (new): a 3D,
anthropometric climber on the real HKUST boulder wall** — holds have real
positions, protrusion depth, pull direction and strength, and the wall is a
true 3D structure (main wall + two perpendicular side walls, all vertical /
0° overhang). Trained with PPO via Stable-Baselines3.

## 3D mode (Stage 6/7) — real wall, depth-aware

```
wall photo + annotations (HKUST climbing app)
   → metric 3D holds (x, y, z-depth, pull axis, grip cone, strength)
   → 3D anthropometric climber (lean, 2-bone limb IK, mass distribution)
   → grip physics: force/torque equilibrium vs per-hold pull cones
   → PPO learns to climb generated boulder problems
```

- **Real wall data**: 417 annotated holds imported from the climbing app
  (`data/hkust/`); depths vision-calibrated against the gym photos.
- **Not a stickman**: pelvis/torso/head, 2-bone arms and legs with IK,
  per-segment masses, torso lean dynamics, 3D pendulum swing.
- **3D angles matter**: hanging from an up-facing hold slips; wide stances
  push foot forces out of their cones; feet on good holds unload the hands.
- **Depth matters**: hold z-position = protrusion (crimp 2.5 cm → volume
  18 cm+); grip cones tilt out of the wall by type.
- **Synthetic routesetter**: generates boulder problems across the real
  holds (easy/medium/hard), task-randomization wrapper for generalization.

See `docs/3D_DESIGN.md` for the full architecture.

## Quick Start

```bash
pip install -r requirements.txt

# --- 3D mode (real HKUST wall) ---
python3.12 run_stage6.py                                # random agent demo
python3.12 training/train_3d.py --difficulty easy --timesteps 200000
python3.12 training/train_3d.py --task-random           # pool of problems
# long route on the tall toprope wall (length curriculum, optional ceiling)
python3.12 training/train_3d.py --wall toprope --long-route --curriculum --top-y 5.0
python3.12 training/eval_long_route.py --model agents/ppo3d_toprope_long_easy_*.zip
python3.12 render/make_demo_video.py --model agents/ppo3d_toprope_long_easy_*.zip --wall toprope --long-route --top-y 5.0 --include-fall
python3.12 run_stage6.py --model agents/ppo3d_easy_*.zip  # replay trained

# --- classic 2D mode (stages 0-5) ---
python3.12 run_stage0.py
python3.12 training/train.py                           # PPO, 2D walls
python3.12 training/train_stage5.py --curriculum --domain-rand
python3.12 run_replay.py
```

## Stack

- Python 3.12, Gymnasium, Stable-Baselines3, Pygame, NumPy

## Repo Structure

```
data/hkust/          Real wall: annotations, grid, photos, depth priors
wallimport/          importer.py (pixels → 3D metric holds), routeset.py
physics/             body.py (2D), body3d.py (3D anthropometric climber)
env/                 climbing_env.py (2D), climbing_env_3d.py (3D, real wall)
                     wrappers.py (2D curriculum), wrappers3d.py (task pool)
render/              renderer.py (2D), renderer3d.py (perspective 3D view)
training/            train.py (2D), train_stage5.py, train_3d.py
experiments/         reward configs, logs
agents/              trained models
tests/               41 tests for 3D stages + 2D stage tests
docs/                STATUS.md, 3D_DESIGN.md, design notes
```

## Tests

```bash
python3.12 -m pytest tests/test_stage6.py tests/test_stage7.py -q   # 3D: 41
python3.12 -m pytest tests/ -q                                      # all
```

## Status

- Stages 0–5 (2D): complete — see `docs/STATUS.md`.
- Stage 6/7 (3D real wall): complete. PPO climbs a real HKUST boulder problem:
  **7/20 top-outs** (3.4 m), mean reward +35.1 vs -5.3 random, using dynos
  and grip-feasible moves. A first training run exposed (and we fixed) a
  dyno-bouncing reward hack. Details in `docs/STATUS.md` and
  `docs/3D_DESIGN.md`.

## Training Results (3D, real wall)

| Policy | Mean reward | Max height (avg/best) | Tops |
|--------|------------|----------------------|------|
| random | -5.3 | 1.31 / 1.63 m | 0/20 |
| PPO 400k (easy problem) | +35.1 | 3.38 / 3.48 m | 7/20 |
| PPO 600k long route, top rope 9.25 m wall (v4) | -4.3 (det) / -12.8 (sampled) | 9.25 / 9.25 m — overshoots the top hold without catching it | 0/10 |
| random baseline, same long route | -12.0 | 3.19 / 4.53 m | 0/10 |

Long-route eval (`training/eval_long_route.py`) applies the saved
VecNormalize stats — **omitting them starves the policy and makes it hover**
(that was an eval bug, not a policy one).
