# Status

_Last updated: 2026-10-24_

---

## Stage 6/7 — 3D Real-Wall Climbing ✅

**Goal:** replace the 2D stickman with a 3D anthropometric climber on the
real HKUST boulder wall, with hold depth/orientation-aware grip physics.
See `docs/3D_DESIGN.md`.

| Item | Status |
|------|--------|
| `data/hkust/` — real annotations, grid, photos, depth priors copied in | ✅ |
| Vision analysis of wall photos (front + side) → depth calibration | ✅ |
| `wallimport/importer.py` — 417 holds → metric 3D (main + 2 side walls, 0° overhang) | ✅ |
| `wallimport/routeset.py` — synthetic boulder problems (easy/med/hard) | ✅ |
| `physics/body3d.py` — 3D anthropometric climber (lean, IK, masses) | ✅ |
| Grip feasibility: force/torque equilibrium vs per-hold pull cones | ✅ |
| `env/climbing_env_3d.py` — Gymnasium env on real wall | ✅ |
| `env/wrappers3d.py` — task-pool randomization | ✅ |
| `render/renderer3d.py` — perspective 3D renderer (headless PNG too) | ✅ |
| `training/train_3d.py` — PPO pipeline | ✅ |
| `run_stage6.py` — demo/replay entry | ✅ |
| `tests/test_stage6.py` (19) + `tests/test_stage7.py` (14) | ✅ all pass |
| Env speed: 75 → 487 steps/s after joint-frame optimization | ✅ |

### Training (easy problem, seed 0, HKUST boulder wall)

**v1 (200k steps)** hit a classic reward hack: dyno attempt/catch bonuses let
it "dyno-bounce" — 126 dynos/episode, 74.6/75.6 reward from bouncing, never
matching the target. Fixed: catch bonus only for height-gaining dynos, no
attempt bonus, + target-touch shaping.

**v2 (400k steps, clean reward)** — real climbing:

| Policy | Mean reward | Max height (avg/best) | Tops |
|--------|------------|----------------------|------|
| random | -5.3 | 1.31 / 1.63 m | 0/20 |
| PPO v2 (sampled) | **+35.1** | **3.38 / 3.48 m** | **7/20** |
| PPO v2 (deterministic) | +48.5 | 3.35 m | reaches top, times out before match |

The topping episode climbs 0 → 3.36 m in 89 steps using real moves plus
dynos; frames in `experiments/frames_v2/` (vision-verified posture, lean and
depth cueing). Logs: `experiments/train3d_easy_v2.log`.

### Long route (toprope wall, 9.25 m) — in progress

Goal: one honest long climb on the real tall wall (`--wall toprope
--long-route --curriculum`, length curriculum over a generated ~26-hold
chain). Obstacles found and fixed on the way:

- **Eval must apply the VecNormalize stats** (`--model`'s sibling
  `*_stats.pkl`). Predictions on raw observations starve the policy and make
  it hover at ~4 m — an eval bug, not a policy bug. `training/eval_long_route.py`
  loads the stats and reports random on the identical task for reference.
- **v4 (600k steps, full 8.74 m route)** learns prefixes well in training
  (reward -8 → +89, ep_len 330) but at eval the policy *overshoots* the route:
  the CoM reaches 9.25 m while **no hand is ever parked on the target hold**
  (0/10 tops, dist ≈ 0.5 m at the top). It free-climbs past the top hold via
  reach/dyno actions instead of finishing. Full-route policy therefore
  trapped in "move to height, don't grab the last hold".
- **Fallback (handoff protocol)**: first train an honest shorter full route
  (`generate_long_route(..., top_y=5.0)` → 14 holds, 1.13 → 5.24 m), then
  extend upward. `train_3d.py --top-y` was added for this; training runs
  `experiments/train3d_mid5*.log`.

| Policy | Mean reward | Max height (avg/best) | Tops |
|--------|------------|----------------------|------|
| random (full route) | -12.0 | 3.19 / 4.53 m | 0/10 |
| PPO v4 600k (full route, det) | -4.3 | 9.25 / 9.25 m | 0/10 |

---

## Stage 0 — Env Scaffolding

**Goal:** Gymnasium env shell, Pygame render, random agent running in a window. ✅

| Item | Status |
|------|--------|
| `env/__init__.py` | ✅ |
| `env/wall.py` — Wall + Hold dataclasses | ✅ |
| `env/climbing_env.py` — `SimpleClimbingEnv` | ✅ |
| `render/__init__.py` | ✅ |
| `render/renderer.py` — Pygame renderer | ✅ |
| `run_stage0.py` — entry point | ✅ |
| `tests/test_stage0.py` — 22 tests | ✅ |

**Achieved:** Agent moves 4-dir on wall, reaches target, falls off bottom. Gymnasium API scaffolding complete.

---

## Stage 1 — Physics + Constraints ✅

| Item | Status |
|------|--------|
| `physics/__init__.py` | ✅ |
| `physics/body.py` — `ClimberBody` + `LimbState` | ✅ |
| `physics/constraints.py` — reach/distance helpers | ✅ |
| `env/climbing_env.py` — `PhysicsClimbingEnv` | ✅ |
| `run_stage1.py` — entry point | ✅ |
| `tests/test_stage1.py` — 35 tests | ✅ |

**Achieved:** 4-limb climber body with gravity (300px/s²). Reach constraint enforcement via position/velocity projection. Actions: REACH_HOLD / RELEASE_LIMB. Body hangs, swings, and falls.

---

## Stage 2 — First Training Loop ✅

| Item | Status |
|------|--------|
| `training/__init__.py` | ✅ |
| `training/train.py` — PPO pipeline | ✅ |
| `tests/test_stage2.py` — 9 tests | ✅ |

**Achieved:** SB3 PPO with `VecNormalize`, TensorBoard logging, model checkpointing. 50k-step training shows reward trending -213 → -147, explained variance 0.79. CLI: `--reward-config` support.

---

## Stage 3 — Reward Engineering ✅

| Item | Status |
|------|--------|
| `experiments/configs.py` — 5 reward configs | ✅ |
| Configurable reward weights in env | ✅ |
| Reward components in `info["comp"]` dict | ✅ |
| `run_replay.py` — trained model viewer | ✅ |
| `tests/test_stage3.py` — 15 tests | ✅ |

**Achieved:** 5 reward profiles (dense, sparse, hybrid, speed, safe). Env computes weighted reward sum. Training pipeline accepts `--reward-config` flag. Replay script loads saved model and shows live Pygame visualization.

### Config comparison

| Config | Progress | Hold | Time | Reach | Fall | Success | Character |
|--------|----------|------|------|-------|------|---------|-----------|
| dense | 2.0 | 0 | 0 | 0 | -1.0 | 5.0 | Pure distance shaping |
| sparse | 0 | 0 | 0 | 0 | -1.0 | 5.0 | Only success/fall |
| hybrid | 1.0 | 0.2 | -0.01 | -0.3 | -1.0 | 5.0 | Balanced |
| speed | 0.5 | 0.1 | -0.05 | -0.3 | -2.0 | 10.0 | Prioritizes speed |
| safe | 2.0 | 0.3 | -0.005 | -0.5 | -3.0 | 3.0 | Avoids risky moves |
| dyno | 1.0 | 0.3 | -0.01 | -0.3 | -1.0 | 8.0 | For dyno-gap walls |

---

## Stage 4 — Dyno Mechanics ✅

| Item | Status |
|------|--------|
| Dyno impulse in ClimberBody (release-all + upward vel) | ✅ |
| DYNO action (n_holds+5) in action space | ✅ |
| Extended reach window (max_reach_dyno=250, 4 env steps) | ✅ |
| dyno_gap wall layout (160-170px gaps, requires dyno) | ✅ |
| Visual feedback: orange body/limbs, "DYNO!" text | ✅ |
| Dyno bonus reward (+0.1 attempt, +0.5 catch) | ✅ |
| `run_stage4.py` — random agent on dyno_gap wall | ✅ |
| `tests/test_stage4.py` — 18 tests | ✅ |
| **Verify:** dyno releases limbs, applies impulse, expires | ✅ |
| **Verify:** extended reach during dyno window | ✅ |
| **Verify:** 80k-step PPO: reward -5740 → +331 on dyno_gap wall | ✅ |
| **Verify:** 50k-step PPO on beginner wall: 5/5 SUCCESS, reward +373.9 | ✅ |

---

## Stage 5 — Generalization ✅

| Item | Status |
|------|--------|
| 3+ wall layouts (beginner, intermediate, dyno_gap, expert) | ✅ |
| Curriculum learning (easy → medium → hard → expert) | ✅ |
| Domain randomization (hold position variance) | ✅ |
| Ablation: beginner-trained model → 10/10 beginner, 0/10 others | ✅ |
| `env/wrappers.py` — CurriculumWrapper + DomainRandomizationWrapper | ✅ |
| `training/train_stage5.py` — --curriculum, --domain-rand flags | ✅ |
| `tests/test_stage5.py` — 22 tests | ✅ |

### Generalization Results

| Model trained on | beginner | intermediate | dyno_gap | expert |
|-----------------|----------|--------------|----------|--------|
| beginner (no rand) | 10/10 ✅ | 0/10 ❌ | 0/10 ❌ | 0/10 ❌ |

**Conclusion:** Agent overfits to training wall. Curriculum + domain randomization needed for true generalization.

---

## Final Deliverables

| Item | Status |
|------|--------|
| Best model → `agents/` | 🟡 trained (50k hybrid + 80k dyno) |
| Replay video → `render/replays/` | ⬜ |
| Training curves → `experiments/results/` | ⬜ |
| One-page summary of learnings | ⬜ |

## Test Suite

| Stage | Tests | Status |
|-------|-------|--------|
| Stage 0 | 22 | ✅ all pass |
| Stage 1 | 35 | ✅ all pass |
| Stage 2 | 9 | ✅ all pass |
| Stage 3 | 15 | ✅ all pass |
| Stage 4 | 18 | ✅ all pass |
| Stage 5 | 22 | ✅ all pass |
| Replay | 1 | ✅ all pass |
| **Total** | **126** | **125 passed, 1 skipped** |

## Quick Start

```bash
# Random agent (no learning)
python3.12 run_stage0.py
python3.12 run_stage1.py

# Train PPO (default: 200k steps, hybrid reward)
python3.12 training/train.py

# Train with custom config
python3.12 training/train.py --timesteps 100000 --reward-config speed --wall intermediate

# Watch trained model (auto-detects wall from model filename)
python3.12 run_replay.py

# Watch specific model on specific wall
python3.12 run_replay.py --model agents/ppo_beginner_hybrid_1785589130_final.zip --wall beginner
```
