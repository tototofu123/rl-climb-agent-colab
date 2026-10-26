# PLAN

## One-line Goal
Train a 2D climbing agent that moves on a wall toward a target using static moves and dynos while respecting human-like constraints — and learn RL by building it.

## Approach

Build from the ground up in 6 stages, each adding one layer of complexity. Never move to the next stage until the current one produces visible results.

1. **Env scaffolding** — Gymnasium shell, Pygame render, random agent
2. **Physics** — Box2D body, reach constraints, gravity, limb mechanics
3. **First training** — PPO (SB3), TensorBoard, model checkpoints
4. **Reward tuning** — Systematic weight sweeps, comparison
5. **Dyno mechanics** — Jump action, momentum, dynamic reaching
6. **Generalization** — Multiple walls, curriculum learning, domain randomization

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| RL framework | Gymnasium | Industry standard env API |
| RL algorithm | PPO (Stable-Baselines3) | Stable, well-tuned defaults, good for discrete control |
| Physics | Box2D (PyBox2D) | Lightweight, contact detection, familiar API |
| Rendering | Pygame | Simple 2D graphics, fast iteration |
| Logging | TensorBoard | Training curves, reward components |
| Math | NumPy | Standard |
| Config | YAML (PyYAML) | Readable experiment configs |
| Hyperparam tuning | Optuna (stretch) | Systematic search |

## Stages

See `docs/LEARNING_ROADMAP.md` for full details.

| Stage | What you build | RL concept |
|-------|---------------|------------|
| 0 | Env shell + random agent | MDP, step loop, obs/action spaces |
| 1 | Body model + constraints | State space design, constraint satisfaction |
| 2 | PPO training loop | PPO, policy gradient, parallel envs |
| 3 | Reward engineering | Shaping, sparse vs dense, reward hacking |
| 4 | Dyno mechanics | Exploration, credit assignment |
| 5 | Multi-wall generalization | Overfitting, curriculum, domain randomization |

## Architecture

See `docs/ARCHITECTURE.md` for full detail.

```
physics/  →  env/  →  training/  →  agents/
  Box2D      Gymnasium    SB3 PPO      .zip files
  body       obs/action   VecEnv       checkpoints
  holds      rewards      callbacks
  gravity    rendering    TensorBoard
```

## Deliverables

1. `env/climbing_env.py` — Gymnasium environment
2. `physics/body.py` — Box2D climber body
3. `training/train.py` — PPO training pipeline
4. `agents/best_model.zip` — trained model
5. `render/replays/` — replay videos
6. `experiments/results/` — training curves + configs
7. `docs/` — all planning and architecture
