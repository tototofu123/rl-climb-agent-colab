# Architecture

## System Layers

```
┌─────────────────────────────────────┐
│         render/ (Pygame)            │  ← Visualization layer
├─────────────────────────────────────┤
│         training/ (SB3 PPO)         │  ← Training loop
├─────────────────────────────────────┤
│    env/ (Gymnasium Env)             │  ← RL interface
├─────────────────────────────────────┤
│    physics/ (Box2D + constraints)   │  ← Simulation
└─────────────────────────────────────┘
```

## Data Flow (one step)

```
Action (discrete: reach-hold-3 / release-1 / shift-weight / jump)
       │
       ▼
env/step(action)
       │
       ▼
physics/  → apply forces, check contacts, enforce constraints
       │
       ▼
env/      → compute reward, check done, build obs
       │
       ▼
Observation (dict/Box):
  - body_pos, body_vel
  - limb_anchors (4 x,y) + limb_states (4 binary)
  - hold_positions (n x 2)
  - target_pos
  - distance_to_target
       │
       ▼
training/ → PPO policy → next action
```

## Body Model

```
         ○  ← center of mass (CoM)
        /|\
       / | \     ← limbs (4: left arm, right arm, left leg, right leg)
      /  |  \
    ◇   ◇   ◇  ← hold contacts (attached/detached)
```

- CoM: position (x,y), velocity (vx,vy), gravity-affected
- 4 limbs: each has max reach (static), anchor point when attached
- States: {attached, detached, reaching, jumping}
- Dyno: all 4 limbs release + apply impulse to CoM

## State Space (Observation)

```
Body:      pos (2), vel (2)                          = 4
Limbs:     anchor_pos (4 × 2), state (4 × 1)         = 12
Holds:     pos (n × 2), reachable_mask (n × 1)       = 3n
Target:    pos (2), dist_to_target (1)                = 3
Total:     19 + 3n   (n = number of holds, typ 5-10)
```

## Action Space (Discrete)

| Action | Description |
|--------|-------------|
| 0      | Do nothing |
| 1..n   | Reach hold i (with free limb) |
| n+1..n+4 | Release limb i |
| n+5    | Shift weight (adjust CoM slightly) |
| n+6    | Dyno (jump: release all + impulse) |

Size: n + 7 (for n holds, grows with wall complexity)

## Reward Function

```
r_step =  α * progress_bonus
        + β * hold_bonus
        + γ * time_penalty
        + δ * reach_penalty
        + ε * fall_penalty
        + ζ * success_bonus

α, β, γ, δ, ε, ζ  = tunable weights
```

- progress_bonus: Δ(-distance_to_target) — how much closer agent got
- hold_bonus: +small for each new hold secured
- time_penalty: -small per step (encourages efficiency)
- reach_penalty: -medium for impossible reach (out of range)
- fall_penalty: -large for falling
- success_bonus: +large for reaching target

## Training Pipeline

```
Config (YAML) ──► VecEnv (4-8 parallel) ──► PPO (SB3) ──► Model
                     │                                     │
                     ▼                                     ▼
               TensorBoard                            agents/*.zip
             (reward, loss,                     (checkpoints, best)
              explained_var)
```

## Key Design Decisions

1. **Discrete actions first** — easier to learn, then continuous as stretch
2. **Box2D** — lightweight, built-in contact/collision, familiar
3. **Dense reward** — sparse rewards rarely work for complex motor tasks; shape carefully
4. **PPO** — stable, well-tuned defaults in SB3, good for this action space
5. **VecEnv** — SB3's SubprocVecEnv for parallel rollouts → faster training
