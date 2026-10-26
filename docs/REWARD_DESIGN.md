# Reward Design

Reward design is the most important lever you have. Bad rewards → agent learns wrong thing. No rewards → agent learns nothing.

## Signal Components

### 1. Progress Bonus (dense shaping)
```
progress = previous_dist_to_target - current_dist_to_target
```
- Positive when agent moves closer, negative when it moves away
- Normalized by wall diagonal so values are in ~[-1, 1]
- Weight: α = 2.0

### 2. Hold Bonus (sparse)
```
+0.2 when limb attaches to a new hold
```
- Encourages active engagement with holds
- Only awarded once per hold per episode (no farming)
- Weight: β = 0.2

### 3. Time Penalty
```
-0.01 per step
```
- Encourages efficiency (don't dither)
- Equivalent to: agent has ~100 steps before penalty dominates
- Weight: γ = 0.01

### 4. Reach Penalty
```
-0.3 when agent tries to reach an out-of-range hold
```
- Teaches agent to only attempt feasible moves
- Without this, agent spams random reaches forever
- Weight: δ = 0.3

### 5. Fall Penalty
```
-1.0 when agent falls
```
- Terminal negative feedback
- Only applied once per episode (at termination)
- Weight: ε = 1.0

### 6. Success Bonus
```
+5.0 when agent reaches target
```
- Terminal positive feedback
- Large enough to outweigh step penalties over a typical episode
- Weight: ζ = 5.0

## Total Reward

```python
reward = (
    α * progress_bonus
    + β * hold_bonus
    + γ * time_penalty
    + δ * reach_penalty
    + ε * fall_penalty
    + ζ * success_bonus
)
```

## Reward Configurations to Experiment With

| Config | α (progress) | β (hold) | γ (time) | δ (reach) | ε (fall) | ζ (success) | Character |
|--------|-------------|---------|---------|----------|---------|------------|-----------|
| dense  | 2.0 | 0.0 | 0.0 | 0.0 | -1.0 | 5.0 | Pure distance shaping |
| sparse | 0.0 | 0.0 | 0.0 | 0.0 | -1.0 | 5.0 | Only success/fall |
| hybrid | 1.0 | 0.2 | -0.01 | -0.3 | -1.0 | 5.0 | Best of both |
| speed  | 0.5 | 0.1 | -0.05 | -0.3 | -2.0 | 10.0 | Prioritizes speed |
| safe   | 2.0 | 0.3 | -0.005 | -0.5 | -3.0 | 3.0 | Penalizes risky moves |

## Common Failure Modes

1. **Agent stays still**: time penalty too high, or progress bonus too noisy
2. **Agent oscillates**: progress bonus has high variance; smooth it with a running average
3. **Agent ignores target**: success bonus too small vs step penalty
4. **Agent falls constantly**: fall penalty too small, or progress bonus at the edge is misleading
5. **Reward hacking**: agent finds way to accumulate hold bonus without progressing (e.g. rapidly attach/detach) — add cooldown

## Debugging Rewards

Always sanity-check with:
```python
# Print each component per step
info = {
    "reward_progress": progress_bonus,
    "reward_hold": hold_bonus,
    "reward_time": time_penalty,
    "reward_reach": reach_penalty,
    "reward_fall": fall_penalty,
    "reward_success": success_bonus,
}
```
Log these to TensorBoard to see which signals the agent actually receives.
