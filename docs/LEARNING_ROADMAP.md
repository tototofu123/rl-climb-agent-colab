# Learning Roadmap

**Overarching goal: learn RL by building something tangible.**

Each stage has:
- A concrete deliverable
- RL concepts you will internalize
- What to run/see at the end

---

## Stage 0 — Setup & First Env

### Deliverable
`env/climbing_env.py` — minimal Gymnasium env with a random agent

### What you build
- A Gymnasium `Env` subclass
- `reset()` and `step()` methods
- Simple observation/action spaces (Box, Discrete)
- A rendering method (Pygame window showing wall + holds + agent dot)

### RL concepts
- The `env` abstraction: what is a Markov Decision Process?
- Observation = what the agent sees
- Action = what the agent can do
- Step loop: observe → act → observe
- Episode = reset to done

### Run
```
python -m env.climbing_env
```
See a window with a wall, holds, and a dot randomly moving.

### Check
- [ ] Env runs without errors
- [ ] Random agent completes episodes
- [ ] Render window shows meaningful state

---

## Stage 1 — Physics & Constraints

### Deliverable
`physics/body.py` + `physics/constraints.py` — body model with reach limits

### What you build
- Box2D world with ground/wall body
- Simplified climber body (CoM + 4 limb anchors)
- Reach constraint: limb can only attach to holds within max_reach distance
- Gravity acting on CoM
- Limb state machine: detached → reaching → attached

### RL concepts
- Constraint satisfaction in RL
- State space design: what matters for the task?
- Action space granularity: too coarse = useless, too fine = impossible to learn

### Run
```
python -m physics.test_body
```
Watch body hang from 2-3 holds, gravity pulling down.

### Check
- [ ] Body hangs from holds correctly
- [ ] Can't reach holds beyond max_reach
- [ ] Releasing a limb makes body swing
- [ ] Falling (no holds) resets episode

---

## Stage 2 — First Training Loop

### Deliverable
`training/train.py` — PPO training on the climbing env

### What you build
- SB3 PPO agent connected to your env
- Vectorized env (4+ parallel instances)
- TensorBoard logging
- Model checkpointing
- Eval callback

### RL concepts
- PPO algorithm in practice: policy, value function, clipped surrogate
- On-policy vs off-policy (why PPO?)
- Parallel environments for better gradient estimates
- Hyperparameters: learning_rate, n_steps, batch_size, gamma, gae_lambda
- Entropy bonus for exploration

### Run
```
tensorboard --logdir ./experiments/logs
python training/train.py --total_timesteps 200_000
```
Watch the mean reward curve climb (hopefully).

### Check
- [ ] Training runs without errors
- [ ] TensorBoard shows reward curve
- [ ] Model checkpoint saved
- [ ] Eval agent shows some progress (not perfect)

---

## Stage 3 — Reward Engineering

### Deliverable
Multiple reward configs tested and compared

### What you do
- Tune reward weights systematically
- Try: dense only / sparse only / hybrid
- Compare: time penalty vs none
- Try: curriculum (easy wall → hard wall)

### RL concepts
- Reward shaping: what helps vs what breaks learning?
- Sparse vs dense rewards — tradeoffs
- Reward hacking: agent finds unintended shortcuts
- Curriculum learning: progressive difficulty

### Run
```
python training/train.py --config experiments/configs/reward_dense.yaml
python training/train.py --config experiments/configs/reward_sparse.yaml
# Compare curves
```

### Check
- [ ] At least 3 reward configs tested
- [ ] Clear comparison chart
- [ ] Best config identified
- [ ] You can explain *why* it worked best

---

## Stage 4 — Dyno (Jump)

### Deliverable
Agent can perform dynamic jumps to reach far holds

### What you build
- Dyno action: release all limbs + apply upward impulse
- Dyno-specific reward shaping
- Wind-up animation (visual only)

### RL concepts
- Exploration in high-risk actions
- Temporal credit assignment: jump reward comes after landing
- Trade-off: static climbing (safe) vs dyno (fast but risky)

### Run
```
python training/train.py --enable_dyno --total_timesteps 500_000
```
Watch agent occasionally dyno to distant holds.

### Check
- [ ] Agent attempts dynos (not just static)
- [ ] Dyno success rate > 20%
- [ ] Training curve converges higher than static-only

---

## Stage 5 — Generalization

### Deliverable
Agent trained on multiple wall layouts

### What you build
- Curriculum: easy wall → medium wall → hard wall
- Multiple wall definitions in `experiments/wall_layouts/`
- Domain randomization: vary hold positions slightly each episode

### RL concepts
- Overfitting to one layout vs generalizing
- Domain randomization for robustness
- Curriculum learning in practice

### Run
```
python training/train.py --curriculum
```
Agent solves walls it's never seen before.

### Check
- [ ] Agent works on 3+ different walls
- [ ] Zero-shot transfer to new wall
- [ ] Ablation: without curriculum, agent fails on new walls

---

## Final Project

**"A 2D climber that can solve multiple walls using static moves and dynos, trained with PPO."**

### Deliverables
- Trained model (`agents/best_model.zip`)
- Replay video (`render/replays/final_run.mp4`)
- Training curves (`experiments/results/final_curves.png`)
- This architecture doc (living document)
- One-page summary: what worked, what didn't, key learnings

### What you'll have learned
1. How to implement a Gymnasium environment from scratch
2. How physics constraints translate to RL state/action spaces
3. How to train PPO end-to-end with SB3
4. How reward shaping drives (or breaks) learning
5. How curriculum and generalization work
6. Debugging RL: why isn't it learning? (most important skill!)
