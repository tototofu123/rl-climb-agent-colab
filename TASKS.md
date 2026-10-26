# TASKS

## Stage 0: Env Scaffolding (RL concept: MDP, step loop)

- [ ] Create `env/__init__.py`
- [ ] Create `env/climbing_env.py` — Gymnasium Env subclass
- [ ] Define observation space (Dict/Box)
- [ ] Define action space (Discrete)
- [ ] Implement `reset()` — place agent at start
- [ ] Implement `step()` — parse action, update state, check done
- [ ] Create `render/simple_renderer.py` — Pygame: wall + holds + agent dot
- [ ] Run with random agent: `python -m env.climbing_env`
- [ ] **Verify:** window opens, agent flails randomly, episodes end

## Stage 1: Physics + Constraints (RL concept: state/action space design)

- [ ] Create `physics/__init__.py`
- [ ] Create `physics/body.py` — Box2D world, climber body, holds
- [ ] Create `physics/constraints.py` — max_reach, limb state machine
- [ ] Create `physics/wall.py` — wall boundary, hold definitions
- [ ] Integrate physics into env (replace simple state)
- [ ] Render body: CoM circle + limb lines to holds
- [ ] **Verify:** body hangs from holds, swinging under gravity, falls if all limbs release

## Stage 2: First Training Loop (RL concept: PPO, policy gradient)

- [ ] Install SB3 + check GPU availability
- [ ] Create `training/__init__.py`
- [ ] Create `training/train.py` — PPO with default params
- [ ] Add vectorized envs (SubprocVecEnv, 4 workers)
- [ ] Add TensorBoard logging
- [ ] Add model checkpoint callback
- [ ] Add evaluation callback
- [ ] Run for 100k steps
- [ ] **Verify:** TensorBoard shows reward trending up, model saves

## Stage 3: Reward Engineering (RL concept: shaping, hacking)

- [ ] Create `experiments/__init__.py`
- [ ] Create `experiments/configs/` directory
- [ ] Write 3+ reward configs (dense, sparse, hybrid)
- [ ] Create `training/train.py` config loading from YAML
- [ ] Train with each config, log to separate TensorBoard runs
- [ ] Compare reward curves
- [ ] Identify best config
- [ ] **Verify:** clear improvement over random, best config identified

## Stage 4: Dyno Mechanics (RL concept: exploration, credit assignment)

- [ ] Add dyno action to action space
- [ ] Add dyno physics: release all + impulse
- [ ] Create advanced wall layout requiring dyno
- [ ] Add dyno reward shaping (bonus for landing dyno)
- [ ] Train with dyno enabled
- [ ] **Verify:** agent attempts and lands dynos, solves dyno-required wall

## Stage 5: Generalization (RL concept: overfitting, curriculum)

- [ ] Create 3+ wall layouts
- [ ] Add curriculum learning: easy → medium → hard
- [ ] Add domain randomization: slight hold position variance
- [ ] Train with curriculum
- [ ] Test zero-shot on new wall
- [ ] Ablation: train on single wall, show it doesn't generalize
- [ ] **Verify:** curriculum agent solves new walls, single-wall agent doesn't

## Final Deliverables

- [ ] Save best model to `agents/best_model.zip`
- [ ] Record replay video to `render/replays/final_run.mp4`
- [ ] Export training curves to `experiments/results/final_curves.png`
- [ ] Write one-page summary: what worked, what didn't, key learnings
- [ ] Clean up README.md with final results
