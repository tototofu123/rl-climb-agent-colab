import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .wall import Wall, Hold
from physics.body import ClimberBody

MAX_HOLDS = 12

DEFAULT_REWARD_CONFIG = {
    "progress": 1.0,
    "hold_bonus": 0.2,
    "time_penalty": -0.01,
    "reach_penalty": -0.3,
    "fall_penalty": -1.0,
    "success_bonus": 5.0,
}


class SimpleClimbingEnv(gym.Env):
    """Stage 0: point agent, 4-directional movement, no physics."""

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, wall=None, max_steps=200, reward_config=None):
        super().__init__()
        self.wall = wall or Wall.beginner()
        self.max_steps = max_steps
        self.step_size = 15
        self.rc = reward_config or dict(DEFAULT_REWARD_CONFIG)

        self.start_pos = np.array([350.0, 450.0], dtype=np.float32)
        self.target_pos = np.array(self.wall.target.pos, dtype=np.float32)

        obs_size = 2 + 2 * MAX_HOLDS + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        self.agent_pos = None
        self.step_count = None
        self._prev_dist = None

    def _get_obs(self):
        holds_flat = np.zeros(2 * MAX_HOLDS, dtype=np.float32)
        for i, h in enumerate(self.wall.holds):
            if i < MAX_HOLDS:
                holds_flat[2 * i : 2 * i + 2] = h.pos
        dist = np.linalg.norm(self.agent_pos - self.target_pos)
        return np.concatenate([self.agent_pos, holds_flat, [dist]]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self.start_pos.copy()
        self.step_count = 0
        self._prev_dist = np.linalg.norm(self.agent_pos - self.target_pos)
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1

        if action == 0:
            self.agent_pos[1] -= self.step_size
        elif action == 1:
            self.agent_pos[1] += self.step_size
        elif action == 2:
            self.agent_pos[0] -= self.step_size
        elif action == 3:
            self.agent_pos[0] += self.step_size

        self.agent_pos[0] = np.clip(self.agent_pos[0], 0, self.wall.width)
        self.agent_pos[1] = np.clip(self.agent_pos[1], 0, self.wall.height)

        dist = np.linalg.norm(self.agent_pos - self.target_pos)
        comp = {}

        comp["progress"] = self.rc["progress"] * (self._prev_dist - dist)
        comp["time_penalty"] = self.rc["time_penalty"]
        self._prev_dist = dist

        terminated = False
        truncated = False

        if dist < 25:
            comp["success_bonus"] = self.rc["success_bonus"]
            terminated = True
        elif self.agent_pos[1] >= self.wall.height - 1:
            comp["fall_penalty"] = self.rc["fall_penalty"]
            terminated = True

        if self.step_count >= self.max_steps:
            truncated = True

        reward = sum(comp.values())
        return self._get_obs(), reward, terminated, truncated, {"comp": comp}

    def render(self):
        pass


class PhysicsClimbingEnv(gym.Env):
    """Stage 1-4: physics-based climbing with body, limbs, gravity, and dyno."""

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, wall=None, max_steps=300, dt=0.05, physics_steps=5,
                 reward_config=None, max_holds=None):
        super().__init__()
        self.wall = wall or Wall.beginner()
        self.max_steps = max_steps
        self.dt = dt
        self.physics_steps = physics_steps
        self.max_reach_static = 150.0
        self.max_reach_dyno = 250.0
        self.dyno_duration = 20
        self.rc = reward_config or dict(DEFAULT_REWARD_CONFIG)

        self.n_holds = len(self.wall.holds)
        self.max_holds = max_holds if max_holds is not None else self.n_holds
        self.target_pos = np.array(self.wall.target.pos, dtype=np.float32)

        self.action_space = spaces.Discrete(1 + self.max_holds + 5)

        obs_size = 4 + 12 + 3 * self.max_holds + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        self.body = None
        self.step_count = None
        self._grabbed_holds = None
        self._prev_dist = None

    def _get_obs(self):
        body_arr = np.concatenate([self.body.pos, self.body.vel])

        limbs_arr = np.zeros(12, dtype=np.float32)
        for i, limb in enumerate(self.body.limbs):
            if limb.attached and limb.hold_pos is not None:
                limbs_arr[3 * i : 3 * i + 2] = limb.hold_pos
                limbs_arr[3 * i + 2] = 1.0

        holds_arr = np.zeros(3 * self.max_holds, dtype=np.float32)
        for i, h in enumerate(self.wall.holds):
            if i < self.max_holds:
                holds_arr[3 * i : 3 * i + 2] = h.pos
                holds_arr[3 * i + 2] = 1.0 if h.is_target else 0.0

        dist = np.linalg.norm(self.body.pos - self.target_pos)
        return np.concatenate([body_arr, limbs_arr, holds_arr, [dist]]).astype(np.float32)

    def _compute_start_pos(self):
        non_target = [(i, h) for i, h in enumerate(self.wall.holds) if not h.is_target]
        non_target.sort(key=lambda x: x[1].pos[1], reverse=True)  # bottom-most first
        bottom = [h.pos for _, h in non_target[:4]]
        if not bottom:
            return [350.0, 450.0]
        cx = sum(p[0] for p in bottom) / len(bottom)
        cy = sum(p[1] for p in bottom) / len(bottom)
        # Offset upward so hands are above body center
        cy -= 25
        return [cx, cy]

    def _pick_start_holds(self):
        """Pick 4 bottom-most non-target holds, one per limb."""
        non_target = [(i, h) for i, h in enumerate(self.wall.holds) if not h.is_target]
        non_target.sort(key=lambda x: x[1].pos[1], reverse=True)
        result = []
        for i, h in non_target[:4]:
            result.append((i, h.pos))
        return result

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        start_pos = self._compute_start_pos()
        self.body = ClimberBody(start_pos)
        self.step_count = 0
        self._grabbed_holds = set()
        self._prev_dist = np.linalg.norm(self.body.pos - self.target_pos)
        # Attach all 4 limbs to bottom holds
        start_holds = self._pick_start_holds()
        hold_positions = [pos for _, pos in start_holds]
        self.body.init_full_grip(hold_positions)
        for i, _ in start_holds:
            self._grabbed_holds.add(i)
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        grabbed_new = False
        failed_reach = False
        dyno_attempted = False
        dyno_caught = False

        current_max_reach = self.max_reach_dyno if self.body.in_dyno else self.max_reach_static

        if 1 <= action <= self.n_holds:
            hold_idx = action - 1
            hold = self.wall.holds[hold_idx]
            if self.body.has_free_limb():
                success = self.body.reach_hold(hold.pos, current_max_reach)
                if success:
                    was_dyno_catch = self.body.in_dyno
                    if hold_idx not in self._grabbed_holds:
                        grabbed_new = True
                        self._grabbed_holds.add(hold_idx)
                    if was_dyno_catch:
                        dyno_caught = True
                else:
                    failed_reach = True

        elif self.max_holds + 1 <= action <= self.max_holds + 4:
            limb_idx = action - self.max_holds - 1
            self.body.release_limb(limb_idx)

        elif action == self.max_holds + 5:
            self.body.dyno_impulse(dyno_duration=self.dyno_duration)
            dyno_attempted = True

        sub_dt = self.dt / self.physics_steps
        for _ in range(self.physics_steps):
            self.body.step(sub_dt, wall_width=self.wall.width, wall_height=self.wall.height)

        dist = np.linalg.norm(self.body.pos - self.target_pos)
        fallen = self.body.is_falling()

        comp = {}
        comp["progress"] = self.rc["progress"] * (self._prev_dist - dist)
        comp["time_penalty"] = self.rc["time_penalty"]
        if grabbed_new:
            comp["hold_bonus"] = self.rc["hold_bonus"]
        elif failed_reach:
            comp["reach_penalty"] = self.rc["reach_penalty"]
        if dyno_attempted:
            comp["dyno_bonus"] = 0.1
        if dyno_caught:
            comp["dyno_bonus"] = comp.get("dyno_bonus", 0) + 0.5
        self._prev_dist = dist

        terminated = False
        truncated = False

        if dist < 25:
            comp["success_bonus"] = self.rc["success_bonus"]
            terminated = True
        elif fallen:
            comp["fall_penalty"] = self.rc["fall_penalty"]
            terminated = True

        if self.step_count >= self.max_steps:
            truncated = True

        reward = sum(comp.values())
        return self._get_obs(), reward, terminated, truncated, {"comp": comp}

    def render(self):
        pass
