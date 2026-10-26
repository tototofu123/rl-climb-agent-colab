"""Gymnasium wrappers for curriculum learning and domain randomization."""

import numpy as np
from gymnasium import Env


class CurriculumWrapper(Env):
    """Switches wall layouts based on performance thresholds.

    Curriculum: beginner → intermediate → dyno_gap → expert
    Promotes when mean reward over `window` episodes >= `threshold`.
    """

    def __init__(self, env, walls, thresholds, window=20):
        """
        Args:
            env: The base PhysicsClimbingEnv (will be re-created per stage).
            walls: List of (name, Wall) tuples in difficulty order.
            thresholds: List of reward thresholds to promote at each stage.
            window: Number of episodes to average over.
        """
        self.env = env
        self.walls = walls
        self.thresholds = thresholds
        self.window = window
        self.stage = 0
        self.rewards = []
        self.episodes = 0
        self.successes = 0
        # Forward action_space and observation_space from env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    @property
    def current_wall_name(self):
        return self.walls[self.stage][0]

    def _maybe_promote(self):
        if self.stage >= len(self.walls) - 1:
            return
        if len(self.rewards) < self.window:
            return
        mean_rew = np.mean(self.rewards[-self.window:])
        if mean_rew >= self.thresholds[self.stage]:
            self.stage += 1
            self.rewards = []
            print(f"[Curriculum] Promoted to stage {self.stage}: {self.current_wall_name}")

    def reset(self, seed=None, options=None):
        self._maybe_promote()
        wall_name, wall = self.walls[self.stage]
        self.env.wall = wall
        self.env.n_holds = len(wall.holds)
        self.env.target_pos = np.array(wall.target.pos, dtype=np.float32)
        obs, info = self.env.reset(seed=seed, options=options)
        info["wall_name"] = wall_name
        info["stage"] = self.stage
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.rewards.append(reward)
        self.episodes += 1
        if terminated:
            comp = info.get("comp", {})
            if comp.get("success_bonus", 0) > 0:
                self.successes += 1
            info["wall_name"] = self.current_wall_name
            info["stage"] = self.stage
            info["curriculum_reward"] = np.mean(self.rewards[-self.window:]) if self.rewards else 0
        return obs, reward, terminated, truncated, info

    def close(self):
        self.env.close()


class DomainRandomizationWrapper(Env):
    """Applies small random offsets to hold positions on each reset."""

    def __init__(self, env, max_offset=15, seed=None):
        self.env = env
        self.max_offset = max_offset
        self.rng = np.random.default_rng(seed)
        # Forward action_space and observation_space from env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self, seed=None, options=None):
        self.env.wall = self.env.wall.clone_randomized(
            max_offset=self.max_offset, rng=self.rng
        )
        self.env.n_holds = len(self.env.wall.holds)
        self.env.target_pos = np.array(self.env.wall.target.pos, dtype=np.float32)
        self.env.action_space = type(self.env.action_space)(
            1 + self.env.n_holds + 5
        )
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()
