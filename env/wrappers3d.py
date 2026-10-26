"""Wrappers for the 3D climbing env."""

import numpy as np

from env.climbing_env_3d import ClimbingEnv3D
from wallimport.routeset import generate_route, task_from_route


class TaskRandomizationWrapper(ClimbingEnv3D):
    """Re-generate the boulder problem each episode from a pool of seeds.

    Keeps one task during an episode (stable action semantics), samples the
    next one on reset. This is the 3D analogue of domain randomization.
    """

    def __init__(self, wall=None, seeds=None, difficulty="medium", pool_size=8,
                 **kwargs):
        super().__init__(wall=wall, route_seed=seeds[0] if seeds else 0,
                         difficulty=difficulty, **kwargs)
        self.pool = list(seeds) if seeds else list(range(pool_size))
        self._pool_pos = 0
        self.task_label = f"pool[{len(self.pool)}]"

    def reset(self, seed=None, options=None):
        self.route_seed = self.pool[self._pool_pos % len(self.pool)]
        self._pool_pos += 1
        self.task_holds = task_from_route(self.wall,
                                          generate_route(self.wall, seed=self.route_seed,
                                                         difficulty=self.difficulty),
                                          max_slots=self.max_slots)
        self.n_slots = len(self.task_holds)
        self.target_hold = self.task_holds[-1]
        return super().reset(seed=seed, options=options)


class LongRouteCurriculumWrapper(ClimbingEnv3D):
    """Length curriculum for one fixed long route.

    Each episode samples either a truncated prefix of the route (target at an
    achievable height) or the full route. Topping out lengthens the required
    prefix; falling shortens it — the curriculum self-tunes from episode
    outcomes, so early training gets frequent reachable targets and later
    training converges on the full route.
    """

    def __init__(self, wall=None, full_task=None, difficulty="easy",
                 route_seed=0, min_holds=10, p_trunc=0.7, **kwargs):
        super().__init__(wall=wall, task_holds=full_task, **kwargs)
        self.full_task = list(full_task)
        self.min_holds = min_holds
        self.p_trunc = p_trunc
        self.max_len = len(full_task)
        self.progress = 0.3        # fraction of the route currently required
        self._task_rng = np.random.default_rng(route_seed)
        self.task_label = "long-route curriculum"

    def _sample_task(self):
        if self._task_rng.random() < self.p_trunc:
            lo = self.min_holds
            hi = max(lo + 1, int(self.min_holds
                                 + (self.max_len - self.min_holds) * self.progress))
            n = int(self._task_rng.integers(lo, hi + 1))
        else:
            n = self.max_len
        return self.full_task[:n]

    def reset(self, seed=None, options=None):
        self.task_holds = self._sample_task()
        self.n_slots = len(self.task_holds)
        self.target_hold = self.task_holds[-1]
        return ClimbingEnv3D.reset(self, seed=seed, options=options)

    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        if info.get("target_reached"):
            self.progress = min(1.0, self.progress + 0.04)
        elif term:                          # fell
            self.progress = max(0.15, self.progress - 0.01)
        return obs, r, term, trunc, info
