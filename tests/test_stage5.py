"""Tests for Stage 5 — Curriculum Learning and Domain Randomization."""

import numpy as np
import pytest

from env.climbing_env import PhysicsClimbingEnv
from env.wall import Wall
from env.wrappers import CurriculumWrapper, DomainRandomizationWrapper


WALLS = [
    ("beginner", Wall.beginner()),
    ("intermediate", Wall.intermediate()),
    ("dyno_gap", Wall.dyno_gap()),
    ("expert", Wall.expert()),
]

THRESHOLDS = [200.0, 250.0, 150.0]


class TestWallLayouts:
    def test_beginner_has_holds(self):
        w = Wall.beginner()
        assert len(w.holds) == 11
        assert w.target is not None

    def test_intermediate_has_holds(self):
        w = Wall.intermediate()
        assert len(w.holds) == 12
        assert w.target is not None

    def test_dyno_gap_has_holds(self):
        w = Wall.dyno_gap()
        assert len(w.holds) == 7
        assert w.target is not None

    def test_expert_has_holds(self):
        w = Wall.expert()
        assert len(w.holds) == 13
        assert w.target is not None

    def test_all_walls_have_target(self):
        for name, wall in WALLS:
            assert wall.target is not None, f"{name} missing target"
            assert wall.target.is_target is True

    def test_expert_has_wide_spread(self):
        w = Wall.expert()
        from physics.body import MAX_REACH
        # Check that some holds are far apart (requires path planning)
        max_dist = 0
        for i, h1 in enumerate(w.holds):
            for h2 in w.holds[i+1:]:
                d = np.linalg.norm(np.array(h1.pos) - np.array(h2.pos))
                max_dist = max(max_dist, d)
        assert max_dist > MAX_REACH * 2, "Expert wall should have holds spread across the wall"


class TestCloneRandomized:
    def test_clone_preserves_hold_count(self):
        w = Wall.beginner()
        w2 = w.clone_randomized(max_offset=10)
        assert len(w2.holds) == len(w.holds)

    def test_clone_preserves_target(self):
        w = Wall.beginner()
        w2 = w.clone_randomized(max_offset=10)
        assert w2.target is not None
        assert w2.target.pos == w.target.pos

    def test_clone_moves_non_target_holds(self):
        rng = np.random.default_rng(42)
        w = Wall.beginner()
        w2 = w.clone_randomized(max_offset=15, rng=rng)
        moved = False
        for h1, h2 in zip(w.holds, w2.holds):
            if not h1.is_target:
                if h1.pos != h2.pos:
                    moved = True
                    break
        assert moved, "At least one non-target hold should move"

    def test_clone_respects_bounds(self):
        rng = np.random.default_rng(42)
        w = Wall.beginner()
        w2 = w.clone_randomized(max_offset=100, rng=rng)
        for h in w2.holds:
            assert 30 <= h.pos[0] <= 770
            assert 30 <= h.pos[1] <= 570


class TestDomainRandomization:
    def test_wrapper_changes_holds(self):
        env = PhysicsClimbingEnv()
        orig_holds = [h.pos for h in env.wall.holds]
        drw = DomainRandomizationWrapper(env, max_offset=15, seed=42)
        drw.reset()
        new_holds = [h.pos for h in env.wall.holds]
        moved = any(o != n for o, n in zip(orig_holds, new_holds) if True)
        assert moved, "Domain randomization should move holds"

    def test_wrapper_preserves_target(self):
        env = PhysicsClimbingEnv()
        drw = DomainRandomizationWrapper(env, max_offset=15, seed=42)
        drw.reset()
        assert env.wall.target is not None
        assert env.wall.target.is_target is True

    def test_wrapper_step_works(self):
        env = PhysicsClimbingEnv()
        drw = DomainRandomizationWrapper(env, max_offset=15, seed=42)
        drw.reset()
        obs, r, term, trunc, info = drw.step(env.action_space.sample())
        assert isinstance(obs, np.ndarray)
        assert isinstance(r, (float, np.floating))

    def test_wrapper_reproducible_with_seed(self):
        env1 = PhysicsClimbingEnv()
        drw1 = DomainRandomizationWrapper(env1, max_offset=15, seed=42)
        drw1.reset()
        holds1 = [h.pos for h in env1.wall.holds]

        env2 = PhysicsClimbingEnv()
        drw2 = DomainRandomizationWrapper(env2, max_offset=15, seed=42)
        drw2.reset()
        holds2 = [h.pos for h in env2.wall.holds]

        assert holds1 == holds2, "Same seed should produce same randomization"


class TestCurriculumWrapper:
    def test_starts_at_beginner(self):
        env = PhysicsClimbingEnv()
        cw = CurriculumWrapper(env, WALLS, THRESHOLDS)
        cw.reset()
        assert cw.current_wall_name == "beginner"

    def test_tracks_rewards(self):
        env = PhysicsClimbingEnv()
        cw = CurriculumWrapper(env, WALLS, THRESHOLDS)
        cw.reset()
        for _ in range(10):
            cw.step(env.action_space.sample())
        assert len(cw.rewards) == 10

    def test_promotes_on_high_reward(self):
        env = PhysicsClimbingEnv()
        cw = CurriculumWrapper(env, WALLS, THRESHOLDS, window=5)
        cw.reset()
        # Simulate high rewards
        cw.rewards = [300.0] * 5
        cw._maybe_promote()
        assert cw.stage == 1
        assert cw.current_wall_name == "intermediate"

    def test_no_promote_below_threshold(self):
        env = PhysicsClimbingEnv()
        cw = CurriculumWrapper(env, WALLS, THRESHOLDS, window=5)
        cw.reset()
        cw.rewards = [100.0] * 5
        cw._maybe_promote()
        assert cw.stage == 0

    def test_no_promote_insufficient_data(self):
        env = PhysicsClimbingEnv()
        cw = CurriculumWrapper(env, WALLS, THRESHOLDS, window=20)
        cw.reset()
        cw.rewards = [300.0] * 10
        cw._maybe_promote()
        assert cw.stage == 0

    def test_curriculum_runs_without_crash(self):
        env = PhysicsClimbingEnv()
        cw = CurriculumWrapper(env, WALLS, THRESHOLDS, window=5)
        obs, _ = cw.reset()
        for _ in range(50):
            obs, r, term, trunc, info = cw.step(env.action_space.sample())
            if term or trunc:
                obs, _ = cw.reset()
        assert cw.episodes > 0


class TestTrainStage5:
    def test_train_curriculum_imports(self):
        from training.train_stage5 import TrainConfig, make_env
        config = TrainConfig(curriculum=True, domain_rand=True)
        fn = make_env(config)
        env = fn()
        assert hasattr(env, "stage") or hasattr(env, "env")

    def test_train_expert_wall(self):
        from training.train_stage5 import TrainConfig, make_env
        config = TrainConfig(wall_layout="expert")
        fn = make_env(config)
        env = fn()
        assert len(env.wall.holds) == 13
