"""TDD tests for Stage 4 — Dyno Mechanics."""

import numpy as np
import pytest

from env.climbing_env import PhysicsClimbingEnv
from env.wall import Wall
from physics.body import ClimberBody


@pytest.fixture
def env():
    return PhysicsClimbingEnv()


@pytest.fixture
def dyno_env():
    return PhysicsClimbingEnv(wall=Wall.dyno_gap())


@pytest.fixture
def body():
    b = ClimberBody([400.0, 300.0])
    b.reach_hold([410.0, 290.0])
    return b


class TestBodyDyno:
    def test_dyno_releases_all_limbs(self, body):
        body.dyno_impulse()
        assert body.num_attached_limbs() == 0

    def test_dyno_applies_upward_impulse(self, body):
        vy0 = body.vel[1]
        body.dyno_impulse(upward_velocity=400.0)
        assert body.vel[1] < vy0 - 100
        assert body.vel[1] < -300

    def test_dyno_sets_active_flag(self, body):
        body.dyno_impulse(dyno_duration=10)
        assert body.in_dyno is True
        assert body.dyno_steps == 10

    def test_dyno_expires_after_steps(self, body):
        body.dyno_impulse(dyno_duration=4)
        for _ in range(4):
            assert body.in_dyno is True
            body.step(0.05)
        assert body.in_dyno is False
        assert body.dyno_steps == 0

    def test_dyno_does_not_crash_on_fresh_body(self):
        b = ClimberBody([400.0, 300.0])
        b.dyno_impulse()
        assert b.in_dyno is True

    def test_dyno_reach_allows_far_holds(self):
        b = ClimberBody([400.0, 400.0])
        b.dyno_impulse()
        far_hold = (400.0, 200.0)
        dist = np.linalg.norm(b.pos - np.array(far_hold))
        assert dist > 150.0
        assert dist <= 250.0
        success = b.reach_hold(far_hold, max_reach=250.0)
        assert success is True


class TestEnvDyno:
    def test_dyno_action_exists(self, env):
        n = env.n_holds
        dyno_action = n + 5
        assert dyno_action < env.action_space.n

    def test_dyno_action_triggers(self, env):
        env.reset()
        n = env.n_holds
        env.step(n + 5)
        assert env.body.in_dyno is True

    def test_dyno_increases_reach(self, env):
        env.reset()
        n = env.n_holds
        env.step(n + 5)
        assert env.body.in_dyno
        assert env.max_reach_dyno > env.max_reach_static

    def test_dyno_info_contains_bonus(self, env):
        env.reset()
        n = env.n_holds
        _, _, _, _, info = env.step(n + 5)
        assert "comp" in info
        assert "dyno_bonus" in info["comp"]

    def test_dyno_action_space_includes_dyno(self, env):
        n = env.n_holds
        expected = 1 + n + 5
        assert env.action_space.n == expected

    def test_dyno_catch_reward(self, env):
        env.reset()
        n = env.n_holds
        env.step(n + 5)
        env.step(0)
        last_hold_action = n
        _, _, _, _, info = env.step(last_hold_action)
        comp = info.get("comp", {})
        caught = comp.get("dyno_bonus", 0) > 0.4
        if not caught:
            assert "hold_bonus" in comp


class TestDynoWall:
    def test_dyno_wall_has_holds(self, dyno_env):
        assert len(dyno_env.wall.holds) > 4

    def test_dyno_wall_has_gaps_over_150(self, dyno_env):
        holds = dyno_env.wall.holds
        target = holds[0]
        for i in range(1, len(holds)):
            dist = np.linalg.norm(np.array(holds[i].pos) - np.array(holds[i-1].pos) if i > 0 else 0)
            if i <= 2:
                assert dist > 150.0, f"Gap between hold {i-1} and {i} is {dist}, should be >150"

    def test_dyno_wall_target_is_top(self, dyno_env):
        assert dyno_env.wall.target is not None
        assert dyno_env.wall.target.pos[1] < dyno_env.wall.holds[1].pos[1]

    def test_dyno_wall_requires_dyno_for_target(self, dyno_env):
        dyno_env.reset()
        holds = dyno_env.wall.holds
        from_target = np.linalg.norm(np.array(holds[0].pos) - np.array(holds[1].pos))
        assert from_target > dyno_env.max_reach_static
        assert from_target <= dyno_env.max_reach_dyno


class TestTrainWall:
    def test_train_accepts_dyno_wall(self):
        from training.train import TrainConfig, make_env
        config = TrainConfig(wall_layout="dyno_gap")
        fn = make_env(config)
        env = fn()
        assert len(env.wall.holds) > 4


class TestRandomRolloutDyno:
    def test_dyno_rollout_completes(self, dyno_env):
        dyno_env.reset()
        steps = 0
        for _ in range(300):
            a = dyno_env.action_space.sample()
            obs, r, term, trunc, info = dyno_env.step(a)
            steps += 1
            if term or trunc:
                dyno_env.reset()
        assert steps == 300
