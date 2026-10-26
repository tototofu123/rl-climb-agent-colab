"""TDD tests for Stage 0 — SimpleClimbingEnv."""

import numpy as np
import pytest

from env.climbing_env import SimpleClimbingEnv


@pytest.fixture
def env():
    return SimpleClimbingEnv()


class TestObservationSpace:
    def test_shape_is_correct(self, env):
        obs, _ = env.reset()
        assert obs.shape == (27,), f"Expected (27,), got {obs.shape}"
        assert obs.dtype == np.float32

    def test_contains_agent_position(self, env):
        obs, _ = env.reset()
        assert 0 <= obs[0] <= env.wall.width
        assert 0 <= obs[1] <= env.wall.height

    def test_contains_hold_positions(self, env):
        obs, _ = env.reset()
        holds_flat = obs[2:10]
        assert len(holds_flat) == 8

    def test_contains_target_distance(self, env):
        obs, _ = env.reset()
        dist = obs[-1]
        assert dist >= 0

    def test_observation_space_matches(self, env):
        assert env.observation_space.shape == (27,)


class TestActionSpace:
    def test_four_actions(self, env):
        assert env.action_space.n == 4

    def test_all_actions_are_valid(self, env):
        env.reset()
        for a in range(4):
            env.step(a)


class TestStep:
    def test_moves_agent_up(self, env):
        env.reset()
        x0, y0 = env.agent_pos
        env.step(0)
        assert env.agent_pos[1] == y0 - env.step_size

    def test_moves_agent_down(self, env):
        env.reset()
        _, y0 = env.agent_pos[1], env.agent_pos[1]
        env.step(1)
        assert env.agent_pos[1] == y0 + env.step_size

    def test_moves_agent_left(self, env):
        env.reset()
        x0, _ = env.agent_pos
        env.step(2)
        assert env.agent_pos[0] == x0 - env.step_size

    def test_moves_agent_right(self, env):
        env.reset()
        x0, _ = env.agent_pos
        env.step(3)
        assert env.agent_pos[0] == x0 + env.step_size

    def test_clips_to_wall_bounds(self, env):
        env.reset()
        for _ in range(200):
            env.step(0)
        assert 0 <= env.agent_pos[0] <= env.wall.width
        assert 0 <= env.agent_pos[1] <= env.wall.height


class TestTermination:
    def test_terminates_on_target_reached(self, env):
        obs, _ = env.reset()
        env.agent_pos = env.target_pos.copy()
        _, _, terminated, _, _ = env.step(0)
        assert terminated

    def test_terminates_on_bottom(self, env):
        obs, _ = env.reset()
        env.agent_pos[1] = env.wall.height - 1
        _, _, terminated, _, _ = env.step(1)
        assert terminated

    def test_truncates_on_max_steps(self, env):
        env.reset()
        for i in range(env.max_steps):
            _, _, terminated, truncated, _ = env.step(0)
            if i == env.max_steps - 1:
                assert truncated

    def test_success_reward_is_positive(self, env):
        env.reset()
        env.agent_pos = env.target_pos.copy()
        _, reward, terminated, _, _ = env.step(0)
        assert terminated
        assert reward > 0

    def test_fall_reward_is_negative(self, env):
        env.reset()
        env.agent_pos[1] = env.wall.height - 1
        _, reward, terminated, _, _ = env.step(1)
        assert terminated
        assert reward < 0


class TestReset:
    def test_returns_obs_and_info_dict(self, env):
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_resets_step_count(self, env):
        env.reset()
        env.step(0)
        env.reset()
        assert env.step_count == 0

    def test_resets_agent_position(self, env):
        env.reset()
        env.step(0)
        env.step(0)
        env.step(0)
        env.reset()
        assert np.allclose(env.agent_pos, env.start_pos)


class TestRandomRollout:
    def test_rollout_completes(self, env):
        obs, _ = env.reset()
        steps = 0
        for _ in range(500):
            a = env.action_space.sample()
            obs, r, term, trunc, info = env.step(a)
            steps += 1
            if term or trunc:
                obs, _ = env.reset()
        assert steps == 500

    def test_observation_is_deterministic_shape(self, env):
        env.reset()
        for _ in range(20):
            a = env.action_space.sample()
            obs, _, _, _, _ = env.step(a)
            assert obs.shape == (27,)
