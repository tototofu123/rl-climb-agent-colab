"""TDD tests for Stage 1 — PhysicsClimbingEnv."""

import numpy as np
import pytest

from env.climbing_env import PhysicsClimbingEnv
from physics.body import ClimberBody, LimbState


@pytest.fixture
def env():
    return PhysicsClimbingEnv()


@pytest.fixture
def body():
    return ClimberBody([400.0, 300.0])


class TestLimbState:
    def test_starts_detached(self):
        limb = LimbState()
        assert limb.attached is False
        assert limb.hold_pos is None

    def test_attach_sets_state(self):
        limb = LimbState()
        limb.attached = True
        limb.hold_pos = (400, 200)
        assert limb.attached
        assert limb.hold_pos == (400, 200)


class TestClimberBody:
    def test_starts_at_position(self, body):
        assert np.allclose(body.pos, [400.0, 300.0])

    def test_starts_with_zero_velocity(self, body):
        assert np.allclose(body.vel, [0.0, 0.0])

    def test_has_four_limbs(self, body):
        assert len(body.limbs) == 4

    def test_gravity_moves_body_down(self, body):
        y0 = body.pos[1].copy()
        body.step(0.1)
        assert body.pos[1] > y0

    def test_gravity_increases_velocity(self, body):
        v0 = body.vel[1].copy()
        body.step(0.1)
        assert body.vel[1] > v0

    def test_free_limb_count(self, body):
        assert body.num_attached_limbs() == 0
        assert body.has_free_limb()

    def test_reach_hold_attaches_limb(self, body):
        body.reach_hold(body.pos.copy(), max_reach=150.0)
        assert body.num_attached_limbs() == 1

    def test_reach_hold_fails_out_of_range(self, body):
        far_hold = (body.pos[0] + 1000, body.pos[1])
        result = body.reach_hold(far_hold, max_reach=150.0)
        assert result is False
        assert body.num_attached_limbs() == 0

    def test_reach_uses_free_limbs(self, body):
        body.reach_hold(body.pos.copy(), max_reach=150.0)
        body.reach_hold(body.pos.copy(), max_reach=150.0)
        body.reach_hold(body.pos.copy(), max_reach=150.0)
        body.reach_hold(body.pos.copy(), max_reach=150.0)
        assert body.num_attached_limbs() == 4
        assert body.has_free_limb() is False

    def test_cannot_attach_when_no_free_limbs(self, body):
        for _ in range(4):
            body.reach_hold(body.pos.copy(), max_reach=150.0)
        result = body.reach_hold(body.pos.copy(), max_reach=150.0)
        assert result is False

    def test_release_limb_detaches(self, body):
        body.reach_hold(body.pos.copy(), max_reach=150.0)
        body.release_limb(0)
        assert body.num_attached_limbs() == 0

    def test_release_does_nothing_for_already_detached(self, body):
        body.release_limb(2)
        assert body.num_attached_limbs() == 0

    def test_constraint_enforces_max_reach(self, body):
        hold_pos = (400.0, 200.0)
        reach = 150.0
        body.reach_hold(hold_pos, max_reach=reach)
        for _ in range(100):
            body.step(0.05)
        # Constraint enforces distance from joint (not body center) to hold
        joint = body.joint_world(0)
        d_joint = np.linalg.norm(joint - np.array(hold_pos))
        assert d_joint <= reach + 5.0, f"Joint constraint failed: {d_joint} > {reach}"
        # Body center should also be within reasonable range (joint offset + reach)
        d_center = np.linalg.norm(body.pos - np.array(hold_pos))
        assert d_center <= reach + 25.0, f"Center too far: {d_center}"

    def test_all_limbs_detached_is_falling(self, body):
        assert body.is_falling() is True

    def test_attached_limb_means_not_falling(self, body):
        body.reach_hold(body.pos.copy(), max_reach=150.0)
        assert body.is_falling() is False

    def test_get_free_limb_returns_index(self, body):
        assert body.get_free_limb() == 0
        body.reach_hold(body.pos.copy(), max_reach=150.0)
        assert body.get_free_limb() == 1


class TestObservationSpace:
    def test_shape_is_correct(self, env):
        obs, _ = env.reset()
        expected = 4 + 12 + 3 * env.n_holds + 1
        assert obs.shape == (expected,), f"Expected ({expected},), got {obs.shape}"
        assert obs.dtype == np.float32

    def test_observation_space_matches(self, env):
        expected = 4 + 12 + 3 * env.n_holds + 1
        assert env.observation_space.shape == (expected,)


class TestActionSpace:
    def test_action_count(self, env):
        n_holds = len(env.wall.holds)
        expected = 1 + n_holds + 5
        assert env.action_space.n == expected, f"Expected {expected}, got {env.action_space.n}"

    def test_all_actions_are_valid(self, env):
        env.reset()
        for a in range(env.action_space.n):
            env.step(a)
            env.reset()


class TestStep:
    def test_returns_five_values(self, env):
        env.reset()
        result = env.step(0)
        assert len(result) == 5

    def test_noop_action_does_not_crash(self, env):
        env.reset()
        obs2, _, _, _, _ = env.step(0)
        assert isinstance(obs2, np.ndarray)

    def test_reach_action_reaches_nearby_hold(self, env):
        env.reset()
        # Release one limb first so we have a free one to reach
        env.body.release_limb(2)
        last_hold_idx = len(env.wall.holds) - 1
        action = last_hold_idx + 1
        _, _, _, _, _ = env.step(action)
        assert env.body.num_attached_limbs() > 0, \
            f"Action {action} should reach hold {last_hold_idx} at {env.wall.holds[last_hold_idx].pos} " \
            f"from {env.body.pos} (max_reach={env.max_reach})"

    def test_release_action_detaches_limb(self, env):
        env.reset()
        # Release initial grip + any held limbs
        held = [i for i, l in enumerate(env.body.limbs) if l.attached]
        n_holds = len(env.wall.holds)
        for idx in held:
            env.step(n_holds + 1 + idx)
        assert env.body.num_attached_limbs() == 0


class TestTermination:
    def _release_all(self, env):
        n_holds = len(env.wall.holds)
        for i in range(4):
            env.body.release_limb(i)

    def test_terminates_on_target_reached(self, env):
        env.reset()
        self._release_all(env)
        env.body.pos = env.target_pos.copy()
        env.body.vel[:] = 0
        _, _, terminated, _, _ = env.step(0)
        assert terminated

    def test_terminates_on_fall(self, env):
        env.reset()
        self._release_all(env)
        env.body.pos[1] = env.wall.height + 100
        _, _, terminated, _, _ = env.step(0)
        assert terminated

    def test_success_reward_positive(self, env):
        env.reset()
        self._release_all(env)
        env.body.pos = env.target_pos.copy()
        env.body.vel[:] = 0
        _, reward, terminated, _, _ = env.step(0)
        assert terminated
        assert reward > 0

    def test_fall_reward_negative(self, env):
        env.reset()
        self._release_all(env)
        env.body.pos[1] = env.wall.height + 100
        _, reward, terminated, _, _ = env.step(0)
        assert terminated
        assert reward < 0

    def test_truncates_on_max_steps(self, env):
        env.reset()
        for i in range(env.max_steps):
            _, _, terminated, truncated, _ = env.step(0)
            if i == env.max_steps - 1:
                assert truncated


class TestReset:
    def test_returns_obs_and_info(self, env):
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_resets_step_count(self, env):
        env.reset()
        env.step(0)
        env.reset()
        assert env.step_count == 0

    def test_resets_body_position(self, env):
        env.reset()
        for _ in range(10):
            env.step(env.action_space.sample())
        env.reset()
        assert 0 <= env.body.pos[0] <= env.wall.width
        assert 0 <= env.body.pos[1] <= env.wall.height
        assert env.body.num_attached_limbs() == 4


class TestRandomRollout:
    def test_rollout_completes(self, env):
        env.reset()
        steps = 0
        for _ in range(500):
            a = env.action_space.sample()
            obs, r, term, trunc, info = env.step(a)
            steps += 1
            if term or trunc:
                env.reset()
        assert steps == 500
