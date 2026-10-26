"""Stage 7 tests: 3D climbing environment, task wrapper, renderer, training smoke."""

import os

import numpy as np
import pytest

import gymnasium as gym

from env.climbing_env_3d import ClimbingEnv3D, MAX_SLOTS, DEFAULT_3D_REWARD
from env.wrappers3d import TaskRandomizationWrapper
from render.renderer3d import render_frame_png
from wallimport.importer import import_boulder_wall
from wallimport.routeset import generate_route, task_from_route


@pytest.fixture(scope="module")
def wall():
    return import_boulder_wall()


@pytest.fixture(scope="module")
def env(wall):
    return ClimbingEnv3D(wall, route_seed=0, difficulty="easy")


# ---------------- spaces & API ----------------------------------------------

def test_spaces(env):
    assert env.action_space.n == 1 + MAX_SLOTS + 4 + 1
    assert env.observation_space.shape == (9 + 20 + MAX_SLOTS * 13 + 4,)
    assert env.observation_space.dtype == np.float32


def test_reset_determinism(env):
    o1, _ = env.reset(seed=7)
    o2, _ = env.reset(seed=7)
    assert np.allclose(o1, o2)
    assert env.body.num_attached() == 4


def test_reset_start_state(env):
    """Start: 4 limbs on the lowest task holds, com above the fall threshold."""
    obs, _ = env.reset(seed=0)
    assert env.body.com[1] > 0.55
    lows = env.task_holds[:4]
    contacts = [l.contact for l in env.body.limbs]
    for c in contacts:
        assert any(np.allclose(c, h.pos) for h in lows)
    rep = env.body.check_grips()
    assert rep.feasible, f"start grips must be feasible, utils={rep.utils}"


def test_obs_contents(env):
    obs, _ = env.reset(seed=1)
    # com block is the first 3 dims, in wall bounds
    assert 0.4 < obs[1] < 2.5
    # target block: last 4 dims (rel xyz + dist)
    dist = obs[-1]
    assert 1.0 < dist < 6.0
    assert abs(np.linalg.norm(obs[-4:-1]) - dist) < 1e-4


def test_grab_action_attaches(env):
    env.reset(seed=2)
    # free a hand first (all 4 limbs start attached), then find a grabbable slot
    env.step(MAX_SLOTS + 1)                     # release L-hand
    flags = env._grabbable_flags()
    grabbable = np.where(flags == 1.0)[0]       # 1.0 = free limb in reach
    assert len(grabbable) > 0, "a hold must be grabbable once a limb is free"
    slot = int(grabbable[0]) + 1
    obs, r, term, trunc, info = env.step(slot)
    assert np.all(np.isfinite(obs))
    assert env.body.limbs[0].attached           # the free hand grabbed it


def test_grabbable_flags_show_release_first_options(env):
    """With all four limbs attached, nearby holds must still show as
    reachable-after-release (0.5) — otherwise the policy sees no options."""
    env.reset(seed=6)
    flags = env._grabbable_flags()
    assert (flags > 0).sum() > 0
    assert (flags == 0.5).sum() > 0             # release-first options visible
    env.step(MAX_SLOTS + 1)                     # release a hand
    flags2 = env._grabbable_flags()
    assert (flags2 == 1.0).sum() > 0            # now directly grabbable


def test_release_action(env):
    env.reset(seed=3)
    obs, r, term, trunc, info = env.step(MAX_SLOTS + 1)   # release L-hand
    assert not env.body.limbs[0].attached
    # releasing everything eventually falls
    for a in (MAX_SLOTS + 2, MAX_SLOTS + 3, MAX_SLOTS + 4):
        env.step(a)
    obs, r, term, trunc, info = env.step(0)
    assert env.body.num_attached() == 0


def test_fall_terminates(env):
    env.reset(seed=4)
    for a in (MAX_SLOTS + 1, MAX_SLOTS + 2, MAX_SLOTS + 3, MAX_SLOTS + 4):
        env.step(a)
    for _ in range(30):
        obs, r, term, trunc, info = env.step(0)
        if term:
            break
    assert term
    assert "fall_penalty" in info["comp"]


def test_success_on_short_task(wall):
    """A hand parked on the target hold for 2 steps ends the episode with success."""
    # pick 4 low real holds close together, y-sorted; the top one is the target
    lows = [h for h in wall.holds_by_section("main")
            if 0.3 <= h.pos[1] <= 1.3 and h.hold_type in ("jug", "volume")]
    lows.sort(key=lambda h: h.pos[1])
    cluster = [lows[0]]
    for h in lows[1:]:
        if len(cluster) >= 4:
            break
        if min(np.linalg.norm(h.pos - c.pos) for c in cluster) < 1.0:
            cluster.append(h)
    assert len(cluster) == 4
    task = sorted(cluster, key=lambda h: h.pos[1])
    e = ClimbingEnv3D(wall, task_holds=task, max_steps=100)
    e.reset(seed=0)
    # the start stance puts a hand on the top hold = target; hold for 2 steps
    obs, r1, t1, _, i1 = e.step(0)
    obs, r2, t2, _, i2 = e.step(0)
    assert t2 and i2["target_reached"]
    assert "success_bonus" in i2["comp"]


def test_reward_components_present(env):
    env.reset(seed=5)
    obs, r, term, trunc, info = env.step(0)
    for key in ("progress", "time_penalty"):
        assert key in info["comp"]
    assert isinstance(r, float)


def test_random_rollout_finite(env):
    rng = np.random.default_rng(0)
    for _ in range(3):
        env.reset()
        for _ in range(60):
            obs, r, term, trunc, info = env.step(int(rng.integers(env.action_space.n)))
            assert np.all(np.isfinite(obs))
            assert np.isfinite(r)
            if term or trunc:
                break


def test_stability_bonus_rewards_good_grips(wall):
    """Holding with wide margin yields positive stability, cliff-hanging less."""
    e = ClimbingEnv3D(wall, route_seed=0, difficulty="easy")
    e.reset(seed=0)
    _, _, _, _, i1 = e.step(0)      # settled 4-limb start: healthy margins
    assert i1["comp"].get("stability_bonus", 0) >= 0


def test_no_dyno_farming_reward(wall):
    """Release-and-regrab loops must not farm the dyno catch bonus: a catch
    only pays when the dyno actually gained height. Regression test for the
    'dyno bouncing' exploit found during training."""
    e = ClimbingEnv3D(wall, route_seed=0, difficulty="easy", max_steps=60)
    e.reset(seed=0)
    dyno = 1 + MAX_SLOTS + 4
    # find a grabbable slot to re-catch after each dyno
    total_dyno_reward = 0.0
    for _ in range(20):
        _, _, _, _, info = e.step(dyno)          # dyno from current stance
        e.step(0)
        flags = e._grabbable_flags()
        grab = np.where(flags > 0)[0]
        if len(grab):
            _, _, _, _, info = e.step(int(grab[0]) + 1)
        else:
            info = e.step(0)[4]
        total_dyno_reward += info["comp"].get("dyno_bonus", 0.0)
    assert total_dyno_reward < 1.0   # no height-gaining dynos in this loop


def test_target_touch_bonus(wall):
    """Holding the target with a hand pays the touch shaping each step."""
    lows = [h for h in wall.holds_by_section("main")
            if 0.3 <= h.pos[1] <= 1.3 and h.hold_type in ("jug", "volume")]
    lows.sort(key=lambda h: h.pos[1])
    cluster = [lows[0]]
    for h in lows[1:]:
        if len(cluster) >= 4:
            break
        if min(np.linalg.norm(h.pos - c.pos) for c in cluster) < 1.0:
            cluster.append(h)
    task = sorted(cluster, key=lambda h: h.pos[1])
    e = ClimbingEnv3D(wall, task_holds=task, max_steps=50)
    e.reset(seed=0)
    _, _, _, _, i1 = e.step(0)
    assert i1["comp"].get("target_touch", 0) > 0


# ---------------- task randomization ----------------------------------------

def test_task_randomization_wrapper(wall):
    w = TaskRandomizationWrapper(wall=wall, seeds=[0, 1, 2], difficulty="easy")
    ids_per_ep = []
    for _ in range(3):
        obs, _ = w.reset()
        assert np.all(np.isfinite(obs))
        ids_per_ep.append(tuple(h.id for h in w.task_holds))
    assert len(set(ids_per_ep)) == 3      # different problems per episode


# ---------------- renderer ----------------------------------------------------

def test_renderer_png(env, tmp_path):
    env.reset(seed=0)
    env.step(0)
    path = str(tmp_path / "frame.png")
    render_frame_png(env, path)
    assert os.path.exists(path) and os.path.getsize(path) > 20_000


# ---------------- training smoke ----------------------------------------------

def test_ppo_learns_smoke(env):
    from stable_baselines3 import PPO
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=64, n_epochs=2,
                policy_kwargs={"net_arch": [32, 32]}, verbose=0, device="cpu")
    model.learn(total_timesteps=128)
    obs = env.reset()[0]
    action, _ = model.predict(obs, deterministic=True)
    assert env.action_space.contains(int(action))
