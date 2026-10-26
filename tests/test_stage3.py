"""TDD tests for Stage 3 — Reward Engineering."""

import os
import shutil
import tempfile

import numpy as np
import pytest

from env.climbing_env import PhysicsClimbingEnv, DEFAULT_REWARD_CONFIG
from experiments.configs import get_config, ALL


class TestConfigs:
    def test_all_configs_have_required_keys(self):
        required = {"progress", "hold_bonus", "time_penalty", "reach_penalty", "fall_penalty", "success_bonus"}
        for name, cfg in ALL.items():
            assert set(cfg.keys()) == required, f"{name} missing keys"

    def test_get_config_returns_copy(self):
        c1 = get_config("dense")
        c2 = get_config("dense")
        assert c1 == c2
        assert c1 is not c2

    def test_unknown_config_raises(self):
        with pytest.raises(ValueError):
            get_config("nonexistent")

    def test_dense_has_only_progress_and_terminals(self):
        cfg = get_config("dense")
        assert cfg["progress"] != 0
        assert cfg["hold_bonus"] == 0
        assert cfg["time_penalty"] == 0
        assert cfg["reach_penalty"] == 0

    def test_sparse_has_no_shaping(self):
        cfg = get_config("sparse")
        assert cfg["progress"] == 0
        assert cfg["hold_bonus"] == 0
        assert cfg["time_penalty"] == 0
        assert cfg["reach_penalty"] == 0


class TestEnvWithConfig:
    def test_default_config_matches_default(self):
        env = PhysicsClimbingEnv()
        assert env.rc == DEFAULT_REWARD_CONFIG

    def test_accepts_custom_config(self):
        cfg = get_config("sparse")
        env = PhysicsClimbingEnv(reward_config=cfg)
        assert env.rc["success_bonus"] == 5.0
        assert env.rc["progress"] == 0.0

    def test_info_contains_reward_components(self):
        env = PhysicsClimbingEnv()
        env.reset()
        _, _, _, _, info = env.step(0)
        assert "comp" in info
        assert isinstance(info["comp"], dict)

    def test_components_sum_to_reward(self):
        env = PhysicsClimbingEnv()
        env.reset()
        obs, r, _, _, info = env.step(1)
        assert abs(sum(info["comp"].values()) - r) < 1e-6

    def test_dense_gives_no_hold_bonus(self):
        env = PhysicsClimbingEnv(reward_config=get_config("dense"))
        env.reset()
        n_holds = len(env.wall.holds)
        env.step(n_holds)  # reach nearest hold (last one is closest to start)
        _, _, _, _, info = env.step(0)
        # hold_bonus should be 0 since dense has hold_bonus=0
        comp = info["comp"]
        assert comp.get("hold_bonus", 0) == 0 or "hold_bonus" not in comp

    def test_sparse_gives_no_progress_reward(self):
        env = PhysicsClimbingEnv(reward_config=get_config("sparse"))
        env.reset()
        env.step(0)
        _, _, _, _, info = env.step(0)
        # In sparse, progress weight is 0
        # reward should only be time_penalty (which is 0 in sparse too)
        assert abs(info["comp"].get("progress", 0)) < 1e-6

    def test_progress_tracks_distance_change(self):
        env = PhysicsClimbingEnv()
        env.reset()
        prev_dist = env._prev_dist
        # Move body closer to target by setting pos
        env.body.pos = np.array([390.0, 80.0], dtype=np.float32)
        _, reward, _, _, info = env.step(0)
        # Should have positive progress
        assert info["comp"].get("progress", 0) >= -1e-6, f"Expected non-negative progress, got {info['comp']}"


class TestTrainingWithConfig:
    def test_training_with_sparse_config(self, tmp_path):
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from training.train import TrainConfig, make_env, train

        config = TrainConfig(
            total_timesteps=500,
            n_envs=1,
            log_dir=str(tmp_path / "logs"),
            model_dir=str(tmp_path / "agents"),
            eval_freq=1000,
            save_freq=1000,
            reward_config=get_config("sparse"),
        )
        model = train(config)
        assert isinstance(model, PPO)

    def test_training_with_dense_config(self, tmp_path):
        from stable_baselines3 import PPO
        from training.train import TrainConfig, train

        config = TrainConfig(
            total_timesteps=500,
            n_envs=1,
            log_dir=str(tmp_path / "logs2"),
            model_dir=str(tmp_path / "agents2"),
            eval_freq=1000,
            save_freq=1000,
            reward_config=get_config("dense"),
        )
        model = train(config)
        assert isinstance(model, PPO)

    def test_training_with_multiple_configs_produces_separate_runs(self, tmp_path):
        from training.train import TrainConfig, train

        for name in ["dense", "sparse", "hybrid"]:
            config = TrainConfig(
                total_timesteps=300,
                n_envs=1,
                log_dir=str(tmp_path / "logs"),
                model_dir=str(tmp_path / name),
                eval_freq=1000,
                save_freq=1000,
                reward_config=get_config(name),
            )
            train(config)
        dirs = os.listdir(tmp_path)
        assert "dense" in dirs
        assert "sparse" in dirs
        assert "hybrid" in dirs
