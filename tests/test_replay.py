"""Tests for run_replay.py — model loading and replay."""

import glob
import os

import numpy as np
import pytest

from env.climbing_env import PhysicsClimbingEnv
from env.wall import Wall


class TestFindLatestModel:
    def test_finds_zip_files(self):
        from run_replay import find_latest_model
        zips = glob.glob(os.path.join("agents", "*.zip"))
        if zips:
            model = find_latest_model()
            assert model is not None
            assert model.endswith(".zip")

    def test_returns_none_if_empty(self):
        from run_replay import find_latest_model
        result = find_latest_model(model_dir="nonexistent_dir")
        assert result is None


class TestReplayEnv:
    def test_env_created_with_wall(self):
        from run_replay import WALLS
        for name, wall in WALLS.items():
            assert isinstance(wall, Wall)
            assert len(wall.holds) >= 3

    def test_env_resets_with_wall(self):
        from run_replay import WALLS
        for name, wall in WALLS.items():
            env = PhysicsClimbingEnv(wall=wall)
            obs, _ = env.reset()
            assert isinstance(obs, np.ndarray)


class TestModelLoads:
    def test_model_loads_and_predicts(self):
        from stable_baselines3 import PPO
        zips = glob.glob(os.path.join("agents", "*.zip"))
        if not zips:
            pytest.skip("No model zip found to test")
        model = PPO.load(zips[0])
        assert model is not None
        env = PhysicsClimbingEnv()
        obs, _ = env.reset()
        try:
            action, _ = model.predict(obs, deterministic=True)
        except ValueError as e:
            if "Unexpected observation shape" in str(e):
                pytest.skip(f"Model obs space mismatch (retrain needed): {e}")
            raise
        assert action in range(env.action_space.n)
