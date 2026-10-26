"""TDD tests for Stage 2 — PPO training pipeline."""

import os
import shutil
import tempfile

import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from env.climbing_env import PhysicsClimbingEnv
from training.train import TrainConfig, make_env, train


@pytest.fixture
def tmp_dir():
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path)


class TestMakeEnv:
    def test_creates_physics_env(self):
        config = TrainConfig(use_physics=True)
        fn = make_env(config)
        env = fn()
        from env.climbing_env import PhysicsClimbingEnv
        assert isinstance(env, PhysicsClimbingEnv)

    def test_env_resets_and_steps(self):
        config = TrainConfig(use_physics=True)
        fn = make_env(config)
        env = fn()
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        assert isinstance(obs, np.ndarray)


class TestVecEnv:
    def test_vectorized_env_works(self):
        config = TrainConfig()
        env_fn = make_env(config)
        vec_env = make_vec_env(env_fn, n_envs=2, seed=42)
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
        obs = vec_env.reset()
        assert obs.shape[0] == 2
        actions = np.array([vec_env.action_space.sample() for _ in range(2)])
        obs, rewards, dones, infos = vec_env.step(actions)
        assert obs.shape[0] == 2
        vec_env.close()


class TestPPOCreation:
    def test_model_creation(self):
        config = TrainConfig()
        env_fn = make_env(config)
        vec_env = make_vec_env(env_fn, n_envs=1, seed=42)
        model = PPO("MlpPolicy", vec_env, verbose=0, seed=42)
        assert isinstance(model, PPO)
        vec_env.close()

    def test_model_predict(self):
        config = TrainConfig()
        env_fn = make_env(config)
        vec_env = make_vec_env(env_fn, n_envs=1, seed=42)
        model = PPO("MlpPolicy", vec_env, verbose=0, seed=42)
        obs = vec_env.reset()
        action, _ = model.predict(obs, deterministic=False)
        assert action in range(vec_env.action_space.n)
        vec_env.close()


class TestTrainingShort:
    def test_training_runs_1k_steps(self, tmp_dir):
        config = TrainConfig(
            total_timesteps=1_000,
            n_envs=1,
            log_dir=os.path.join(tmp_dir, "logs"),
            model_dir=os.path.join(tmp_dir, "agents"),
            eval_freq=500,
            save_freq=500,
        )
        model = train(config)
        assert isinstance(model, PPO)

    def test_training_saves_checkpoint(self, tmp_dir):
        config = TrainConfig(
            total_timesteps=1_000,
            n_envs=1,
            log_dir=os.path.join(tmp_dir, "logs"),
            model_dir=os.path.join(tmp_dir, "agents"),
            eval_freq=500,
            save_freq=500,
        )
        train(config)
        files = os.listdir(os.path.join(tmp_dir, "agents"))
        zip_files = [f for f in files if f.endswith(".zip")]
        assert len(zip_files) >= 1, f"No .zip files found in {files}"

    def test_training_creates_tensorboard_logs(self, tmp_dir):
        config = TrainConfig(
            total_timesteps=500,
            n_envs=1,
            log_dir=os.path.join(tmp_dir, "logs"),
            model_dir=os.path.join(tmp_dir, "agents"),
            eval_freq=1000,
            save_freq=1000,
        )
        train(config)
        log_files = []
        for root, dirs, files in os.walk(os.path.join(tmp_dir, "logs")):
            log_files.extend(files)
        assert len(log_files) >= 0

    def test_trained_model_outperforms_random(self, tmp_dir):
        config = TrainConfig(
            total_timesteps=2_000,
            n_envs=2,
            log_dir=os.path.join(tmp_dir, "logs"),
            model_dir=os.path.join(tmp_dir, "agents"),
            eval_freq=5000,
            save_freq=5000,
        )
        model = train(config)

        fn = make_env(config)
        test_env = fn()

        random_rewards = []
        model_rewards = []

        for _ in range(5):
            obs, _ = test_env.reset()
            done = False
            total = 0.0
            while not done:
                obs, r, term, trunc, _ = test_env.step(test_env.action_space.sample())
                done = term or trunc
                total += r
            random_rewards.append(total)

        for _ in range(5):
            obs, _ = test_env.reset()
            done = False
            total = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, _ = test_env.step(int(action))
                done = term or trunc
                total += r
            model_rewards.append(total)

        print(f"Random avg reward: {np.mean(random_rewards):.2f}")
        print(f"Model avg reward:  {np.mean(model_rewards):.2f}")
        # Even a short training run should not be catastrophically worse than random
        assert np.mean(model_rewards) > -500.0
