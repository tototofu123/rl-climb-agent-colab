"""Stage 2+3: PPO training pipeline with configurable reward."""

import argparse
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from env.climbing_env import PhysicsClimbingEnv, SimpleClimbingEnv
from env.wall import Wall
from experiments.configs import get_config, ALL as ALL_CONFIGS


@dataclass
class TrainConfig:
    total_timesteps: int = 200_000
    n_envs: int = 4
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    max_steps_per_episode: int = 300
    seed: int = 42
    log_dir: str = "experiments/logs"
    model_dir: str = "agents"
    eval_episodes: int = 10
    eval_freq: int = 10_000
    save_freq: int = 25_000
    use_physics: bool = True
    wall_layout: str = "beginner"
    reward_config: Optional[dict] = None


WALL_REGISTRY: dict[str, Wall] = {
    "beginner": Wall.beginner(),
    "intermediate": Wall.intermediate(),
    "dyno_gap": Wall.dyno_gap(),
}


def make_env(config: TrainConfig):
    wall = WALL_REGISTRY.get(config.wall_layout, Wall.beginner())
    def _init():
        if config.use_physics:
            return PhysicsClimbingEnv(
                wall=wall,
                max_steps=config.max_steps_per_episode,
                reward_config=config.reward_config,
            )
        return SimpleClimbingEnv(
            wall=wall,
            max_steps=config.max_steps_per_episode,
            reward_config=config.reward_config,
        )
    return _init


def train(config: TrainConfig):
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.model_dir, exist_ok=True)

    rc_name = "default"
    if config.reward_config:
        for name, cfg in ALL_CONFIGS.items():
            if cfg == config.reward_config:
                rc_name = name
                break

    run_name = f"ppo_{config.wall_layout}_{rc_name}_{int(time.time())}"
    tensorboard_log = os.path.join(config.log_dir, run_name)

    env_fn = make_env(config)
    vec_env = make_vec_env(env_fn, n_envs=config.n_envs, seed=config.seed)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        policy_kwargs={"net_arch": [64, 64]},
        verbose=1,
        tensorboard_log=tensorboard_log,
        seed=config.seed,
    )

    eval_env = make_vec_env(env_fn, n_envs=1, seed=config.seed + 1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=config.save_freq,
        save_path=config.model_dir,
        name_prefix=run_name,
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=config.model_dir,
        log_path=config.log_dir,
        eval_freq=config.eval_freq,
        n_eval_episodes=config.eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback])

    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    final_path = os.path.join(config.model_dir, f"{run_name}_final.zip")
    model.save(final_path)
    print(f"Model saved to {final_path}")

    stats_path = os.path.join(config.model_dir, f"{run_name}_final_stats.pkl")
    vec_env.save(stats_path)
    print(f"Normalization stats saved to {stats_path}")

    vec_env.close()
    eval_env.close()

    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train RL Climb agent with PPO")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wall", type=str, default="beginner", choices=["beginner", "intermediate", "dyno_gap"])
    parser.add_argument("--reward-config", type=str, default="hybrid", choices=list(ALL_CONFIGS.keys()))
    parser.add_argument("--no-physics", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=str, default="experiments/logs")
    parser.add_argument("--model-dir", type=str, default="agents")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = TrainConfig(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        learning_rate=args.lr,
        wall_layout=args.wall,
        use_physics=not args.no_physics,
        seed=args.seed,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        reward_config=get_config(args.reward_config),
    )
    train(config)
