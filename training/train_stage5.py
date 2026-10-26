"""Stage 5: PPO training with curriculum learning and domain randomization."""

import argparse
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.climbing_env import PhysicsClimbingEnv
from env.wall import Wall
from env.wrappers import CurriculumWrapper, DomainRandomizationWrapper
from experiments.configs import get_config, ALL as ALL_CONFIGS

MAX_HOLDS = 12  # max across all walls

WALLS = [
    ("beginner", Wall.beginner()),
    ("intermediate", Wall.intermediate()),
    ("dyno_gap", Wall.dyno_gap()),
    ("expert", Wall.expert()),
]

THRESHOLDS = [200.0, 250.0, 150.0]  # reward thresholds to promote


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
    wall_layout: str = "beginner"
    reward_config: Optional[dict] = None
    curriculum: bool = False
    domain_rand: bool = False
    domain_rand_offset: float = 15.0


def make_single_env(config, wall=None, wrap_curriculum=False, wrap_domain_rand=False):
    """Create a single env with optional curriculum + domain randomization."""
    wall = wall or Wall.beginner()

    def _init():
        env = PhysicsClimbingEnv(
            wall=wall,
            max_steps=config.max_steps_per_episode,
            reward_config=config.reward_config,
            max_holds=13,
        )
        if wrap_domain_rand:
            env = DomainRandomizationWrapper(env, max_offset=config.domain_rand_offset, seed=config.seed)
        if wrap_curriculum:
            env = CurriculumWrapper(env, WALLS, THRESHOLDS)
        return env
    return _init


def make_env(config):
    wall_name = config.wall_layout
    wall = Wall.beginner()
    for name, w in WALLS:
        if name == wall_name:
            wall = w
            break

    def _init():
        env = PhysicsClimbingEnv(
            wall=wall,
            max_steps=config.max_steps_per_episode,
            reward_config=config.reward_config,
            max_holds=13,  # Fixed action space for curriculum compatibility
        )
        if config.domain_rand:
            env = DomainRandomizationWrapper(env, max_offset=config.domain_rand_offset, seed=config.seed)
        if config.curriculum:
            env = CurriculumWrapper(env, WALLS, THRESHOLDS)
        return env
    return _init


def train(config):
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.model_dir, exist_ok=True)

    rc_name = "default"
    if config.reward_config:
        for name, cfg in ALL_CONFIGS.items():
            if cfg == config.reward_config:
                rc_name = name
                break

    mode = "curriculum" if config.curriculum else config.wall_layout
    if config.domain_rand:
        mode += "_rand"
    run_name = f"ppo_{mode}_{rc_name}_{int(time.time())}"
    tensorboard_log = os.path.join(config.log_dir, run_name)

    env_fn = make_env(config)
    vec_env = DummyVecEnv([env_fn for _ in range(config.n_envs)])
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

    eval_fn = make_single_env(config, wrap_domain_rand=False, wrap_curriculum=False)
    eval_env = DummyVecEnv([eval_fn])
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
    parser.add_argument("--wall", type=str, default="beginner",
                        choices=["beginner", "intermediate", "dyno_gap", "expert"])
    parser.add_argument("--reward-config", type=str, default="hybrid",
                        choices=list(ALL_CONFIGS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=str, default="experiments/logs")
    parser.add_argument("--model-dir", type=str, default="agents")
    parser.add_argument("--curriculum", action="store_true",
                        help="Train with curriculum: beginner → intermediate → dyno_gap → expert")
    parser.add_argument("--domain-rand", action="store_true",
                        help="Enable domain randomization (hold position variance)")
    parser.add_argument("--domain-rand-offset", type=float, default=15.0,
                        help="Max hold position offset for domain randomization")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = TrainConfig(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        learning_rate=args.lr,
        wall_layout=args.wall,
        seed=args.seed,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        reward_config=get_config(args.reward_config),
        curriculum=args.curriculum,
        domain_rand=args.domain_rand,
        domain_rand_offset=args.domain_rand_offset,
    )
    train(config)
