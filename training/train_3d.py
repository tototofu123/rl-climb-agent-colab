"""Stage 6: PPO training on the 3D HKUST wall environment."""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from env.climbing_env_3d import ClimbingEnv3D
from env.wrappers3d import TaskRandomizationWrapper, LongRouteCurriculumWrapper
from wallimport.importer import import_wall
from wallimport.routeset import generate_route, generate_long_route, task_from_route


def make_env_fn(wall_name="boulder", route_seed=0, difficulty="medium",
                task_random=False, pool_size=8, max_steps=250,
                reward_config=None, long_route=False, max_slots=16,
                curriculum=False, top_y=None):
    wall = import_wall(wall_name)

    def _task(seed):
        if long_route:
            chain = generate_long_route(wall, seed=seed, difficulty=difficulty,
                                        top_y=top_y)
            return task_from_route(wall, chain, max_slots=max_slots, extra_volumes=2)
        return task_from_route(wall, generate_route(wall, seed=seed, difficulty=difficulty),
                               max_slots=max_slots)

    def _init():
        if long_route and curriculum:
            return LongRouteCurriculumWrapper(wall=wall, full_task=_task(route_seed),
                                              difficulty=difficulty,
                                              route_seed=route_seed,
                                              max_slots=max_slots, max_steps=max_steps,
                                              reward_config=reward_config)
        if task_random:
            return TaskRandomizationWrapper(wall=wall, difficulty=difficulty,
                                            pool_size=pool_size, max_steps=max_steps,
                                            reward_config=reward_config)
        return ClimbingEnv3D(wall, task_holds=_task(route_seed), max_slots=max_slots,
                             max_steps=max_steps, reward_config=reward_config)
    return _init


def train(args):
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    run_name = (f"ppo3d_{args.wall}{'_long' if args.long_route else ''}"
                f"_{args.difficulty}{'_rand' if args.task_random else ''}_{int(time.time())}")
    tb_log = os.path.join(args.log_dir, run_name)

    env_fn = make_env_fn(args.wall, args.route_seed, args.difficulty, args.task_random,
                         args.pool_size, args.max_steps, long_route=args.long_route,
                         max_slots=args.max_slots, curriculum=args.curriculum,
                         top_y=args.top_y)
    vec_env = make_vec_env(env_fn, n_envs=args.n_envs, seed=args.seed)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    if args.resume:
        # Continue training from a checkpoint: fresh VecNormalize (stats from the
        # resumed run are re-estimated), policy weights carried over.
        model = PPO.load(args.resume, vec_env, device="cpu")
        print(f"resumed policy from {args.resume}")
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs={"net_arch": [256, 256]},
            verbose=1,
            tensorboard_log=tb_log,
            seed=args.seed,
        )

    eval_env = make_vec_env(
        make_env_fn(args.wall, args.route_seed, args.difficulty, False,
                    max_steps=args.max_steps, long_route=args.long_route,
                    max_slots=args.max_slots, top_y=args.top_y),
        n_envs=1, seed=args.seed + 1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)

    callbacks = CallbackList([
        CheckpointCallback(save_freq=args.save_freq, save_path=args.model_dir,
                           name_prefix=run_name, save_vecnormalize=True),
        EvalCallback(eval_env, best_model_save_path=args.model_dir,
                     log_path=args.log_dir, eval_freq=args.eval_freq,
                     n_eval_episodes=10, deterministic=True, verbose=1),
    ])

    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=True)

    final = os.path.join(args.model_dir, f"{run_name}_final.zip")
    model.save(final)
    vec_env.save(os.path.join(args.model_dir, f"{run_name}_final_stats.pkl"))
    print(f"saved {final}")
    vec_env.close()
    eval_env.close()
    return model


def parse_args():
    p = argparse.ArgumentParser(description="Train 3D climbing PPO agent")
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard"])
    p.add_argument("--wall", default="boulder", choices=["boulder", "toprope"])
    p.add_argument("--long-route", action="store_true",
                   help="generate a long DAG route (for the tall toprope wall)")
    p.add_argument("--curriculum", action="store_true",
                   help="length curriculum on the long route (self-tuning prefixes)")
    p.add_argument("--max-slots", type=int, default=16)
    p.add_argument("--route-seed", type=int, default=0)
    p.add_argument("--task-random", action="store_true",
                   help="sample a new boulder problem each episode (pool)")
    p.add_argument("--pool-size", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--top-y", type=float, default=None,
                   help="long-route ceiling in metres (default: wall height - 0.9); "
                        "lower it to train an honest shorter full route first")
    p.add_argument("--resume", default=None,
                   help="checkpoint to continue training from (policy weights only)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-freq", type=int, default=50_000)
    p.add_argument("--eval-freq", type=int, default=20_000)
    p.add_argument("--log-dir", default="experiments/logs")
    p.add_argument("--model-dir", default="agents")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
