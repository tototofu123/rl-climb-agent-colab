"""Evaluate a 3D climbing policy: reward, max height, tops. Random baseline included.

    python3.12 training/eval_3d.py --model agents/ppo3d_easy_..._final.zip
    python3.12 training/eval_3d.py --model ... --difficulty medium --route-seed 3
"""

import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.climbing_env_3d import ClimbingEnv3D
from wallimport.importer import import_boulder_wall


def evaluate(policy, env, n, seeds):
    rs, hs, lens, tops = [], [], [], 0
    for i in range(n):
        obs, _ = env.reset(seed=seeds[i])
        done = False
        R = maxh = steps = 0
        while not done:
            obs, r, term, trunc, info = env.step(policy(obs))
            R += r
            maxh = max(maxh, info["height"])
            steps += 1
            done = term or trunc
        rs.append(R)
        hs.append(maxh)
        lens.append(steps)
        tops += bool(info.get("target_reached"))
    return dict(mean_reward=float(np.mean(rs)), max_height_avg=float(np.mean(hs)),
                max_height_best=float(np.max(hs)), ep_len=float(np.mean(lens)),
                tops=f"{tops}/{n}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None)
    p.add_argument("--difficulty", default="easy")
    p.add_argument("--route-seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--deterministic", action="store_true")
    args = p.parse_args()

    env = ClimbingEnv3D(import_boulder_wall(), route_seed=args.route_seed,
                        difficulty=args.difficulty)
    seeds = np.arange(1000, 1000 + args.episodes).tolist()

    rand = evaluate(lambda o: env.action_space.sample(), env, args.episodes, seeds)
    print(f"random    : {rand}")

    if args.model:
        model = PPO.load(args.model, device="cpu")
        det = args.deterministic
        # Training wraps the env in VecNormalize; the frozen stats ship beside
        # the model. Predicting on raw observations starves the policy.
        stem = args.model[:-len(".zip")] if args.model.endswith(".zip") else args.model
        stem = re.sub(r"_\d+_steps$", "", stem)
        cands = [stem + "_final_stats.pkl"] + sorted(glob.glob(stem + "_*stats.pkl"))
        stats = next((c for c in cands if os.path.exists(c)), None)
        if stats:
            venv = VecNormalize.load(stats, DummyVecEnv([lambda: ClimbingEnv3D(
                import_boulder_wall(), route_seed=args.route_seed,
                difficulty=args.difficulty)]))
            venv.training = False
            venv.norm_reward = False
            obs_fn = lambda o: venv.normalize_obs(np.array([o], dtype=np.float32))[0]
        else:
            print(f"WARNING: no VecNormalize stats for {args.model}")
            obs_fn = lambda o: o
        res = evaluate(
            lambda o: int(model.predict(obs_fn(o), deterministic=det)[0]),
            env, args.episodes, seeds)
        print(f"ppo({os.path.basename(args.model)[:40]}): {res}")


if __name__ == "__main__":
    main()
