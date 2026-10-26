"""Evaluate a policy on the FULL long toprope route (no curriculum truncation).

Same eval pattern as eval_3d.py but the task is the complete long route:

    python3.12 training/eval_long_route.py \
        --model agents/ppo3d_toprope_long_easy_*_600000_steps.zip
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
from wallimport.importer import import_wall
from wallimport.routeset import generate_long_route, task_from_route


def evaluate(policy, env, n, seeds):
    rs, hs, lens, tops, holds = [], [], [], 0, []
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
        holds.append(len(getattr(env.body, "grabs", []) or []))
    return dict(mean_reward=float(np.mean(rs)), max_height_avg=float(np.mean(hs)),
                max_height_best=float(np.max(hs)), ep_len=float(np.mean(lens)),
                tops=f"{tops}/{n}", n_env_holds=float(np.mean(holds)))


def make_env(wall, seed, n_holds, max_slots, max_steps):
    chain = generate_long_route(wall, seed=seed, n_holds=n_holds, difficulty="easy")
    task = task_from_route(wall, chain, max_slots=max_slots, extra_volumes=2)
    return ClimbingEnv3D(wall, task_holds=task, max_slots=max_slots,
                         max_steps=max_steps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--wall", default="toprope")
    p.add_argument("--route-seed", type=int, default=0)
    p.add_argument("--n-holds", type=int, default=26)
    p.add_argument("--max-slots", type=int, default=28)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--deterministic", action="store_true")
    args = p.parse_args()

    wall = import_wall(args.wall)
    env = make_env(wall, args.route_seed, args.n_holds, args.max_slots,
                   args.max_steps)

    model = PPO.load(args.model, device="cpu")

    # Train used VecNormalize; the frozen stats ship next to the model as
    # <stem>_final_stats.pkl / <stem>_vecnormalize_*_stats.pkl.
    stem = args.model[:-len(".zip")] if args.model.endswith(".zip") else args.model
    stem = re.sub(r"_\d+_steps$", "", stem)
    cands = [stem + "_final_stats.pkl"] + glob.glob(stem + "_*stats.pkl")
    stats = next((c for c in cands if os.path.exists(c)), None)

    if stats:
        raw_env = make_env(wall, args.route_seed, args.n_holds, args.max_slots,
                           args.max_steps)
        venv = VecNormalize.load(stats, VecNormalize(DummyVecEnv([lambda: raw_env])))
        venv.training = False
        venv.norm_reward = False
    else:
        venv = None
        print("WARNING: no VecNormalize stats found — raw obs (mismatch with train)")

    def _obs_fn(obs):
        if venv is not None:
            return venv.normalize_obs(np.array([obs], dtype=np.float32))[0]
        return obs

    seeds = np.arange(1000, 1000 + args.episodes).tolist()

    print(f"route: wall={args.wall} seed={args.route_seed} n_holds={args.n_holds} "
          f"slots={env.n_slots} top_y={env.target_hold.pos[1]:.2f} "
          f"stats={'yes' if stats else 'NO'}")

    rand = evaluate(lambda o: env.action_space.sample(), env, args.episodes, seeds)
    print(f"random                 : {rand}")

    res = evaluate(lambda o: int(model.predict(_obs_fn(o),
                                               deterministic=args.deterministic)[0]),
                   env, args.episodes, seeds)
    print(f"ppo({os.path.basename(args.model)[:40]}): {res}")


if __name__ == "__main__":
    main()
