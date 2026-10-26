"""Stage 6 demo: run a random or trained agent on the 3D HKUST wall.

    python3.12 run_stage6.py                          # random agent, live window
    python3.12 run_stage6.py --model agents/<zip>     # trained agent
    python3.12 run_stage6.py --headless --episodes 3  # no window, saves PNGs
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.climbing_env_3d import ClimbingEnv3D
from env.wrappers3d import TaskRandomizationWrapper
from render.renderer3d import Renderer3D
from wallimport.importer import import_wall
from wallimport.routeset import generate_long_route, task_from_route


def load_model(path, env):
    from stable_baselines3 import PPO
    stats = path.replace(".zip", "_stats.pkl")
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
    venv = DummyVecEnv([lambda: env])
    if os.path.exists(stats):
        venv = VecNormalize.load(stats, venv)
        venv.training = False
        venv.norm_reward = False
        model = PPO.load(path, env=venv)
        return model, venv
    model = PPO.load(path, env=env)
    return model, None


def run(args):
    wall = import_wall(args.wall)
    max_slots = args.max_slots or (24 if args.long_route else 16)
    if args.long_route:
        chain = generate_long_route(wall, seed=args.route_seed, difficulty=args.difficulty)
        task = task_from_route(wall, chain, max_slots=max_slots)
        env = ClimbingEnv3D(wall, task_holds=task, max_slots=max_slots,
                            max_steps=args.max_steps)
        task_name = f"long {args.difficulty} route ({len(task)} holds)"
    elif args.task_random:
        env = TaskRandomizationWrapper(wall=wall, difficulty=args.difficulty,
                                       pool_size=args.pool_size, max_steps=args.max_steps)
        task_name = f"pool[{args.pool_size}] {args.difficulty}"
    else:
        env = ClimbingEnv3D(wall, route_seed=args.route_seed, difficulty=args.difficulty,
                            max_steps=args.max_steps)
        task_name = f"seed {args.route_seed} {args.difficulty}"
    env.task_label = task_name

    model = venv = None
    if args.model:
        model, venv = load_model(args.model, env)

    renderer = None
    if not args.headless:
        renderer = Renderer3D()

    results = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total_r, steps = 0.0, 0
        while not done and steps < args.max_steps:
            if model is not None:
                if venv is not None:
                    batch = venv.normalize_obs(obs[None])
                    action, _ = model.predict(batch, deterministic=True)
                    action = int(np.asarray(action).reshape(-1)[0])
                else:
                    action, _ = model.predict(obs, deterministic=True)
                    action = int(np.asarray(action).reshape(-1)[0])
            else:
                action = env.action_space.sample()
            obs, r, term, trunc, info = env.step(action)
            total_r += r
            steps += 1
            done = term or trunc

            if renderer is not None:
                renderer.render(env, extra_hud=[
                    f"ep {ep + 1}/{args.episodes}  R={total_r:+.1f}"
                    + ("  [PPO]" if model else "  [random]"),
                ])
                renderer.clock.tick(30)
                for evt in renderer.pygame.event.get():
                    if evt.type == renderer.pygame.QUIT:
                        renderer.quit()
                        return
            if args.headless and args.save_pngs and steps % 25 == 0:
                os.makedirs("experiments/frames", exist_ok=True)
                from render.renderer3d import render_frame_png
                render_frame_png(env, f"experiments/frames/ep{ep}_s{steps:03d}.png")

        results.append((total_r, steps, info.get("target_reached", False), info.get("height", 0)))
        print(f"ep {ep + 1}: R={total_r:+.1f} steps={steps} "
              f"top={info.get('target_reached')} height={info.get('height', 0):.2f}m")

    if renderer:
        renderer.quit()

    succ = sum(1 for r in results if r[2])
    print(f"\n{succ}/{len(results)} episodes reached the top hold")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None, help="path to trained PPO .zip")
    p.add_argument("--difficulty", default="easy", choices=["easy", "medium", "hard"])
    p.add_argument("--wall", default="boulder", choices=["boulder", "toprope"])
    p.add_argument("--long-route", action="store_true")
    p.add_argument("--max-slots", type=int, default=None)
    p.add_argument("--route-seed", type=int, default=0)
    p.add_argument("--task-random", action="store_true")
    p.add_argument("--pool-size", type=int, default=8)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--save-pngs", action="store_true")
    run(p.parse_args())
