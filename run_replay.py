"""Load a trained model and visualize it in a Pygame window."""

import argparse
import glob
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.climbing_env import PhysicsClimbingEnv
from env.wall import Wall
from render.renderer import Renderer

WALLS = {
    "beginner": Wall.beginner(),
    "intermediate": Wall.intermediate(),
    "dyno_gap": Wall.dyno_gap(),
}


def find_latest_model(model_dir="agents"):
    zips = glob.glob(os.path.join(model_dir, "*.zip"))
    if not zips:
        print(f"No .zip models found in {model_dir}/")
        print("Train one first: python3.12 training/train.py --timesteps 50000")
        return None
    latest = max(zips, key=os.path.getmtime)
    return latest


def infer_wall_from_path(model_path):
    basename = os.path.basename(model_path).lower()
    for wall_name in WALLS:
        if wall_name in basename:
            return wall_name
    return None


def find_stats_for_model(model_path):
    base = model_path.replace(".zip", "")
    stats_path = base + "_stats.pkl"
    if os.path.exists(stats_path):
        return stats_path
    return None


def main():
    parser = argparse.ArgumentParser(description="Replay a trained model")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model .zip file (default: latest in agents/)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--wall", type=str, default=None,
                        choices=list(WALLS.keys()),
                        help="Wall layout (auto-detected from model filename if omitted)")
    args = parser.parse_args()

    model_path = args.model or find_latest_model()
    if model_path is None:
        return

    wall_name = args.wall or infer_wall_from_path(model_path) or "beginner"
    wall = WALLS[wall_name]
    stats_path = find_stats_for_model(model_path)

    print(f"Loading model: {model_path}")
    print(f"Wall: {wall_name}")
    print(f"VecNormalize stats: {stats_path or 'NONE (using raw obs)'}")

    model = PPO.load(model_path)

    def _make_env():
        return PhysicsClimbingEnv(wall=wall)

    env = DummyVecEnv([_make_env])
    if stats_path:
        env = VecNormalize.load(stats_path, env)
        env.training = False
        env.norm_reward = False

    renderer = Renderer(wall, title=f"RL Climb — Replay: {os.path.basename(model_path)}")
    episode = 0

    while not renderer.closed and episode < args.episodes:
        raw_obs = env.reset()
        obs = raw_obs
        done = False
        total = 0.0
        steps = 0
        info = {}

        while not done and not renderer.closed:
            action, _ = model.predict(obs, deterministic=True)
            raw_obs, r, term, trunc, info = env.step(action)
            obs = raw_obs
            done = term[0] or trunc[0]
            total += r[0]
            steps += 1
            renderer.render_physics(env.envs[0])

        if not renderer.closed:
            comp = info[0].get("comp", {})
            comp_str = "  [" + ", ".join(f"{k}={v:+.2f}" for k, v in comp.items()) + "]" if comp else ""
            print(f"Episode {episode}: reward={total:+.2f}  steps={steps}{comp_str}")
            episode += 1

    renderer.close()
    env.close()
    print("Replay done.")


if __name__ == "__main__":
    main()
