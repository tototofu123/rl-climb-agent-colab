"""Stage 0: random agent on SimpleClimbingEnv."""

from env.climbing_env import SimpleClimbingEnv
from render.renderer import Renderer


def main():
    env = SimpleClimbingEnv()
    renderer = Renderer(env.wall, title="RL Climb — Stage 0: Simple Movement")
    episode = 0

    while not renderer.closed:
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done and not renderer.closed:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            renderer.render_simple(env)

        if not renderer.closed:
            print(f"Episode {episode}: reward={total_reward:7.2f}  steps={env.step_count:3d}")
            episode += 1

    renderer.close()
    print("Done.")


if __name__ == "__main__":
    main()
