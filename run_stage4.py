"""Stage 4: random agent on dyno wall — see the dyno mechanic."""

from env.climbing_env import PhysicsClimbingEnv
from env.wall import Wall
from render.renderer import Renderer


def main():
    env = PhysicsClimbingEnv(wall=Wall.dyno_gap())
    renderer = Renderer(env.wall, title="RL Climb — Stage 4: Dyno Mechanics")
    episode = 0

    while not renderer.closed:
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done and not renderer.closed:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            renderer.render_physics(env)

        if not renderer.closed:
            print(f"Episode {episode}: reward={total_reward:+7.2f}  steps={env.step_count:3d}")
            episode += 1

    renderer.close()
    print("Done.")


if __name__ == "__main__":
    main()
