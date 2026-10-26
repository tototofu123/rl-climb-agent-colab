"""Record a REAL demonstration video of a trained 3D climbing agent.

Nothing here is scripted or faked: the policy network is rolled out in the
environment step by step with model.predict(), every frame is rendered from
the resulting simulator state, and episodes are only selected (never edited) —
the script prints what happened in each recorded episode. Frames are copied
and streamed to disk immediately (the renderer reuses one surface), then
encoded with ffmpeg.

    python3.12 render/make_demo_video.py --model agents/ppo3d_easy_..._final.zip
    python3.12 render/make_demo_video.py --model ... --episode-count 2 --include-fall
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.climbing_env_3d import ClimbingEnv3D
from render.renderer3d import Renderer3D
from wallimport.importer import import_wall
from wallimport.routeset import generate_long_route, task_from_route

FPS = 30
FRAMES_PER_STEP = 10         # env dt is 0.1 s; 10 frames/step = slow-motion
                             # (long climbs are only ~15-20 env steps, so 3
                             # frames/step yields a sub-second clip)


class FrameWriter:
    """Streams frame snapshots to a temp dir; encodes with ffmpeg on close."""

    def __init__(self, renderer, tmp):
        self.renderer = renderer
        self.tmp = tmp
        self.n = 0

    def emit(self, surf, times=1):
        snap = surf.copy()               # renderer reuses its screen surface
        for k in range(times):
            self.renderer.pygame.image.save(snap, os.path.join(self.tmp, f"f_{self.n:05d}.png"))
            self.n += 1

    def encode(self, out):
        cmd = ["ffmpeg", "-y", "-framerate", str(FPS),
               "-i", os.path.join(self.tmp, "f_%05d.png"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out]
        subprocess.run(cmd, check=True, capture_output=True)
        return self.n


def find_episode(env, model, venv, seed_start, want_top, max_tries=150):
    """Roll out the real policy until an episode with the wanted outcome.

    Returns (seed, actions) where actions are the policy's own sampled
    actions; recording replays this exact action sequence so the video shows
    precisely the episode that was found.
    """
    for seed in range(seed_start, seed_start + max_tries):
        obs, _ = env.reset(seed=seed)
        done = False
        actions, topped = [], False
        while not done:
            batch = venv.normalize_obs(obs[None]) if venv is not None else obs[None]
            action, _ = model.predict(batch, deterministic=False)
            a = int(np.asarray(action).reshape(-1)[0])
            actions.append(a)
            obs, r, term, trunc, info = env.step(a)
            topped = bool(info.get("target_reached"))
            done = term or trunc
        if topped == want_top:
            return seed, actions
    return None, None


def record_episode(renderer, writer, env, seed, actions):
    """Replay the captured policy actions, rendering every simulator step."""
    env.reset(seed=seed)
    peak = 0.0
    for step, a in enumerate(actions, start=1):
        obs, r, term, trunc, info = env.step(a)
        peak = max(peak, info["height"])
        surf = renderer.render(env, extra_hud=[
            f"REAL POLICY ROLLOUT — seed {seed} — step {step}/{len(actions)}",
            "every frame rendered from the live simulator state",
        ])
        writer.emit(surf, FRAMES_PER_STEP)
    topped = bool(info.get("target_reached"))
    print(f"  episode seed {seed}: {len(actions)} steps, peak {peak:.2f} m, "
          f"topped={'YES' if topped else 'no'}")
    return topped


def title_card(renderer, writer, lines, hold_frames=45):
    surf = renderer.screen.copy()
    renderer.pygame.draw.rect(surf, (24, 26, 32), surf.get_rect())
    y = renderer.height * 0.3
    for i, line in enumerate(lines):
        font = renderer.font if i == 0 else renderer.small
        surf.blit(font.render(line, True, (235, 235, 235)),
                  (renderer.width * 0.12, y))
        y += 34
    writer.emit(surf, hold_frames)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--difficulty", default="easy")
    p.add_argument("--wall", default="boulder", choices=["boulder", "toprope"])
    p.add_argument("--long-route", action="store_true")
    p.add_argument("--route-seed", type=int, default=0)
    p.add_argument("--max-slots", type=int, default=None)
    p.add_argument("--top-y", type=float, default=None,
                   help="long-route ceiling in metres (must match training)")
    p.add_argument("--episode-count", type=int, default=2)
    p.add_argument("--include-fall", action="store_true")
    p.add_argument("--out", default="experiments/demo/ppo3d_climb_demo.mp4")
    args = p.parse_args()

    wall = import_wall(args.wall)
    if args.long_route:
        chain = generate_long_route(wall, seed=args.route_seed,
                                    difficulty=args.difficulty, top_y=args.top_y)
        task = task_from_route(wall, chain, max_slots=args.max_slots or 24)
        env = ClimbingEnv3D(wall, task_holds=task, max_slots=args.max_slots or 24,
                            max_steps=600)
        label = f"long {args.difficulty} route ({len(task)} holds, wall: {args.wall})"
    else:
        env = ClimbingEnv3D(wall, route_seed=args.route_seed, difficulty=args.difficulty,
                            max_slots=args.max_slots or 16)
        label = f"{args.difficulty} problem (route seed {args.route_seed})"
    env.task_label = label

    venv = None
    stats = args.model.replace(".zip", "_stats.pkl")
    if os.path.exists(stats):
        venv = VecNormalize.load(stats, DummyVecEnv([lambda: env]))
        venv.training = False
        venv.norm_reward = False
        model = PPO.load(args.model, env=venv, device="cpu")
    else:
        model = PPO.load(args.model, env=env, device="cpu")

    print(f"searching for real {args.episode_count} topping episode(s) of "
          f"{os.path.basename(args.model)} ...")
    episodes = []
    seed = 1000
    for k in range(args.episode_count):
        s, acts = find_episode(env, model, venv, seed, want_top=True)
        if s is None:
            print("  no topping episode found in search window; recording a raw attempt instead")
            s, acts = seed, []
            obs, _ = env.reset(seed=s)
            done = False
            while not done:
                batch = venv.normalize_obs(obs[None]) if venv is not None else obs[None]
                action, _ = model.predict(batch, deterministic=False)
                a = int(np.asarray(action).reshape(-1)[0])
                acts.append(a)
                obs, r, term, trunc, info = env.step(a)
                done = term or trunc
            episodes.append(("CLIMB ATTEMPT (no top found in search)", s, acts))
        else:
            episodes.append(("TOP CLIMB", s, acts))
        seed = s + 1
    if args.include_fall:
        s, acts = find_episode(env, model, venv, seed, want_top=False)
        if s is not None:
            episodes.append(("ATTEMPT (no top)", s, acts))

    renderer = Renderer3D(headless=True)
    tmp = tempfile.mkdtemp(prefix="climb_frames_")
    try:
        writer = FrameWriter(renderer, tmp)
        title_card(renderer, writer, [
            "RL Climb Agent — 3D HKUST boulder wall",
            f"policy: {os.path.basename(args.model)}",
            f"task: {label}",
            "all motion = real PPO rollouts, nothing scripted",
        ])
        for label, s, acts in episodes:
            title_card(renderer, writer, [label, f"episode seed {s}"], hold_frames=30)
            record_episode(renderer, writer, env, s, acts)

        out = args.out
        os.makedirs(os.path.dirname(out), exist_ok=True)
        n = writer.encode(out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        renderer.quit()
    print(f"\nvideo written: {out}  ({n} frames, {FPS} fps)")


if __name__ == "__main__":
    main()
