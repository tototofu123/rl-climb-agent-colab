"""Stage 6 tests: wall importer, routesetter, and 3D climber body physics."""

import math

import numpy as np
import pytest

from physics.body3d import ClimberBody3D, two_bone_ik
from wallimport.importer import (
    GymWall3D, Hold3D, import_boulder_wall, img_to_world, summarize,
    MAIN_WALL_W, WALL_H, LEFT_SIDE_DEPTH, RIGHT_SIDE_DEPTH,
)
from wallimport.routeset import generate_route, task_from_route


@pytest.fixture(scope="module")
def wall():
    return import_boulder_wall()


# ---------------- importer --------------------------------------------------

def test_import_counts(wall):
    assert len(wall.holds) == 417
    secs = {"main": 0, "left": 0, "right": 0}
    for h in wall.holds:
        secs[h.wall_section] += 1
    assert secs["main"] == 390
    assert secs["left"] + secs["right"] == 27


def test_hold_positions_in_bounds(wall):
    for h in wall.holds:
        x, y, z = h.pos
        assert -0.05 <= x <= MAIN_WALL_W + 0.35, h
        assert 0.0 <= y <= WALL_H + 0.01, h
        assert 0.0 <= z <= max(LEFT_SIDE_DEPTH, 0.35) + 0.01, h


def test_depths_by_type(wall):
    """Vision-calibrated depth ordering: volumes deepest, crimps shallowest."""
    by_type = {}
    for h in wall.holds:
        by_type.setdefault(h.hold_type, []).append(h.depth)
    assert np.median(by_type["volume"]) > np.median(by_type["jug"])
    assert np.median(by_type["jug"]) > np.median(by_type["crimp"])
    assert all(0.01 <= d <= 0.35 for d in by_type["volume"])


def test_pull_axes_are_unit_and_face_out(wall):
    for h in wall.holds:
        assert abs(np.linalg.norm(h.pull_axis) - 1.0) < 1e-6
        # out-of-plane component is non-negative (holds open toward the climber)
        assert np.dot(h.pull_axis, h.normal) >= -1e-9


def test_down_facing_hold_pulls_down(wall):
    """A hold annotated direction=180 (down) on the main wall must allow
    pulling downward — the classic hang."""
    downs = [h for h in wall.holds_by_section("main")
             if h.direction == 180 and h.hold_type == "jug"]
    assert downs
    h = downs[0]
    assert h.pull_axis[1] < -0.5          # mostly downward
    assert h.pull_axis[2] > 0.1           # tilted out of the wall (3D opening)


def test_img_to_world_sections():
    pos, n, sec = img_to_world(1000, 1375, 0.1)   # bottom-centre of main wall
    assert sec == "main"
    assert abs(pos[0] - (1000 - 447) / 300) < 1e-9
    assert pos[1] == 0.0
    assert pos[2] == 0.1
    assert np.allclose(n, [0, 0, 1])

    pos, n, sec = img_to_world(300, 1000, 0.08)   # left side wall
    assert sec == "left"
    assert abs(pos[0] - 0.08) < 1e-9              # protrudes +x
    assert abs(pos[2] - (447 - 300) / 300) < 1e-9
    assert np.allclose(n, [1, 0, 0])

    pos, n, sec = img_to_world(2050, 1000, 0.06)  # right side wall
    assert sec == "right"
    assert abs(pos[0] - (MAIN_WALL_W - 0.06)) < 1e-9
    assert np.allclose(n, [-1, 0, 0])


def test_holds_have_svg_outlines(wall):
    """Every hold carries its traced SVG outline mapped into 3D."""
    with_outline = [h for h in wall.holds if h.outline is not None and len(h.outline) >= 3]
    assert len(with_outline) == len(wall.holds)
    for h in with_outline[:50]:
        assert h.outline.shape[1] == 3
        # outline surrounds the hold centre on its own face plane
        rel = h.outline - h.pos
        assert np.abs(rel).max() < 1.5
        # on the protrusion plane: outline lies at the hold's depth along its normal
        proj = rel @ h.normal
        assert np.abs(proj).max() < 0.31


def test_parse_svg_polygon():
    from wallimport.importer import parse_svg_polygon
    pts = parse_svg_polygon("M 0 0 L 10 0 L 10 10 L 0 10 Z")
    assert len(pts) >= 3
    assert (0.0, 0.0) == pts[0]
    assert all(len(p) == 2 for p in pts)


def test_wall_quads(wall):
    for sec in ("main", "left", "right"):
        quad = wall.wall_quad(sec)
        assert quad.shape == (4, 3)
        assert np.allclose(quad[:, 1].min(), 0.0)


def test_summarize(wall):
    s = summarize(wall)
    assert s["n_holds"] == 417
    assert s["n_routes"] == 10


# ---------------- routesetter ------------------------------------------------

def test_generate_route_chains_upward(wall):
    for seed in (0, 1, 2):
        chain = generate_route(wall, seed=seed, difficulty="medium")
        assert 5 <= len(chain) <= 14
        ys = [h.pos[1] for h in chain]
        assert ys == sorted(ys)                      # bottom-up
        assert ys[-1] - ys[0] > 1.5                  # real climbing span
        for a, b in zip(chain[:-1], chain[1:]):
            step = np.linalg.norm(b.pos - a.pos)
            assert step < 1.6                        # one move apart


def test_generate_route_difficulty_types(wall):
    easy = generate_route(wall, seed=3, difficulty="easy")
    hard = generate_route(wall, seed=3, difficulty="hard")
    easy_hard_frac = np.mean([h.hold_type in ("crimp", "sloper") for h in easy])
    hard_hard_frac = np.mean([h.hold_type in ("crimp", "sloper") for h in hard])
    assert hard_hard_frac >= easy_hard_frac


def test_task_from_route_caps_slots(wall):
    chain = generate_route(wall, seed=0, difficulty="easy")
    task = task_from_route(wall, chain, max_slots=16)
    assert 5 <= len(task) <= 16
    assert task[-1] is chain[-1] or task[-1].pos[1] >= chain[-1].pos[1] - 1e-9


# ---------------- toprope wall + long routes ---------------------------------

@pytest.fixture(scope="module")
def toprope():
    from wallimport.importer import import_wall
    return import_wall("toprope")


def test_toprope_import(toprope):
    assert len(toprope.holds) == 479
    assert 8.5 < toprope.height < 10.0          # the tall wall (~9.25 m)
    assert 4.5 < toprope.width < 7.0
    for h in toprope.holds:
        x, y, z = h.pos
        assert 0.0 <= y <= toprope.height + 0.01
        assert x < toprope.width + 0.4
        assert z <= 1.0


def test_generate_long_route(toprope):
    from wallimport.routeset import generate_long_route
    for seed in (0, 1):
        route = generate_long_route(toprope, seed=seed, n_holds=30, difficulty="easy")
        assert len(route) >= 12
        ys = [h.pos[1] for h in route]
        assert ys[-1] - ys[0] > 4.0             # spans multiple metres
        assert ys == sorted(ys)
        for a, b in zip(route[:-1], route[1:]):
            assert np.linalg.norm(b.pos - a.pos) < 1.0   # every move is reachable


def test_long_route_task_env(toprope):
    from env.climbing_env_3d import ClimbingEnv3D
    from wallimport.routeset import generate_long_route, task_from_route
    task = task_from_route(toprope, generate_long_route(toprope, seed=0, difficulty="easy"),
                           max_slots=24)
    env = ClimbingEnv3D(toprope, task_holds=task, max_slots=24, max_steps=500)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.shape[0] == obs.shape[0]
    assert env.target_hold.pos[1] > toprope.height * 0.5
    obs, r, term, trunc, info = env.step(0)
    assert np.all(np.isfinite(obs))


# ---------------- body physics -----------------------------------------------

def make_hold(pos, axis, cone_deg=55.0, f_hand=900.0, f_foot=1400.0, hold_type="jug"):
    return Hold3D(
        id=hash(tuple(pos)) & 0xFFFF, pos=np.asarray(pos, dtype=float),
        normal=np.array([0.0, 0.0, 1.0]), depth=0.1, radius=0.05,
        pull_axis=np.asarray(axis, dtype=float) / np.linalg.norm(axis),
        cone_cos=math.cos(math.radians(cone_deg)),
        f_max_hand=f_hand, f_max_foot=f_foot, hold_type=hold_type,
        cat="C", route=None, wall_section="main", img_pos=(0, 0),
        hand_difficulty=3, foot_difficulty=3, direction=180,
    )


def test_two_bone_ik_lengths():
    root = np.zeros(3)
    end = np.array([0.5, 0.0, 0.0])
    for hint in (np.array([0, 1, 0.0]), np.array([0, 0, 1.0])):
        mid = two_bone_ik(root, end, 0.3, 0.29, hint)
        assert abs(np.linalg.norm(mid - root) - 0.3) < 1e-6
        assert abs(np.linalg.norm(end - mid) - 0.29) < 1e-6
        # joint bows toward the hint side
        assert np.dot(mid, hint) > 0


def test_hanging_equilibrium_below_anchor():
    """Two-hand hang: com settles below the anchors, no slip, arms stretched."""
    holds = [make_hold([0, 2.2, 0.1], [0, -1, 0.4]),
             make_hold([0.4, 2.2, 0.1], [0, -1, 0.4])]
    body = ClimberBody3D(np.array([0.2, 1.9, 0.2]))
    for h in holds:
        ok, _ = body.try_grab(h)
        assert ok
    for _ in range(400):
        body.step(0.01)
        rep = body.check_grips()
    assert body.num_attached() == 2
    assert rep.feasible
    assert body.com[1] < 2.2                      # hangs below the holds
    assert body.com[1] > 2.2 - 1.15               # ...but within arm reach


def test_slip_on_impossible_orientation():
    """Hanging from two up-facing holds (pull axis up) must slip: you cannot
    pull upward on an hold that opens upward without opposition."""
    up = [make_hold([0, 2.2, 0.1], [0, 1, 0.3], cone_deg=35.0),
          make_hold([0.4, 2.2, 0.1], [0, 1, 0.3], cone_deg=35.0)]
    body = ClimberBody3D(np.array([0.2, 1.9, 0.2]))
    for h in up:
        body.try_grab(h)
    slipped = False
    for _ in range(50):
        body.step(0.01)
        rep = body.check_grips()
        if not rep.feasible:
            slipped = True
            break
    assert slipped


def test_feet_support_reduces_hand_load():
    """Standing with feet on good holds below: hands carry less force."""
    hand_hold = make_hold([0.2, 1.9, 0.1], [0, -1, 0.4])
    foot_holds = [make_hold([0.0, 0.4, 0.1], [0, -1, 0.4], f_foot=1500.0),
                  make_hold([0.4, 0.4, 0.1], [0, -1, 0.4], f_foot=1500.0)]

    hang_only = ClimberBody3D(np.array([0.2, 1.6, 0.2]))
    hang_only.try_grab(hand_hold)
    for _ in range(200):
        hang_only.step(0.01)
    rep_hang = hang_only.check_grips()

    with_feet = ClimberBody3D(np.array([0.2, 1.4, 0.2]))
    with_feet.try_grab(hand_hold)
    for h in foot_holds:
        with_feet.limbs[2 + foot_holds.index(h)].attached = True
        with_feet.limbs[2 + foot_holds.index(h)].hold = h
        with_feet.limbs[2 + foot_holds.index(h)].contact = h.pos.copy()
    for _ in range(200):
        with_feet.step(0.01)
    rep_feet = with_feet.check_grips()

    assert abs(rep_feet.forces[0][1]) < abs(rep_hang.forces[0][1])  # hand load drops


def test_out_of_plane_swing_is_3d():
    """A lateral + out-of-wall push swings the com in z (3D pendulum)."""
    holds = [make_hold([0, 2.2, 0.1], [0, -1, 0.4]),
             make_hold([0.4, 2.2, 0.1], [0, -1, 0.4])]
    body = ClimberBody3D(np.array([0.2, 1.9, 0.2]))
    for h in holds:
        body.try_grab(h)
    body.vel += np.array([0.0, 0.0, 0.8])
    zs = []
    for _ in range(300):
        body.step(0.01)
        body.check_grips()
        zs.append(body.com[2])
    zs = np.array(zs)
    assert zs.max() > zs[0] + 0.15        # swings out of the wall plane
    assert zs.min() < zs.max() - 0.1      # and comes back (oscillation)


def test_dyno_releases_and_flies():
    hold = make_hold([0.2, 2.0, 0.1], [0, -1, 0.4])
    body = ClimberBody3D(np.array([0.2, 1.7, 0.2]))
    body.try_grab(hold)
    y0 = body.com[1]
    body.dyno(np.array([0.2, 2.6, 0.3]))
    assert body.num_attached() == 0
    assert body.in_dyno
    peak = y0
    for _ in range(30):
        body.step(0.01)
        peak = max(peak, body.com[1])
    assert peak > y0 + 0.3                 # actually flew upward


def test_limb_role_assignment():
    body = ClimberBody3D(np.array([1.0, 1.0, 0.3]))
    high = make_hold([1.0, 1.6, 0.1], [0, -1, 0.4])
    low = make_hold([1.0, 0.5, 0.1], [0, -1, 0.4])
    assert body.limb_for_hold(high.pos) in (0, 1)   # hands for high holds
    assert body.limb_for_hold(low.pos) in (2, 3)    # feet for low holds


def test_skeleton_shape():
    holds = [make_hold([0, 2.2, 0.1], [0, -1, 0.4]),
             make_hold([0.4, 2.2, 0.1], [0, -1, 0.4])]
    body = ClimberBody3D(np.array([0.2, 1.9, 0.2]))
    for h in holds:
        body.try_grab(h)
    for _ in range(50):
        body.step(0.01)
    sk = body.draw_skeleton()
    assert set(sk) >= {"pelvis", "neck", "head", "shoulders", "hips", "limbs"}
    assert len(sk["limbs"]) == 4
    for limb in sk["limbs"]:
        # 2-bone limbs: root -> mid -> end, each bone a sane length
        d1 = np.linalg.norm(limb["mid"] - limb["root"])
        d2 = np.linalg.norm(limb["end"] - limb["mid"])
        assert 0.15 < d1 < 0.5 and 0.15 < d2 < 0.5
    # hands end exactly at their contact points
    for i in (0, 1):
        assert np.allclose(sk["limbs"][i]["end"], holds[i].pos, atol=1e-9)
