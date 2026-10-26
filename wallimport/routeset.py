"""Synthetic routesetter: generate boulder problems on the imported wall.

The climbing app stores its real routes in Supabase (not in the bundled
annotations — `cat`/`num` there is a hold-set grouping, not a route), so we
generate problems the way a routesetter would: pick a start hold low on the
wall, then chain upward through holds that are one move apart (within arm
reach, sensible lateral drift), preferring hold qualities that match the
requested difficulty. The top hold of the chain is the task target.

All generated problems live on the main wall section and use the real hold
positions, depths, orientations and strengths from the HKUST annotations.
"""

from typing import List, Optional, Sequence

import numpy as np

from .importer import GymWall3D, Hold3D

DIFFICULTY_PRESETS = {
    # hold types preferred when chaining (weights), spacing, lateral band
    "easy": {"pref": {"jug": 1.0, "volume": 0.9, "pocket": 0.7, "pinch": 0.4, "uncategorized": 0.5, "sloper": 0.2, "crimp": 0.1, "jib": 0.1},
             "dy": (0.30, 0.62), "band": 0.85},
    "medium": {"pref": {"jug": 0.8, "volume": 0.7, "pocket": 0.8, "pinch": 0.7, "uncategorized": 0.5, "sloper": 0.6, "crimp": 0.5, "jib": 0.2},
               "dy": (0.35, 0.75), "band": 1.0},
    "hard": {"pref": {"jug": 0.4, "volume": 0.4, "pocket": 0.7, "pinch": 0.8, "uncategorized": 0.5, "sloper": 0.9, "crimp": 1.0, "jib": 0.4},
             "dy": (0.42, 0.85), "band": 1.15},
}

# direction quality for pulling while climbing up: down-facing holds are best
_DIR_BONUS = {"down": 0.30, "down-left": 0.20, "down-right": 0.20,
              "left": 0.05, "right": 0.05, "up": -0.10,
              "up-left": -0.05, "up-right": -0.05}


def _dir_bonus(hold: Hold3D) -> float:
    deg = hold.direction
    if 150 <= deg <= 210:
        return _DIR_BONUS["down"]
    if 210 < deg <= 260:
        return _DIR_BONUS["down-left"]
    if 100 <= deg < 150:
        return _DIR_BONUS["down-right"]
    if 60 <= deg < 100 or 260 < deg <= 300:
        return _DIR_BONUS["right"]
    if deg < 60 or deg > 300:
        return _DIR_BONUS["up"]
    return 0.0


def generate_route(wall: GymWall3D, seed: int = 0, difficulty: str = "medium",
                   n_holds: Optional[int] = None, top_y: float = 3.0,
                   start_hold: Optional[Hold3D] = None) -> List[Hold3D]:
    """Chain real holds into one boulder problem, bottom to top."""
    rng = np.random.default_rng(seed)
    preset = DIFFICULTY_PRESETS[difficulty]
    pref, (dy_lo, dy_hi), band = preset["pref"], preset["dy"], preset["band"]

    main = [h for h in wall.holds_by_section("main") if 0.15 <= h.pos[1] <= top_y + 0.6]
    if not main:
        raise RuntimeError("no main-wall holds available")

    if start_hold is None:
        lows = [h for h in main if 0.3 <= h.pos[1] <= 0.9]
        if not lows:
            lows = [h for h in main if h.pos[1] <= 1.2]
        start_hold = lows[int(rng.integers(len(lows)))]

    chain = [start_hold]
    x0 = start_hold.pos[0]
    target_len = n_holds or int(rng.integers(7, 12))

    while len(chain) < target_len:
        cur = chain[-1]
        if cur.pos[1] >= top_y:
            break
        cands = []
        for h in main:
            if any(h.id == c.id for c in chain):
                continue
            dx = h.pos[0] - cur.pos[0]
            dy = h.pos[1] - cur.pos[1]
            if not (dy_lo * 0.75 <= dy <= dy_hi) or abs(dx) > band:
                continue
            if abs(h.pos[0] - x0) > band * 1.4:
                continue
            score = (
                pref.get(h.hold_type, 0.3)
                + _dir_bonus(h)
                - 0.35 * abs(dx)
                - 0.4 * max(0.0, dy - dy_hi + 0.1)
                + 0.15 * (h.f_max_hand / 1000.0)
                + rng.uniform(0.0, 0.25)
            )
            cands.append((score, h))
        if not cands:
            break
        cands.sort(key=lambda t: -t[0])
        # pick among the top candidates (some setting variety)
        pick = cands[int(rng.integers(min(3, len(cands))))]
        chain.append(pick[1])

    if len(chain) < 5:
        return generate_route(wall, seed=seed + 1000, difficulty=difficulty,
                              n_holds=n_holds, top_y=top_y)
    return sorted(chain, key=lambda h: h.pos[1])


def task_from_route(wall: GymWall3D, route: Sequence[Hold3D],
                    max_slots: int = 16, extra_volumes: int = 2) -> List[Hold3D]:
    """Route chain plus a couple of nearby volumes usable as intermediate feet."""
    holds = list(route)
    if extra_volumes > 0:
        pts = np.array([h.pos for h in route])
        vols = [v for v in wall.holds_by_section("main") if v.cat == "F"]
        vols = sorted(vols, key=lambda v: np.min(np.linalg.norm(pts - v.pos, axis=1)))
        for v in vols:
            if len(holds) >= max_slots - extra_volumes:
                break
            if all(v.id != h.id for h in holds):
                holds.append(v)
    holds = sorted(holds, key=lambda h: h.pos[1])
    if len(holds) > max_slots:
        holds = holds[: max_slots - 1] + [holds[-1]]
    return holds


# ---------------------------------------------------------------------------
# Long routes (toprope-style), following the climbing app's route-graph design:
# holds form a DAG; an edge exists when a climber can physically move between
# two holds; edge weight = move difficulty (reach + hold quality + angle).
# A route is a bottom-to-top path whose per-move difficulty stays inside the
# requested budget — long endurance routes chain many cheap moves.
# ---------------------------------------------------------------------------

_TYPE_QUALITY = {  # higher = nicer to grab (lower move difficulty)
    "jug": 1.0, "volume": 0.95, "pocket": 0.8, "pinch": 0.6,
    "uncategorized": 0.5, "sloper": 0.45, "crimp": 0.35, "jib": 0.3,
}


def _move_difficulty(cur: Hold3D, nxt: Hold3D) -> float:
    d = nxt.pos - cur.pos
    dy, dx = d[1], d[0]
    dist = float(np.linalg.norm(d))
    quality = _TYPE_QUALITY.get(nxt.hold_type, 0.5)
    dir_bonus = _dir_bonus(nxt)          # down-facing holds are easier to use
    w = (
        1.5 * dist                       # reach cost dominates
        + 0.8 * max(0.0, abs(dx) - 0.3)  # lateral wandering costs
        - 0.9 * quality
        - 0.5 * dir_bonus
        + 0.6 * max(0.0, dy - 0.7)       # big spans are hard
    )
    return max(0.05, w)


def route_dag_neighbors(hold: Hold3D, holds: Sequence[Hold3D],
                        dy_range=(0.22, 0.55), max_dx=0.75):
    """All holds one move above `hold`, with move difficulty."""
    out = []
    for h in holds:
        if h.id == hold.id:
            continue
        d = h.pos - hold.pos
        if dy_range[0] <= d[1] <= dy_range[1] and abs(d[0]) <= max_dx:
            out.append((_move_difficulty(hold, h), h))
    return out


def generate_long_route(wall: GymWall3D, seed: int = 0, n_holds: int = 30,
                        difficulty: str = "easy", top_y: Optional[float] = None,
                        start_hold: Optional[Hold3D] = None) -> List[Hold3D]:
    """Chain ~n_holds real holds into one long (multi-metre) route.

    Bottom-to-top DAG walk that samples among the cheapest moves, so the
    route stays inside the difficulty budget while meandering naturally.
    """
    rng = np.random.default_rng(seed)
    H = wall.height
    top_y = top_y if top_y is not None else H - 0.9
    pool = [h for h in wall.holds_by_section("main") if 0.3 <= h.pos[1] <= top_y + 0.6]
    if not pool:
        raise RuntimeError("no main-wall holds for a long route")

    if start_hold is None:
        lows = [h for h in pool if h.pos[1] <= 1.6]
        start_hold = lows[int(rng.integers(len(lows)))] if lows else \
            min(pool, key=lambda h: h.pos[1])

    # slightly harder budgets allow crimp/sloper links on request
    max_move = {"easy": 1.15, "medium": 1.5, "hard": 1.9}[difficulty]

    chain = [start_hold]
    x0 = start_hold.pos[0]
    while chain[-1].pos[1] < top_y and len(chain) < n_holds + 16:
        cur = chain[-1]
        remaining = top_y - cur.pos[1]
        want = remaining / max(1, n_holds - len(chain))
        dy_hi = 0.5 if remaining > 1.0 else 0.7
        cands = []
        for w, h in route_dag_neighbors(cur, pool, dy_range=(0.20, dy_hi), max_dx=0.8):
            if any(h.id == c.id for c in chain):
                continue
            if abs(h.pos[0] - x0) > 1.4:
                continue
            # prefer steps close to the even progression pace
            pace = 0.3 * abs((h.pos[1] - cur.pos[1]) - want)
            cands.append((w + pace + rng.uniform(0.0, 0.35), h))
        if not cands:
            break
        cands.sort(key=lambda t: t[0])
        affordable = [c for c in cands if c[0] <= max_move + 0.8] or cands[:2]
        pick = affordable[int(rng.integers(min(3, len(affordable))))]
        chain.append(pick[1])

    chain = sorted(chain, key=lambda h: h.pos[1])
    if len(chain) < 14 or chain[-1].pos[1] < H * 0.55:
        return generate_long_route(wall, seed=seed + 997, n_holds=n_holds,
                                   difficulty=difficulty, top_y=top_y)
    return chain
