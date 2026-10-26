"""Import the HKUST boulder wall from the climbing-app annotations into
metric 3D hold geometry.

Coordinate system (metres):
    x -> along the main wall (0 at the left corner)
    y -> up (0 at the mat)
    z -> out of the main wall, toward the climber

The wall photo is a flattened panorama of a 3D structure:
    - main wall  : grid columns 1-4, plane z = 0, holds protrude +z
    - left side  : grid column 0, plane x = 0, holds protrude +x
    - right side : grid column 5, plane x = W, holds protrude -x
Every section is vertical (0 deg overhang); the 3D structure comes from the
two inside corners where the side walls fold toward the climber.

Each hold's usable pull direction is built from the app's `direction` field
(compass convention in the flattened image: 0=up, 90=right, 180=down,
270=left) mapped into the hold's wall-plane frame, tilted out of the wall by
a per-type opening angle. Depth priors are vision-calibrated (see
data/hkust/hold_depth_priors.json).
"""

import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "hkust")
BOULDER_DIR = os.path.join(DATA_DIR, "boulder")
TOPROPE_DIR = os.path.join(DATA_DIR, "toprope")


@dataclass
class WallGeometry:
    """Flattened-panorama -> 3D mapping for one wall structure.

    The gym photos are panoramas of 3D structures: a main wall plane plus a
    left side wall folding toward the climber (grid column 0). Every section
    is vertical (0 deg overhang).
    """
    viewbox_w: float
    viewbox_h: float
    px_per_m: float
    left_corner_x: float           # image x of the left fold
    left_depth: float              # how deep the left side wall runs (m)
    right_corner_x: Optional[float] = None   # None -> no right side wall
    right_depth: float = 0.45

    @property
    def width(self) -> float:
        right = self.right_corner_x if self.right_corner_x is not None else self.viewbox_w
        return (right - self.left_corner_x) / self.px_per_m

    @property
    def height(self) -> float:
        return self.viewbox_h / self.px_per_m


BOULDER_GEO = WallGeometry(
    viewbox_w=2089.0, viewbox_h=1375.0, px_per_m=300.0,
    left_corner_x=447.0, left_depth=0.90, right_corner_x=1962.5, right_depth=0.45,
)
# toprope scale: same physical holds photograph ~3x larger (box median 154 vs 52.6 px)
TOPROPE_GEO = WallGeometry(
    viewbox_w=6120.0, viewbox_h=8160.0, px_per_m=300.0 * 154.6 / 52.6,
    left_corner_x=1132.0, left_depth=0.85, right_corner_x=None,
)

# Backwards-compatible boulder constants
VIEWBOX_W = BOULDER_GEO.viewbox_w
VIEWBOX_H = BOULDER_GEO.viewbox_h
PX_PER_M = BOULDER_GEO.px_per_m
LEFT_CORNER_X = BOULDER_GEO.left_corner_x
RIGHT_CORNER_X = BOULDER_GEO.right_corner_x
MAIN_WALL_W = BOULDER_GEO.width
WALL_H = BOULDER_GEO.height
LEFT_SIDE_DEPTH = BOULDER_GEO.left_depth
RIGHT_SIDE_DEPTH = BOULDER_GEO.right_depth

MAIN_NORMAL = np.array([0.0, 0.0, 1.0])
LEFT_NORMAL = np.array([1.0, 0.0, 0.0])
RIGHT_NORMAL = np.array([-1.0, 0.0, 0.0])

WALL_FILES = {
    "boulder": dict(dir=BOULDER_DIR, geo=BOULDER_GEO, box_median=52.6,
                    spec="HOLD_SPEC_v12.json", ann="hold_annotations_v12.json"),
    "toprope": dict(dir=TOPROPE_DIR, geo=TOPROPE_GEO, box_median=154.6,
                    spec="TOPROPE_HOLD_SPEC_v12.json", ann="toprope_hold_annotations_v12.json"),
}

# Compass angle (deg) -> 2D unit vector in image coords (x right, y down).
def _compass_to_img_vec(deg: float) -> np.ndarray:
    theta = math.radians(deg or 180.0)
    return np.array([math.sin(theta), -math.cos(theta)])


@dataclass
class Hold3D:
    id: int
    pos: np.ndarray            # (3,) contact point (m, world)
    normal: np.ndarray         # (3,) wall-section outward normal
    depth: float               # protrusion from wall plane (m)
    radius: float              # effective grip radius (m)
    pull_axis: np.ndarray      # (3,) unit vector of usable pull direction
    cone_cos: float            # cos of grip-cone half angle
    f_max_hand: float          # N
    f_max_foot: float          # N
    hold_type: str
    cat: str                   # C = climbing hold, F = feature/volume, I = insert
    route: Optional[Tuple[str, int]]
    wall_section: str          # 'main' | 'left' | 'right'
    img_pos: Tuple[float, float]
    hand_difficulty: float
    foot_difficulty: float
    direction: float           # original compass direction (deg)
    outline: Optional[np.ndarray] = None  # (N,3) outline polygon on the hold's face

    def to_dict(self):
        return {
            "id": self.id,
            "pos": self.pos.tolist(),
            "normal": self.normal.tolist(),
            "depth": self.depth,
            "radius": self.radius,
            "pull_axis": self.pull_axis.tolist(),
            "cone_cos": self.cone_cos,
            "f_max_hand": self.f_max_hand,
            "f_max_foot": self.f_max_foot,
            "hold_type": self.hold_type,
            "cat": self.cat,
            "route": list(self.route) if self.route else None,
            "wall_section": self.wall_section,
            "img_pos": list(self.img_pos),
            "outline": None if self.outline is None else self.outline.tolist(),
        }


@dataclass
class GymWall3D:
    holds: List[Hold3D] = field(default_factory=list)
    width: float = MAIN_WALL_W
    height: float = WALL_H
    left_depth: float = LEFT_SIDE_DEPTH
    right_depth: float = RIGHT_SIDE_DEPTH
    route_names: Dict[str, str] = field(default_factory=dict)

    # ---- geometry queries -------------------------------------------------
    def wall_quad(self, section: str) -> np.ndarray:
        """Four corner points (m) of a wall section, for rendering."""
        W, H, dl, dr = self.width, self.height, self.left_depth, self.right_depth
        if section == "main":
            return np.array([[0, 0, 0], [W, 0, 0], [W, H, 0], [0, H, 0]])
        if section == "left":
            return np.array([[0, 0, 0], [0, 0, dl], [0, H, dl], [0, H, 0]])
        if section == "right":
            return np.array([[W, 0, 0], [W, 0, dr], [W, H, dr], [W, H, 0]])
        raise ValueError(section)

    def holds_by_section(self, section: str) -> List[Hold3D]:
        return [h for h in self.holds if h.wall_section == section]

    def routes(self) -> Dict[Tuple[str, int], List[Hold3D]]:
        out: Dict[Tuple[str, int], List[Hold3D]] = {}
        for h in self.holds:
            if h.cat == "C" and h.route is not None:
                out.setdefault(h.route, []).append(h)
        return out

    def route_name(self, route: Tuple[str, int]) -> str:
        return self.route_names.get(f"{route[0]}{route[1]}", f"route {route[0]}{route[1]}")

    def default_route(self) -> Tuple[str, int]:
        """Best training route: vertical, in the main wall, 8+ holds."""
        best, best_score = None, -1.0
        for key, holds in self.routes().items():
            main = [h for h in holds if h.wall_section == "main"]
            if len(main) < 8:
                continue
            ys = [h.pos[1] for h in main]
            span = max(ys) - min(ys)
            xs = [h.pos[0] for h in main]
            lateral = max(xs) - min(xs)
            score = span - 0.5 * max(0.0, lateral - 1.2)
            if score > best_score:
                best, best_score = key, score
        if best is None:
            raise RuntimeError("no route with >=8 main-wall holds")
        return best

    def task_holds(self, route: Tuple[str, int], include_volumes: bool = True) -> List[Hold3D]:
        """Route holds plus volumes near the route line (real problems share volumes)."""
        holds = sorted(self.routes()[route], key=lambda h: h.pos[1])
        if include_volumes:
            xs = np.array([h.pos[0] for h in holds])
            zs = np.array([h.pos[2] for h in holds])
            y_min, y_max = holds[0].pos[1] - 0.4, holds[-1].pos[1] + 0.4
            for v in self.holds:
                if v.cat != "F":
                    continue
                near = (
                    y_min <= v.pos[1] <= y_max
                    and min(abs(v.pos[0] - xs)) < 0.9
                    and min(abs(v.pos[2] - zs)) < 0.9
                )
                if near:
                    holds.append(v)
            holds = sorted(holds, key=lambda h: h.pos[1])
        return holds


# ---- conversion helpers ----------------------------------------------------

def _section_for(x_img: float, geo: WallGeometry = BOULDER_GEO) -> str:
    if x_img < geo.left_corner_x:
        return "left"
    if geo.right_corner_x is not None and x_img > geo.right_corner_x:
        return "right"
    return "main"


def img_to_world(x_img: float, y_img: float, depth: float,
                 geo: WallGeometry = BOULDER_GEO) -> Tuple[np.ndarray, np.ndarray, str]:
    """Map a hold centre from panorama pixels to a 3D contact point + wall normal.

    Side-wall distances are clamped to the wall extents: a few huge volume
    annotations have centres outside the flattened panorama.
    """
    section = _section_for(x_img, geo)
    y_m = (geo.viewbox_h - y_img) / geo.px_per_m
    if section == "main":
        x_m = float(np.clip((x_img - geo.left_corner_x) / geo.px_per_m, 0.0, geo.width))
        pos = np.array([x_m, y_m, depth])
        return pos, MAIN_NORMAL.copy(), section
    if section == "left":
        z_m = float(np.clip((geo.left_corner_x - x_img) / geo.px_per_m, 0.0, geo.left_depth))
        pos = np.array([depth, y_m, z_m])
        return pos, LEFT_NORMAL.copy(), section
    z_m = float(np.clip((x_img - geo.right_corner_x) / geo.px_per_m, 0.0, geo.right_depth))
    pos = np.array([geo.width - depth, y_m, z_m])
    return pos, RIGHT_NORMAL.copy(), section


def _pull_axis(direction_deg: float, section: str, tilt_deg: float) -> np.ndarray:
    """Compass direction in the flattened image -> 3D usable-pull unit vector."""
    d2 = _compass_to_img_vec(direction_deg)
    if section == "main":
        # image +x -> world +x, image +y(down) -> world -y
        inplane = np.array([d2[0], -d2[1], 0.0])
        normal = MAIN_NORMAL
    elif section == "left":
        # image +x -> toward corner -> world -z ; image down -> -y
        inplane = np.array([0.0, -d2[1], -d2[0]])
        normal = LEFT_NORMAL
    else:
        # image +x -> away from corner -> world +z ; image down -> -y
        inplane = np.array([0.0, -d2[1], d2[0]])
        normal = RIGHT_NORMAL
    tilt = math.radians(tilt_deg)
    axis = math.cos(tilt) * inplane + math.sin(tilt) * normal
    return axis / np.linalg.norm(axis)


def _load_priors() -> dict:
    with open(os.path.join(DATA_DIR, "hold_depth_priors.json"), encoding="utf-8") as f:
        return json.load(f)


def _type_of(spec: dict) -> str:
    if spec.get("cat") == "F":
        return "volume"
    return (spec.get("type") or "uncategorized").lower()


_PATH_TOKEN = re.compile(r"([MLZmlz])|(-?\d*\.?\d+)")


def parse_svg_polygon(path_str: str, max_points: int = 40) -> List[Tuple[float, float]]:
    """Parse an `M x y L x y ... Z` outline into downsampled (x, y) pixels."""
    tokens = []
    for cmd, num in _PATH_TOKEN.findall(path_str):
        tokens.append(cmd if cmd else float(num))
    pts, cur_cmd, i = [], None, 0
    while i < len(tokens):
        t = tokens[i]
        if isinstance(t, str):
            cur_cmd = t.upper()
            i += 1
            if cur_cmd == "Z":
                break
            continue
        if cur_cmd in ("M", "L") and i + 1 < len(tokens) and not isinstance(tokens[i + 1], str):
            pts.append((t, tokens[i + 1]))
            i += 2
        else:
            i += 1
    if len(pts) < 3:
        return []
    # Douglas-Peucker-lite: uniform stride keeps the shape at <= max_points
    stride = max(1, len(pts) // max_points)
    return pts[::stride]


def outline_to_world(pts_img: List[Tuple[float, float]], depth: float,
                     geo: WallGeometry = BOULDER_GEO) -> Optional[np.ndarray]:
    """Map an image-space outline onto the hold's face plane in 3D."""
    if not pts_img:
        return None
    out = []
    for x_img, y_img in pts_img:
        pos, _, _ = img_to_world(x_img, y_img, depth, geo)
        out.append(pos)
    return np.array(out)


def import_wall(wall_name: str = "boulder") -> GymWall3D:
    """Import a wall ('boulder' ~5x4.6 m or 'toprope' ~5.6x9.3 m)."""
    if wall_name not in WALL_FILES:
        raise ValueError(f"unknown wall '{wall_name}'; available: {list(WALL_FILES)}")
    cfg = WALL_FILES[wall_name]
    data_dir, geo, med_box = cfg["dir"], cfg["geo"], cfg["box_median"]

    priors = _load_priors()
    types = priors["types"]

    with open(os.path.join(data_dir, cfg["spec"]), encoding="utf-8") as f:
        specs = json.load(f)
    with open(os.path.join(data_dir, cfg["ann"]), encoding="utf-8") as f:
        annotations = {a["id"]: a for a in json.load(f)}

    wall = GymWall3D()
    wall.width = geo.width
    wall.height = geo.height
    wall.left_depth = geo.left_depth
    wall.right_depth = geo.right_depth
    wall.geo = geo
    wall.name = wall_name

    for i, s in enumerate(specs):
        htype = _type_of(s)
        p = types.get(htype, types["uncategorized"])

        box = float(s.get("boxSize") or med_box)
        size_f = float(np.clip(0.7 + 0.6 * box / med_box, 0.6, 1.6))
        depth = p["depth_m"] * size_f

        hand_diff = float(s.get("handDifficulty") or 3)
        foot_diff = float(s.get("footDifficulty") or 3)
        foot_rating = float(s.get("footRating") or 3)
        f_hand = p["f_max_hand_N"] * (0.7 + 0.12 * hand_diff)
        f_foot = p["f_max_foot_N"] * (0.75 + 0.09 * foot_rating)

        x_img, y_img = float(s["center_x"]), float(s["center_y"])
        pos, normal, section = img_to_world(x_img, y_img, depth, geo)
        axis = _pull_axis(float(s.get("direction") or 180), section, p["out_of_plane_tilt_deg"])

        route = None
        if s.get("cat") in ("C", "F", "I") and s.get("num") is not None:
            route = (str(s["cat"]), int(s["num"]))

        ann = annotations.get(int(s["id"]))
        outline = None
        if ann and ann.get("path"):
            pts_img = parse_svg_polygon(ann["path"])
            outline = outline_to_world(pts_img, depth, geo)

        wall.holds.append(
            Hold3D(
                id=int(s["id"]),
                pos=pos.astype(np.float64),
                normal=normal,
                depth=depth,
                radius=max(0.02, box / PX_PER_M / 2.0),
                pull_axis=axis,
                cone_cos=math.cos(math.radians(p["cone_half_angle_deg"])),
                f_max_hand=f_hand,
                f_max_foot=f_foot,
                hold_type=htype,
                cat=str(s.get("cat") or "C"),
                route=route,
                wall_section=section,
                img_pos=(x_img, y_img),
                hand_difficulty=hand_diff,
                foot_difficulty=foot_diff,
                direction=float(s.get("direction") or 180),
                outline=outline,
            )
        )

    # Route display names: alphabetical list indexed by C-route order (C1, C2, ...).
    if wall_name == "boulder":
        with open(os.path.join(data_dir, "route_names_en.json"), encoding="utf-8") as f:
            names_list = json.load(f)
        c_routes = sorted(k[1] for k in wall.routes().keys() if k[0] == "C")
        for idx, num in enumerate(c_routes):
            if idx < len(names_list):
                wall.route_names[f"C{num}"] = names_list[idx]

    return wall


def import_boulder_wall(data_dir: str = None) -> GymWall3D:
    """Back-compatible alias for import_wall('boulder')."""
    return import_wall("boulder")


def summarize(wall: GymWall3D) -> dict:
    from collections import Counter

    secs = Counter(h.wall_section for h in wall.holds)
    tps = Counter(h.hold_type for h in wall.holds)
    routes = wall.routes()
    return {
        "n_holds": len(wall.holds),
        "sections": dict(secs),
        "types": dict(tps),
        "n_routes": len(routes),
        "route_sizes": {f"{k[0]}{k[1]}": len(v) for k, v in sorted(routes.items())},
        "main_wall_w_m": round(wall.width, 3),
        "wall_h_m": round(wall.height, 3),
    }


if __name__ == "__main__":
    w = import_boulder_wall()
    print(json.dumps(summarize(w), indent=2))
    print("default route:", w.default_route(), w.route_name(w.default_route()))
