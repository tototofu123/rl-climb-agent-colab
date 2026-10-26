"""3D physics for the climber body.

The climber is an anthropometric articulated body (not a stickman): pelvis,
torso with lean/twist orientation, head, two 2-bone arms and two 2-bone legs
with analytic IK, and per-limb mass. All state lives in 3D world coordinates
(x along wall, y up, z out of the wall).

Physics model (semi-implicit, sub-stepped):
  * The body centre of mass is integrated under gravity with damping.
  * Attached limbs impose 3D distance constraints from hold contact points
    (position projection + radial velocity kill), so the body swings as a 3D
    pendulum - both laterally (x-y) and out of the wall plane (y-z).
  * Torso orientation (lean out of the wall + lateral lean) follows a damped
    spring toward the quasi-static equilibrium implied by the current
    anchors: hanging from hands aligns the torso under the hand anchors,
    standing on feet keeps it upright.
  * Grip feasibility: for the current anchor set we solve the minimum-norm
    contact forces satisfying force + torque equilibrium, then check each
    limb's force against its hold's grip cone (pull axis + half angle from
    the real HKUST hold annotations) and strength limit. Violations cause
    slips (the limb detaches). This is where hold orientation, depth and
    type - the 3D angles - actually matter.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

GRAVITY = np.array([0.0, -9.81, 0.0])

LIMB_NAMES = ["L-hand", "R-hand", "L-foot", "R-foot"]
HANDS = (0, 1)
FEET = (2, 3)


@dataclass
class Anthropometry:
    mass: float = 70.0
    torso_len: float = 0.52          # pelvis -> neck
    shoulder_w: float = 0.40
    hip_w: float = 0.30
    upper_arm: float = 0.30
    forearm: float = 0.29
    thigh: float = 0.44
    shin: float = 0.43
    head_r: float = 0.11
    # segment masses (kg): torso split around the pelvis, per Dempster-ish shares
    m_torso: float = 30.0
    m_pelvis: float = 12.0
    m_head: float = 5.0
    m_upper_arm: float = 2.0
    m_forearm: float = 1.2
    m_hand: float = 0.4
    m_thigh: float = 5.5
    m_shin: float = 3.0
    m_foot: float = 0.9

    @property
    def arm_reach(self) -> float:
        return self.upper_arm + self.forearm

    @property
    def leg_reach(self) -> float:
        return self.thigh + self.shin


DEFAULT_ANTHRO = Anthropometry()


def _limb_lengths(idx: int, a: Anthropometry):
    if idx in HANDS:
        return a.upper_arm, a.forearm
    return a.thigh, a.shin


def two_bone_ik(root: np.ndarray, end: np.ndarray, l1: float, l2: float,
                bend_hint: np.ndarray) -> Optional[np.ndarray]:
    """Analytic 2-bone IK: returns the middle joint (elbow/knee).

    bend_hint: preferred direction the joint should bow toward; its component
    perpendicular to (end - root) is used to pick the IK branch.
    """
    span = end - root
    d = np.linalg.norm(span)
    if d < 1e-6:
        return None
    d = min(d, l1 + l2 - 1e-4)
    d = max(d, abs(l1 - l2) + 1e-4)
    # distance from root to the joint plane along the span
    a = (l1 * l1 - l2 * l2 + d * d) / (2 * d)
    h2 = max(l1 * l1 - a * a, 0.0)
    h = np.sqrt(h2)
    u = span / np.linalg.norm(span)
    perp = bend_hint - np.dot(bend_hint, u) * u
    pn = np.linalg.norm(perp)
    perp = perp / pn if pn > 1e-6 else np.array([0.0, 0.0, 1.0] - np.dot([0, 0, 1], u) * u)
    pn = np.linalg.norm(perp)
    if pn < 1e-6:
        perp = np.array([1.0, 0.0, 0.0])
    else:
        perp = perp / pn
    return root + a * u + h * perp


@dataclass
class LimbState3D:
    attached: bool = False
    hold: object = None            # Hold3D when attached
    contact: Optional[np.ndarray] = None  # (3,) world contact point
    grip_util: float = 0.0         # 0 = solid, >= 1 = slipping


@dataclass
class GripReport:
    feasible: bool
    forces: List[np.ndarray] = field(default_factory=list)
    utils: List[float] = field(default_factory=list)
    worst_limb: int = -1
    min_margin: float = 1.0


class ClimberBody3D:
    """3D articulated climber rooted at the body centre of mass."""

    def __init__(self, com_start: np.ndarray, anthro: Anthropometry = DEFAULT_ANTHRO):
        self.a = anthro
        self.com = np.asarray(com_start, dtype=np.float64).copy()
        self.vel = np.zeros(3)
        self.limbs: List[LimbState3D] = [LimbState3D() for _ in range(4)]
        # torso orientation: lean_out (phi, + = leaning away from wall),
        # lean_side (lambda, + = top tilts toward -x)
        self.lean_out = 0.0
        self.lean_side = 0.0
        self.lean_out_vel = 0.0
        self.lean_side_vel = 0.0
        self.dyno_steps = 0
        self.slip_events: List[int] = []  # limbs that slipped on the last check
        self.wall_normal = np.array([0.0, 0.0, 1.0])

    # ---- frame ------------------------------------------------------------
    def torso_up(self) -> np.ndarray:
        lo, ls = self.lean_out, self.lean_side
        u = np.array([-np.sin(ls), np.cos(ls) * np.cos(lo), np.cos(ls) * np.sin(lo)])
        return u / np.linalg.norm(u)

    def torso_right(self) -> np.ndarray:
        u = self.torso_up()
        r = np.cross(u, self.wall_normal)
        if np.linalg.norm(r) < 1e-6:
            r = np.array([1.0, 0.0, 0.0])
        return r / np.linalg.norm(r)

    def _pelvis_from_com(self) -> np.ndarray:
        """Approximate pelvis position for the current com + orientation.

        The torso+head dominate the mass near the com, so place the pelvis a
        fixed fraction of torso length below the com along -u, then correct
        once for limb masses hanging off it.
        """
        u = self.torso_up()
        pelvis = self.com - u * (self.a.torso_len * 0.35)
        seg = self._segments(pelvis, u)
        m_total = sum(m for _, _, m in seg)
        com_est = sum(c * m for c, _, m in seg) / m_total
        return pelvis + (self.com - com_est)

    def joints_all(self):
        """All four anchor joints + stretched lengths in one frame evaluation."""
        pelvis = self._pelvis_from_com()
        u = self.torso_up()
        r = self.torso_right()
        a = self.a
        sc = pelvis + u * (a.torso_len - 0.07)
        half_sh = r * (a.shoulder_w / 2)
        half_hip = r * (a.hip_w / 2)
        joints = (
            sc - half_sh,            # L shoulder
            sc + half_sh,            # R shoulder
            pelvis - half_hip,       # L hip
            pelvis + half_hip,       # R hip
        )
        return pelvis, u, r, joints

    def _segments(self, pelvis: np.ndarray, u: np.ndarray):
        """(centre, half-direction info, mass) for every body segment."""
        a = self.a
        r = self.torso_right()
        neck = pelvis + u * a.torso_len
        sc = pelvis + u * (a.torso_len - 0.07)
        sh_l, sh_r = sc - r * (a.shoulder_w / 2), sc + r * (a.shoulder_w / 2)
        hip_l, hip_r = pelvis - r * (a.hip_w / 2), pelvis + r * (a.hip_w / 2)
        head = neck + u * a.head_r

        segs = [
            ((pelvis + neck) / 2, a.m_torso + a.m_pelvis),
            (head, a.m_head),
        ]
        joints = [sh_l, sh_r, hip_l, hip_r]
        for i, limb in enumerate(self.limbs):
            root = joints[i]
            l1, l2 = _limb_lengths(i, a)
            end = limb.contact if (limb.attached and limb.contact is not None) else \
                root - u * (l1 + l2) * 0.85
            if i in HANDS:
                bend = -u * 0.6 + self.wall_normal * 0.4  # elbows bow down/out
                m_end = a.m_hand
            else:
                bend = self.wall_normal * 1.0 + (-u) * 0.15  # knees away from wall
                m_end = a.m_foot
            mid = two_bone_ik(root, end, l1, l2, bend)
            if mid is None:
                mid = (root + end) / 2
            segs.append(((root + mid) / 2, a.m_upper_arm if i in HANDS else a.m_thigh))
            segs.append(((mid + end) / 2, a.m_forearm if i in HANDS else a.m_shin))
            segs.append((np.asarray(end, dtype=np.float64), m_end))
        return [(np.asarray(c, dtype=np.float64), None, m) for c, m in segs]

    def joint_world(self, idx: int) -> Tuple[np.ndarray, float]:
        """(shoulder/hip joint, stretched limb length) for limb idx."""
        _, _, _, joints = self.joints_all()
        length = self.a.arm_reach if idx in HANDS else self.a.leg_reach
        return joints[idx], length

    def limb_end(self, idx: int) -> np.ndarray:
        """World position of hand/foot (contact if attached, else hanging)."""
        limb = self.limbs[idx]
        if limb.attached and limb.contact is not None:
            return limb.contact
        j, length = self.joint_world(idx)
        u = self.torso_up()
        return j - u * length * 0.85

    def draw_skeleton(self) -> dict:
        """All landmarks for rendering (pelvis, neck, head, joints, limbs)."""
        pelvis = self._pelvis_from_com()
        u = self.torso_up()
        r = self.torso_right()
        a = self.a
        neck = pelvis + u * a.torso_len
        sc = pelvis + u * (a.torso_len - 0.07)
        sh_l = sc - r * (a.shoulder_w / 2)
        sh_r = sc + r * (a.shoulder_w / 2)
        hip_l = pelvis - r * (a.hip_w / 2)
        hip_r = pelvis + r * (a.hip_w / 2)
        head = neck + u * a.head_r
        limbs = []
        for i in range(4):
            root = [sh_l, sh_r, hip_l, hip_r][i]
            l1, l2 = _limb_lengths(i, a)
            end = self.limb_end(i)
            if i in HANDS:
                bend = -u * 0.6 + self.wall_normal * 0.4
            else:
                bend = self.wall_normal * 1.0 - u * 0.15
            mid = two_bone_ik(root, end, l1, l2, bend)
            if mid is None:
                mid = (root + end) / 2
            limbs.append({"root": root, "mid": mid, "end": end, "is_hand": i in HANDS,
                          "attached": self.limbs[i].attached})
        return {"pelvis": pelvis, "neck": neck, "head": head, "shoulders": (sh_l, sh_r),
                "hips": (hip_l, hip_r), "limbs": limbs, "torso_up": u}

    # ---- actions ----------------------------------------------------------
    def limb_for_hold(self, hold_pos: np.ndarray) -> Optional[int]:
        """Pick the free limb for a hold: high holds -> hands, low -> feet."""
        want_hand = hold_pos[1] > self.com[1] - 0.05
        order = [0, 1, 2, 3] if want_hand else [2, 3, 0, 1]
        for i in order:
            if not self.limbs[i].attached:
                return i
        return None

    def try_grab(self, hold, max_reach_factor: float = 1.0) -> Tuple[bool, int]:
        """Try to grab a Hold3D with the best free limb. Returns (ok, limb)."""
        idx = self.limb_for_hold(hold.pos)
        if idx is None:
            return False, -1
        j, length = self.joint_world(idx)
        reach = length * max_reach_factor
        if np.linalg.norm(np.asarray(hold.pos) - j) > reach:
            return False, idx
        limb = self.limbs[idx]
        limb.attached = True
        limb.hold = hold
        limb.contact = np.asarray(hold.pos, dtype=np.float64).copy()
        limb.grip_util = 0.0
        self._update_wall_normal()
        return True, idx

    def release(self, idx: int) -> bool:
        if 0 <= idx < 4 and self.limbs[idx].attached:
            self.limbs[idx].attached = False
            self.limbs[idx].hold = None
            self.limbs[idx].contact = None
            self.limbs[idx].grip_util = 0.0
            self._update_wall_normal()
            return True
        return False

    def _update_wall_normal(self):
        """Face the torso toward the average wall normal of anchored holds."""
        normals, weights = [], []
        for i, limb in enumerate(self.limbs):
            if limb.attached and limb.hold is not None:
                normals.append(np.asarray(limb.hold.normal))
                weights.append(1.0 if i in HANDS else 0.6)
        if normals:
            n = np.average(np.array(normals), axis=0, weights=weights)
            if np.linalg.norm(n) > 1e-6:
                self.wall_normal = n / np.linalg.norm(n)

    def dyno(self, target: np.ndarray, speed: float = 2.8):
        """3D dyno impulse: release everything, fly toward the target."""
        for limb in self.limbs:
            limb.attached = False
            limb.hold = None
            limb.contact = None
            limb.grip_util = 0.0
        d = np.asarray(target, dtype=np.float64) - self.com
        d[1] = max(d[1], 0.2)
        d = d / max(np.linalg.norm(d), 1e-6)
        # bias: up, slightly out of the wall so the body clears the holds
        imp = np.array([d[0], d[1] * 1.3 + 0.4, d[2] * 0.6 + 0.25])
        imp = imp / np.linalg.norm(imp) * speed
        self.vel += imp
        self.dyno_steps = 6

    @property
    def in_dyno(self) -> bool:
        return self.dyno_steps > 0

    def num_attached(self) -> int:
        return sum(1 for l in self.limbs if l.attached)

    def is_falling(self) -> bool:
        return self.num_attached() == 0 and not self.in_dyno

    # ---- physics ----------------------------------------------------------
    def step(self, dt: float, bounds: Tuple[float, float, float] = (5.0, 4.5, 1.2),
             leg_drive: bool = True):
        self.vel += GRAVITY * dt
        if leg_drive:
            self._apply_leg_drive(dt)
        self.vel *= 0.995
        self.com += self.vel * dt
        self._enforce_constraints()
        W, H, D = bounds
        self.com[1] = min(self.com[1], H)
        if self.com[1] <= 0.0:
            self.com[1] = 0.0
            self.vel[self.vel[1] < 0] = 0.0  # landed on the mat
        self.com[2] = np.clip(self.com[2], 0.0, D)
        if self.com[2] <= 0.0 + 1e-9 and self.vel[2] < 0:
            self.vel[2] = 0.0
        if self.dyno_steps > 0:
            self.dyno_steps -= 1
        self._update_torso(dt)

    def _apply_leg_drive(self, dt: float, k_spring: float = 6.0, f_cap: float = 1.6):
        """Leg extension: attached feet let the climber stand up.

        A bent leg drives the body vertically (knee extension pressing the
        foothold, held by friction) with force up to ~f_cap x bodyweight
        shared across both legs. The leg-length constraint elsewhere caps the
        rise at standing height over the foothold. Without this the body can
        only passively hang and every upward move would need a dyno.
        """
        feet = [i for i in FEET if self.limbs[i].attached and self.limbs[i].contact is not None]
        if not feet:
            return
        _, _, _, joints = self.joints_all()
        drive = 0.0
        for i in feet:
            contact = self.limbs[i].contact
            d = np.linalg.norm(joints[i] - contact)
            L = self.a.leg_reach
            if d >= L * 0.97 or d < 1e-4:
                continue
            drive += k_spring * (1.0 - d / L) * self.a.mass * 9.81
        drive = min(drive, f_cap * self.a.mass * 9.81)
        self.vel[1] += drive / self.a.mass * dt

    def _enforce_constraints(self, tol: float = 0.005):
        for _pass in range(2):
            _, _, _, joints = self.joints_all()
            moved = False
            for i, limb in enumerate(self.limbs):
                if not (limb.attached and limb.contact is not None):
                    continue
                length = self.a.arm_reach if i in HANDS else self.a.leg_reach
                delta = joints[i] - limb.contact
                dist = np.linalg.norm(delta)
                if dist > length + tol:
                    dirn = delta / dist
                    corr = (dist - length) * dirn
                    self.com -= corr
                    v_rad = np.dot(self.vel, dirn)
                    if v_rad > 0:
                        self.vel -= dirn * v_rad * 0.85
                    moved = True
            if not moved:
                break

    def _update_torso(self, dt: float, k: float = 26.0, c: float = 7.0):
        """Damped-spring torso orientation toward the anchor-implied pose."""
        hands = [i for i in HANDS if self.limbs[i].attached and self.limbs[i].contact is not None]
        feet = [i for i in FEET if self.limbs[i].attached and self.limbs[i].contact is not None]
        u_target = self.torso_up()
        if hands:
            s = np.mean([self.limbs[i].contact for i in hands], axis=0)
            u_h = s - self.com
            if np.linalg.norm(u_h) > 1e-3:
                u_h = u_h / np.linalg.norm(u_h)
                u_h = u_h * 0.35 + np.array([0, 1.0, 0]) * 0.65
                u_h = u_h / np.linalg.norm(u_h)
                u_target = u_h
            if feet:
                u_target = u_target * 0.75 + np.array([0, 1.0, 0]) * 0.25
                u_target = u_target / np.linalg.norm(u_target)
        elif feet:
            u_target = np.array([0.0, 1.0, 0.12])  # slight natural lean out
            u_target = u_target / np.linalg.norm(u_target)
        # else free-fall: keep current orientation

        phi_t = np.arctan2(u_target[2], u_target[1])
        lam_t = -np.arcsin(np.clip(u_target[0], -1, 1))
        self.lean_out_vel += (k * (phi_t - self.lean_out) - c * self.lean_out_vel) * dt
        self.lean_side_vel += (k * (lam_t - self.lean_side) - c * self.lean_side_vel) * dt
        self.lean_out = float(np.clip(self.lean_out + self.lean_out_vel * dt, -0.18, 1.15))
        self.lean_side = float(np.clip(self.lean_side + self.lean_side_vel * dt, -0.8, 0.8))

    # ---- grip feasibility ---------------------------------------------------
    def check_grips(self) -> GripReport:
        """Solve min-norm contact forces and check every anchor's grip cone.

        Climber-applied forces f_i on the holds must satisfy
            sum(f_i) = (0, -M g, 0)          (supports the weight)
            sum((c_i - com) x f_i) = 0       (no net torque)
        Each f_i must lie within hold_i's grip cone (angle to pull_axis below
        the half-angle) and below its strength limit; otherwise the limb slips.
        """
        self.slip_events = []
        idxs = [i for i in range(4) if self.limbs[i].attached and self.limbs[i].contact is not None]
        report = GripReport(feasible=True)
        if not idxs:
            return report
        M = self.a.mass
        W = np.array([0.0, -M * 9.81, 0.0])
        n = len(idxs)
        if n == 1:
            f = [W.copy()]
        else:
            A = np.zeros((6, 3 * n))
            b = np.zeros(6)
            b[:3] = W
            for col, i in enumerate(idxs):
                A[:3, 3 * col:3 * col + 3] = np.eye(3)
                c = self.limbs[i].contact - self.com
                A[3:, 3 * col:3 * col + 3] = _skew(c)
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            f = [sol[3 * k:3 * k + 3] for k in range(n)]

        worst_util, worst_i, min_margin = 0.0, -1, 1.0
        for k, i in enumerate(idxs):
            limb = self.limbs[i]
            hold = limb.hold
            fk = f[k]
            mag = np.linalg.norm(fk)
            f_max = hold.f_max_hand if i in HANDS else hold.f_max_foot
            util_mag = mag / max(f_max, 1e-6)
            if mag < 1e-6:
                ang_util = 0.0
                margin = 1.0
            else:
                cosang = float(np.dot(fk / mag, hold.pull_axis))
                # within-cone requirement: cosang >= cone_cos
                ang_util = max(0.0, (hold.cone_cos - cosang)) / max(1e-6, 1.0 - hold.cone_cos + 0.2)
                margin = (cosang - hold.cone_cos) / max(1e-6, 1.0 - hold.cone_cos)
            util = max(util_mag, ang_util)
            limb.grip_util = util
            report.forces.append(fk)
            report.utils.append(util)
            if util > worst_util:
                worst_util, worst_i = util, i
            margin = min(margin, 1.0 - util_mag)
            min_margin = min(min_margin, margin)

        report.worst_limb = worst_i
        report.min_margin = min_margin
        # slip if utilization exceeds 1 by a threshold (hard violation)
        if worst_util > 1.05 and worst_i >= 0:
            report.feasible = False
            self.slip_events.append(worst_i)
            self.release(worst_i)
        return report


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])
