import numpy as np


LIMB_NAMES = ["L-arm", "R-arm", "L-leg", "R-leg"]
# Joint offsets from body center: (dx, dy)
JOINT_OFFSETS = [
    (-16, -6),   # L shoulder
    (16, -6),    # R shoulder
    (-12, 14),   # L hip
    (12, 14),    # R hip
]
ARM_LEN = 70
LEG_LEN = 80
MAX_REACH = 150.0
MIN_REACH = 15


class LimbState:
    def __init__(self):
        self.attached = False
        self.hold_pos = None


class ClimberBody:
    def __init__(self, start_pos):
        self.pos = np.array(start_pos, dtype=np.float32)
        self.vel = np.array([0.0, 0.0], dtype=np.float32)
        self.limbs = [LimbState() for _ in range(4)]
        self.gravity = np.array([0.0, 300.0], dtype=np.float32)
        self.damping = 0.993
        self.dyno_steps = 0

    def joint_world(self, limb_idx):
        ox, oy = JOINT_OFFSETS[limb_idx]
        return np.array([self.pos[0] + ox, self.pos[1] + oy], dtype=np.float32)

    def step(self, dt, wall_width=None, wall_height=None):
        self.vel += self.gravity * dt
        self.vel *= self.damping
        self.pos += self.vel * dt
        self._enforce_constraints()
        if wall_height is not None:
            self.pos[1] = np.clip(self.pos[1], 0, wall_height)
            if self.pos[1] <= 0 and self.vel[1] < 0:
                self.vel[1] = 0
            if self.pos[1] >= wall_height and self.vel[1] > 0:
                self.vel[1] = 0
        if wall_width is not None:
            self.pos[0] = np.clip(self.pos[0], 0, wall_width)
            if self.pos[0] <= 0 and self.vel[0] < 0:
                self.vel[0] = 0
            if self.pos[0] >= wall_width and self.vel[0] > 0:
                self.vel[0] = 0
        if self.dyno_steps > 0:
            self.dyno_steps -= 1

    def dyno_impulse(self, upward_velocity=400.0, dyno_duration=20):
        for limb in self.limbs:
            limb.attached = False
            limb.hold_pos = None
        self.vel += np.array([0.0, -upward_velocity], dtype=np.float32)
        self.dyno_steps = dyno_duration

    @property
    def in_dyno(self):
        return self.dyno_steps > 0

    def _enforce_constraints(self):
        for i, limb in enumerate(self.limbs):
            if limb.attached and limb.hold_pos is not None:
                joint = self.joint_world(i)
                hp = np.array(limb.hold_pos, dtype=np.float32)
                delta = joint - hp
                dist = np.linalg.norm(delta)
                if dist < MIN_REACH and dist > 1e-6:
                    direction = delta / dist
                    self.pos += (MIN_REACH - dist) * direction
                elif dist > MAX_REACH and dist > 1e-6:
                    direction = delta / dist
                    correction = (dist - MAX_REACH) * direction
                    self.pos -= correction
                    v_radial = np.dot(self.vel, direction) * direction
                    self.vel -= v_radial * 0.85

    def has_free_limb(self):
        return any(not limb.attached for limb in self.limbs)

    def get_free_limb(self):
        for i, limb in enumerate(self.limbs):
            if not limb.attached:
                return i
        return None

    def reach_hold(self, hold_pos, max_reach=None):
        limb_idx = self.get_free_limb()
        if limb_idx is None:
            return False
        limb = self.limbs[limb_idx]
        joint = self.joint_world(limb_idx)
        hp = np.array(hold_pos, dtype=np.float32)
        dist = np.linalg.norm(hp - joint)
        reach = max_reach or MAX_REACH
        if dist <= reach:
            limb.attached = True
            limb.hold_pos = tuple(hold_pos)
            return True
        return False

    def release_limb(self, limb_idx):
        if 0 <= limb_idx < len(self.limbs):
            self.limbs[limb_idx].attached = False
            self.limbs[limb_idx].hold_pos = None
            return True
        return False

    def num_attached_limbs(self):
        return sum(1 for limb in self.limbs if limb.attached)

    def is_falling(self):
        return not self.in_dyno and self.num_attached_limbs() == 0

    def init_full_grip(self, hold_positions):
        """Attach all 4 limbs to given hold positions (list of 4 (x,y) tuples)."""
        count = 0
        for i in range(4):
            if i < len(hold_positions):
                joint = self.joint_world(i)
                hp = np.array(hold_positions[i], dtype=np.float32)
                dist = np.linalg.norm(hp - joint)
                if dist <= MAX_REACH:
                    self.limbs[i].attached = True
                    self.limbs[i].hold_pos = tuple(hold_positions[i])
                    count += 1
        return count > 0
