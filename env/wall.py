from dataclasses import dataclass, field
from typing import List
import copy
import numpy as np


@dataclass
class Hold:
    pos: tuple
    radius: float = 15
    is_target: bool = False
    hold_type: str = "jug"


class Wall:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.holds: List[Hold] = []

    def add_hold(self, x, y, radius=15, is_target=False, hold_type="jug"):
        self.holds.append(Hold((x, y), radius, is_target, hold_type))

    @property
    def target(self):
        for h in self.holds:
            if h.is_target:
                return h
        return None

    def clone_randomized(self, max_offset=15, rng=None):
        """Return a copy with small random offsets to non-target hold positions."""
        rng = rng or np.random.default_rng()
        new = Wall(self.width, self.height)
        for h in self.holds:
            if h.is_target:
                new.add_hold(h.pos[0], h.pos[1], h.radius, True, h.hold_type)
            else:
                dx = rng.uniform(-max_offset, max_offset)
                dy = rng.uniform(-max_offset, max_offset)
                nx = np.clip(h.pos[0] + dx, 30, self.width - 30)
                ny = np.clip(h.pos[1] + dy, 30, self.height - 30)
                new.add_hold(nx, ny, h.radius, False, h.hold_type)
        return new

    @classmethod
    def beginner(cls):
        wall = cls()
        wall.add_hold(400, 80, radius=20, is_target=True)      # target
        wall.add_hold(340, 160, radius=18)                     # intermediate
        wall.add_hold(460, 160, radius=18)
        wall.add_hold(340, 260, radius=18)
        wall.add_hold(460, 260, radius=18)
        wall.add_hold(340, 360, radius=18)
        wall.add_hold(460, 360, radius=18)
        wall.add_hold(340, 460, radius=18)                     # start hand L
        wall.add_hold(460, 460, radius=18)                     # start hand R
        wall.add_hold(380, 520, radius=16)                     # start foot L
        wall.add_hold(420, 520, radius=16)                     # start foot R
        return wall

    @classmethod
    def intermediate(cls):
        wall = cls()
        wall.add_hold(400, 60, radius=20, is_target=True)      # target
        wall.add_hold(280, 140, radius=18)
        wall.add_hold(520, 140, radius=16, hold_type="crimp")
        wall.add_hold(220, 260, radius=18)
        wall.add_hold(580, 260, radius=16, hold_type="crimp")
        wall.add_hold(300, 360, radius=18)
        wall.add_hold(500, 360, radius=16, hold_type="crimp")
        wall.add_hold(380, 440, radius=18)
        wall.add_hold(340, 480, radius=18)                     # start hand L
        wall.add_hold(460, 480, radius=18)                     # start hand R
        wall.add_hold(380, 530, radius=16)                     # start foot L
        wall.add_hold(420, 530, radius=16)                     # start foot R
        return wall

    @classmethod
    def dyno_gap(cls):
        wall = cls()
        wall.add_hold(400, 60, radius=22, is_target=True)      # target
        wall.add_hold(400, 220, radius=20)                     # dyno gap 160 from target
        wall.add_hold(400, 400, radius=20)                     # dyno gap 180 from 220
        wall.add_hold(350, 490, radius=18)                     # start hand L
        wall.add_hold(450, 490, radius=18)                     # start hand R
        wall.add_hold(380, 540, radius=16)                     # start foot L
        wall.add_hold(420, 540, radius=16)                     # start foot R
        return wall

    @classmethod
    def expert(cls):
        """Wide holds, overhang section, requires path planning + occasional dynos."""
        wall = cls()
        wall.add_hold(400, 50, radius=20, is_target=True)      # target
        wall.add_hold(200, 130, radius=16, hold_type="crimp")
        wall.add_hold(400, 130, radius=18)
        wall.add_hold(600, 130, radius=16, hold_type="crimp")
        wall.add_hold(150, 260, radius=16, hold_type="crimp")
        wall.add_hold(350, 260, radius=18)
        wall.add_hold(550, 260, radius=16, hold_type="crimp")
        wall.add_hold(250, 390, radius=18)
        wall.add_hold(450, 390, radius=18)
        wall.add_hold(350, 470, radius=18)                     # start hand L
        wall.add_hold(500, 470, radius=18)                     # start hand R
        wall.add_hold(300, 530, radius=16)                     # start foot L
        wall.add_hold(550, 530, radius=16)                     # start foot R
        return wall
