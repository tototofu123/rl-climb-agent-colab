"""Pseudo-3D renderer: perspective projection of the wall, holds and climber.

The camera sits off-axis (to the right and out from the wall) so depth is
visible: holds on the side walls and the climber's lean out of the wall plane
read clearly. Depth is additionally encoded by brightness (nearer = brighter)
and painter-sorted drawing.
"""

import os

import numpy as np

WALL_COLORS = {
    "main": (168, 148, 118),
    "left": (140, 122, 96),
    "right": (150, 132, 104),
}
TYPE_COLORS = {
    "jug": (255, 140, 0),
    "crimp": (220, 40, 40),
    "sloper": (240, 210, 60),
    "pocket": (170, 80, 220),
    "pinch": (60, 130, 240),
    "volume": (150, 90, 50),
    "jib": (160, 160, 160),
    "uncategorized": (90, 190, 90),
}


class Camera:
    def __init__(self, eye, target, up=(0.0, 1.0, 0.0), focal=750.0):
        self.eye = np.asarray(eye, dtype=np.float64)
        self.target = np.asarray(target, dtype=np.float64)
        self.focal = focal
        f = self.target - self.eye
        self.fwd = f / np.linalg.norm(f)
        right = np.cross(self.fwd, np.asarray(up, dtype=np.float64))
        self.right = right / np.linalg.norm(right)
        self.up = np.cross(self.right, self.fwd)

    def project(self, p):
        p = np.asarray(p, dtype=np.float64) - self.eye
        x = float(np.dot(p, self.right))
        y = float(np.dot(p, self.up))
        z = float(np.dot(p, self.fwd))
        if z < 0.05:
            z = 0.05
        return x, y, z


def _shade(color, depth01):
    """Nearer (larger z out of wall) -> brighter."""
    f = 0.55 + 0.45 * float(np.clip(depth01, 0.0, 1.0))
    return tuple(int(c * f) for c in color)


class Renderer3D:
    def __init__(self, width=1100, height=750, headless=False):
        import pygame

        self.pygame = pygame
        self.width = width
        self.height = height
        self.headless = headless
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        pygame.display.set_caption("RL Climb 3D — HKUST boulder wall")
        self.screen = pygame.display.set_mode((width, height)) if not headless \
            else pygame.Surface((width, height))
        self.font = pygame.font.SysFont("consolas", 16)
        self.small = pygame.font.SysFont("consolas", 13)
        self.clock = pygame.time.Clock()

    # ---- helpers -----------------------------------------------------------
    def _to_screen(self, cam, p, cx, cy):
        x, y, z = cam.project(p)
        return (cx + cam.focal * x / z, cy - cam.focal * y / z), z

    def _polygon(self, surface, cam, pts, color, cx, cy, width=0):
        proj = [self._to_screen(cam, p, cx, cy) for p in pts]
        poly = [(p[0][0], p[0][1]) for p in proj]
        try:
            self.pygame.draw.polygon(surface, color, poly, width)
        except ValueError:
            pass
        return float(np.mean([p[1] for p in proj]))

    def _polyline3d(self, surface, cam, pts, color, cx, cy, width, depth01=0.5):
        proj = [self._to_screen(cam, p, cx, cy) for p in pts]
        for (p1, _), (p2, _) in zip(proj[:-1], proj[1:]):
            try:
                self.pygame.draw.line(surface, _shade(color, depth01), p1, p2, width)
            except ValueError:
                pass

    def _circle3d(self, surface, cam, p, radius_m, color, cx, cy, ring=None):
        (sx, sy), z = self._to_screen(cam, p, cx, cy)
        r = max(2, int(cam.focal * radius_m / z))
        try:
            self.pygame.draw.circle(surface, color, (int(sx), int(sy)), r)
            if ring:
                self.pygame.draw.circle(surface, ring, (int(sx), int(sy)), r + 3, 2)
        except ValueError:
            pass

    def _hold3d(self, surface, cam, hold, fill, cx, cy, ring=None):
        """Draw a hold: its real SVG outline projected through the camera,
        falling back to a circle when no outline is available."""
        if hold.outline is not None and len(hold.outline) >= 3:
            poly = []
            for p in hold.outline:
                (sx, sy), z = self._to_screen(cam, p, cx, cy)
                poly.append((sx, sy))
            try:
                self.pygame.draw.polygon(surface, fill, poly)
                if ring:
                    # scale ring thickness with apparent size
                    r0 = max(3, int(cam.focal * hold.radius / cam.project(hold.pos)[2]))
                    self.pygame.draw.polygon(surface, ring, poly, max(2, r0 // 8 + 2))
            except ValueError:
                pass
            return
        self._circle3d(surface, cam, hold.pos, max(hold.radius, 0.03), fill, cx, cy, ring)

    # ---- frame -------------------------------------------------------------
    def render(self, env, extra_hud=None):
        pg = self.pygame
        surf = self.screen
        b = env.body
        com = b.com

        cam = Camera(
            eye=com + np.array([2.2, 0.45, 4.6]),
            target=com + np.array([0.0, 0.1, -0.3]),
            focal=0.62 * self.width,
        )
        cx, cy = self.width * 0.44, self.height * 0.52

        surf.fill((24, 26, 32))

        # ---- wall sections (painter: far first) ----
        quads = []
        for section in ("left", "right", "main"):
            quad = env.wall.wall_quad(section)
            center = np.mean(quad, axis=0)
            zc = cam.project(center)[2]
            quads.append((zc, section, quad))
        quads.sort(key=lambda t: -t[0])
        for _, section, quad in quads:
            color = WALL_COLORS[section]
            self._polygon(surf, cam, quad, color, cx, cy)

        # mat
        mat_pts = [np.array([x, 0.0, z])
                   for x in (-0.8, env.wall.width + 0.8)
                   for z in (0.0, 2.2)]
        self._polygon(surf, cam,
                      [np.array([-0.8, 0.0, 0.0]), np.array([env.wall.width + 0.8, 0.0, 0.0]),
                       np.array([env.wall.width + 0.8, 0.0, 2.2]), np.array([-0.8, 0.0, 2.2])],
                      (60, 70, 80), cx, cy)

        # ---- holds ----
        task_ids = {h.id for h in env.task_holds}
        draws = []
        for h in env.wall.holds:
            zc = cam.project(h.pos)[2]
            on_task = h.id in task_ids
            color = TYPE_COLORS.get(h.hold_type, (120, 120, 120))
            if not on_task:
                color = (105, 100, 95)
            depth01 = h.pos[2] / 1.0 if h.wall_section == "main" else 0.5
            draws.append((zc, h, color, on_task, depth01))
        draws.sort(key=lambda t: -t[0])
        target_id = env.target_hold.id
        for zc, h, color, on_task, depth01 in draws:
            bright = _shade(color, min(1.0, depth01 + 0.35 if on_task else depth01 * 0.6))
            ring = None
            if on_task:
                ring = (230, 230, 230)
            if h.id == target_id:
                ring = (255, 60, 220)
                bright = _shade(TYPE_COLORS.get(h.hold_type, (200, 200, 200)), 1.0)
            held = any(l.attached and l.hold is not None and l.hold.id == h.id
                       for l in b.limbs)
            if held:
                ring = (90, 255, 90)
            self._hold3d(surf, cam, h, bright, cx, cy, ring)

        # ---- climber ----
        sk = b.draw_skeleton()
        torso_c = (250, 250, 255) if not b.in_dyno else (255, 170, 60)
        lean_deg = np.degrees(b.lean_out)
        depth01 = float(np.clip(b.com[2] / 1.2, 0.0, 1.0))
        # legs first (behind torso on average)
        for limb in sk["limbs"]:
            if limb["is_hand"]:
                continue
            color = (120, 200, 255) if limb["attached"] else (150, 120, 90)
            self._polyline3d(surf, cam, [limb["root"], limb["mid"], limb["end"]],
                             color, cx, cy, 8, depth01)
            self._circle3d(surf, cam, limb["end"], 0.045, color, cx, cy)
        self._polyline3d(surf, cam, [sk["hips"][0], sk["hips"][1]],
                         torso_c, cx, cy, 7, depth01)
        self._polyline3d(surf, cam, [sk["shoulders"][0], sk["shoulders"][1]],
                         torso_c, cx, cy, 7, depth01)
        self._polyline3d(surf, cam, [sk["pelvis"], sk["neck"]],
                         torso_c, cx, cy, 12, depth01)
        self._circle3d(surf, cam, sk["head"], 0.11, torso_c, cx, cy)
        for limb in sk["limbs"]:
            if not limb["is_hand"]:
                continue
            color = (120, 200, 255) if limb["attached"] else (150, 120, 90)
            self._polyline3d(surf, cam, [limb["root"], limb["mid"], limb["end"]],
                             color, cx, cy, 6, depth01)
            self._circle3d(surf, cam, limb["end"], 0.04, color, cx, cy)
        # lean indicator: com -> along torso up
        self._polyline3d(surf, cam, [b.com, b.com + sk["torso_up"] * 0.45],
                         (255, 120, 120), cx, cy, 2, depth01)

        # ---- HUD ----
        hud = [
            f"task {info_task(env)}   step {env.step_count}/{env.max_steps}",
            f"com  x={b.com[0]:.2f} y={b.com[1]:.2f} z={b.com[2]:.2f} m   lean {lean_deg:+.0f} deg",
            f"limbs attached {b.num_attached()}/4   grip util "
            + " ".join(f"{l.grip_util:.2f}" if l.attached else " -- " for l in b.limbs),
        ]
        if extra_hud:
            hud = list(extra_hud) + hud
        y = 10
        for line in hud:
            surf.blit(self.font.render(line, True, (235, 235, 235)), (12, y))
            y += 20
        if b.in_dyno:
            surf.blit(self.font.render("DYNO!", True, (255, 170, 60)),
                      (self.width - 110, 12))

        if not self.headless:
            pg.display.flip()
        return surf

    def save_png(self, path):
        try:
            self.pygame.image.save(self.screen, path)
        except self.pygame.error:
            pass

    def quit(self):
        self.pygame.quit()


def info_task(env):
    return getattr(env, "task_label", f"seed {getattr(env, 'route_seed', '?')}")


def render_frame_png(env, path, width=1100, height=750):
    """Headless single-frame render for tests / CI."""
    r = Renderer3D(width, height, headless=True)
    try:
        r.render(env)
        r.save_png(path)
    finally:
        r.quit()
    return path
