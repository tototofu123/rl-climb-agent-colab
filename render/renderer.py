import pygame
import numpy as np


class Renderer:
    """Draws a human stick figure climbing with 2 arms and 2 legs."""

    def __init__(self, wall, title="RL Climb"):
        self.wall = wall
        pygame.init()
        self.screen = pygame.display.set_mode((wall.width, wall.height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 22)
        self.big = pygame.font.SysFont(None, 30)
        self._closed = False
        self.arm_len = 70
        self.leg_len = 80
        # Default angles for free limbs (degrees, 0=right, 90=down)
        self.free_arm_a = [155, 25]
        self.free_leg_a = [120, 60]

    def _check_quit(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self._closed = True

    def render_physics(self, env):
        if self._closed:
            return
        self._check_quit()
        if self._closed:
            return

        w, h = self.wall.width, self.wall.height
        self.screen.fill((40, 40, 45))
        pygame.draw.line(self.screen, (100, 100, 100), (0, h), (w, h), 2)

        # --- Holds ---
        for hold in self.wall.holds:
            c = (255, 215, 0) if hold.is_target else (70, 140, 220)
            r = hold.radius
            rect = (int(hold.pos[0]) - r, int(hold.pos[1]) - r, r * 2, r * 2)
            pygame.draw.rect(self.screen, c, rect, border_radius=4)
            pygame.draw.rect(self.screen, (200, 200, 200), rect, 2, border_radius=4)

        # Label START and GOAL
        for hold in self.wall.holds:
            hx, hy = int(hold.pos[0]), int(hold.pos[1])
            if hold.is_target:
                self.screen.blit(self.font.render("GOAL", True, (255, 215, 0)), (hx - 24, hy - hold.radius - 22))
        start = max(self.wall.holds, key=lambda h: h.pos[1])
        self.screen.blit(self.font.render("START", True, (140, 220, 140)), (int(start.pos[0]) - 26, int(start.pos[1]) + start.radius + 6))

        # --- Body ---
        in_dyno = getattr(env.body, "in_dyno", False)
        bx = int(np.clip(env.body.pos[0], 0, w))
        by = int(np.clip(env.body.pos[1], 0, h))

        # Joint positions
        head = (bx, by - 26)
        neck = (bx, by - 12)
        lsh = (bx - 16, by - 6)
        rsh = (bx + 16, by - 6)
        lhip = (bx - 12, by + 14)
        rhip = (bx + 12, by + 14)
        joints = [lsh, rsh, lhip, rhip]

        # Torso (neck to mid-hip)
        mid_hip = ((lhip[0] + rhip[0]) // 2, (lhip[1] + rhip[1]) // 2)
        pygame.draw.line(self.screen, (200, 200, 200), neck, mid_hip, 5)

        # Hanging lines from attached holds → shoulder/hip to show "hanging" not floating
        for i, limb in enumerate(env.body.limbs):
            if limb.attached and limb.hold_pos is not None:
                jx2, jy2 = joints[i]
                ddx = limb.hold_pos[0] - jx2
                ddy = limb.hold_pos[1] - jy2
                dd = np.sqrt(ddx*ddx + ddy*ddy)
                if dd > 5:
                    nx = -ddy / dd
                    ny = ddx / dd
                    # Draw a small "hold" bracket at the hold
                    hx, hy = int(limb.hold_pos[0]), int(limb.hold_pos[1])
                    br = 8
                    pygame.draw.line(self.screen, (200, 200, 200), (hx-br, hy), (hx+br, hy), 2)
                    pygame.draw.line(self.screen, (200, 200, 200), (hx, hy), (hx, hy+br), 2)

        # Limbs (fixed length: arms 70, legs 80)
        for i, limb in enumerate(env.body.limbs):
            jx, jy = joints[i]
            max_l = self.arm_len if i < 2 else self.leg_len

            if limb.attached and limb.hold_pos is not None:
                dx = limb.hold_pos[0] - jx
                dy = limb.hold_pos[1] - jy
                d = np.sqrt(dx * dx + dy * dy)
                if d > max_l and d > 1:
                    # Cap at max limb length, point toward hold
                    ex = int(jx + dx / d * max_l)
                    ey = int(jy + dy / d * max_l)
                else:
                    ex, ey = int(limb.hold_pos[0]), int(limb.hold_pos[1])
                c = (50, 230, 80)
                thick = 6
                pygame.draw.circle(self.screen, c, (ex, ey), 5)
            else:
                a = np.radians(self.free_arm_a[i] if i < 2 else self.free_leg_a[i - 2])
                ex = int(jx + max_l * np.cos(a))
                ey = int(jy + max_l * np.sin(a))
                c = (160, 80, 80)
                thick = 3
                # Small circle at hand/foot
                pygame.draw.circle(self.screen, c, (ex, ey), 4)
            pygame.draw.line(self.screen, c, (jx, jy), (ex, ey), thick)

        # Head
        pygame.draw.circle(self.screen, (255, 190, 140), head, 12)
        pygame.draw.circle(self.screen, (255, 255, 255), head, 12, 2)

        # --- Info ---
        status = "DYNO" if in_dyno else "CLIMB"
        sc = (255, 200, 50) if in_dyno else (100, 255, 100)
        self.screen.blit(self.big.render(f" {status} ", True, (0, 0, 0), sc), (w // 2 - 40, 6))

        dist = np.linalg.norm(env.body.pos - env.target_pos)
        for i, t in enumerate([
            f"Step {env.step_count}  Holds {env.body.num_attached_limbs()}/4",
            f"Dist {dist:.0f}  ({env.body.pos[0]:.0f},{env.body.pos[1]:.0f})",
        ]):
            self.screen.blit(self.font.render(t, True, (200, 200, 200)), (10, 36 + i * 22))

        falling = getattr(env.body, 'is_falling', lambda: False)()
        if falling:
            self.screen.blit(self.big.render("FELL", True, (255, 60, 60)), (w // 2 - 32, h // 2 - 14))

        pygame.display.flip()
        self.clock.tick(env.metadata["render_fps"])

    def render_simple(self, env):
        if self._closed:
            return
        self._check_quit()
        if self._closed:
            return
        w, h = self.wall.width, self.wall.height
        self.screen.fill((40, 40, 40))
        for hold in self.wall.holds:
            c = (255, 215, 0) if hold.is_target else (100, 150, 255)
            r = hold.radius
            pygame.draw.rect(self.screen, c, (int(hold.pos[0]) - r, int(hold.pos[1]) - r, r * 2, r * 2), border_radius=2)
            pygame.draw.rect(self.screen, (40, 40, 40), (int(hold.pos[0]) - r, int(hold.pos[1]) - r, r * 2, r * 2), 2, border_radius=2)
        ap = (int(np.clip(env.agent_pos[0], 0, w)), int(np.clip(env.agent_pos[1], 0, h)))
        pygame.draw.circle(self.screen, (220, 50, 50), ap, 12)
        pygame.draw.circle(self.screen, (40, 40, 40), ap, 12, 2)
        for i, t in enumerate([f"Step {env.step_count}"]):
            self.screen.blit(self.font.render(t, True, (200, 200, 200)), (10, 10 + i * 22))
        pygame.display.flip()
        self.clock.tick(env.metadata["render_fps"])

    @property
    def closed(self):
        return self._closed

    def close(self):
        if not self._closed:
            pygame.quit()
            self._closed = True
