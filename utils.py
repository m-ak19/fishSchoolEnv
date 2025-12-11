import math
import pygame
from typing import Literal

def shortest_angle_diff(target_angle, current_angle):
    diff = target_angle - current_angle
    diff = (diff + math.pi) % (2 * math.pi) - math.pi
    return diff


def draw_fish(screen, fish, reward : float, is_agent=False, show_vision_cone=False):
    x = int(fish.pos.x)
    y = int(fish.pos.y)

    # --- DESSIN DU CORPS ---
    length = 10
    dx = math.cos(fish.dir.angle)
    dy = math.sin(fish.dir.angle)

    tip = (int(x + dx * length), int(y + dy * length))
    left = (int(x + math.cos(fish.dir.angle + 2.5) * length * 0.7),
            int(y + math.sin(fish.dir.angle + 2.5) * length * 0.7))
    right = (int(x + math.cos(fish.dir.angle - 2.5) * length * 0.7),
             int(y + math.sin(fish.dir.angle - 2.5) * length * 0.7))

    color = (50, 50, 200)
    if is_agent:
        color = (255, 0, 0) if reward < 0 else (0, 255, 255)
    pygame.draw.polygon(screen, color, [tip, left, right])

    # ============================
    #  DESSIN DU CONE DE VISION
    # ============================
    if show_vision_cone and is_agent:
        vr = fish.params.vision_range
        half_angle = fish.vision_angle / 2

        # Bornes du cône
        angle_left = fish.dir.angle + half_angle
        angle_right = fish.dir.angle - half_angle

        # Points extrêmes
        p_left = (int(x + math.cos(angle_left) * vr),
                  int(y + math.sin(angle_left) * vr))
        p_right = (int(x + math.cos(angle_right) * vr),
                   int(y + math.sin(angle_right) * vr))

        # Couleur du cône
        cone_color = (80, 180, 255, 50)

        # Rayons
        pygame.draw.line(screen, (80, 180, 255), (x, y), p_left, 1)
        pygame.draw.line(screen, (80, 180, 255), (x, y), p_right, 1)

        # ---- ARC du cône (discretisé) ----
        arc_points = []
        steps = 25
        for i in range(steps + 1):
            a = angle_right + (i / steps) * (angle_left - angle_right)
            px = int(x + math.cos(a) * vr)
            py = int(y + math.sin(a) * vr)
            arc_points.append((px, py))

        # Dessine le contour de l’arc
        for i in range(len(arc_points) - 1):
            pygame.draw.line(screen, (80, 180, 255), arc_points[i], arc_points[i+1], 1)

        # Optionnel : remplir légèrement
        # pygame.draw.polygon(screen, (80, 180, 255, 40), [(x,y)] + arc_points)


def show_text(
            text,
            pos,
            screen: pygame.surface.Surface,
            font: pygame.font.SysFont,
            font_color: tuple[Literal[255], Literal[255], Literal[255]],
            ):
    text_surface = font.render(text, True, font_color)
    screen.blit(text_surface, pos)


class Slider:
    def __init__(self, x, y, width, min_val, max_val, value, label, font, color):
        self.rect = pygame.Rect(x, y, width, 20)  # zone "barre"
        self.min_val = min_val
        self.max_val = max_val
        self.value = value
        self.label = label
        self.font = font
        self.color = color
        self.handle_radius = 6
        self.dragging = False

    def val_to_x(self):
        """Convertit la valeur en position x du handle."""
        t = (self.value - self.min_val) / (self.max_val - self.min_val)
        return int(self.rect.x + t * self.rect.width)

    def x_to_val(self, mouse_x):
        """Convertit une position x de la souris en valeur."""
        t = (mouse_x - self.rect.x) / self.rect.width
        t = max(0.0, min(1.0, t))
        return self.min_val + t * (self.max_val - self.min_val)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # clic gauche
                # Si on clique proche du handle, on commence à drag
                mx, my = event.pos
                hx = self.val_to_x()
                hy = self.rect.y + self.rect.height // 2
                if (mx - hx)**2 + (my - hy)**2 <= (self.handle_radius + 3)**2:
                    self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                mx, _ = event.pos
                self.value = self.x_to_val(mx)

    def draw(self, surface):
        # Barre
        pygame.draw.rect(surface, (100, 100, 100), self.rect, border_radius=3)
        # Handle
        hx = self.val_to_x()
        hy = self.rect.y + self.rect.height // 2
        pygame.draw.circle(surface, (200, 200, 255), (hx, hy), self.handle_radius)
        # Label + valeur
        text_surf = self.font.render(
            f"{self.label}: {self.value:.3f}", True, (230, 230, 230)
        )
        surface.blit(text_surf, (self.rect.x, self.rect.y - 18))