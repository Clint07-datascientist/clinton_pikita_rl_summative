import pygame
import numpy as np

CELL_SIZE = 40
MARGIN = 2
HUD_HEIGHT = 120
WIDTH_PADDING = 20

# Colors
COLOR_BG = (30, 30, 30)
COLOR_GRID = (50, 50, 50)
COLOR_OBSTACLE = (100, 100, 100)
COLOR_NO_FLY = (200, 50, 50)
COLOR_BASE = (50, 200, 50)
COLOR_WAYPOINT = (0, 150, 255)
COLOR_SCANNED = (0, 100, 100)
COLOR_HOTSPOT = (255, 200, 0)
COLOR_DRONE = (255, 255, 255)
COLOR_TEXT = (220, 220, 220)

class DroneRenderer:
    def __init__(self, grid_w, grid_h):
        pygame.init()
        self.grid_w = grid_w
        self.grid_h = grid_h
        
        self.width = grid_w * (CELL_SIZE + MARGIN) + MARGIN + WIDTH_PADDING
        self.height = grid_h * (CELL_SIZE + MARGIN) + MARGIN + HUD_HEIGHT
        
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("AgroInsightX: Drone Scouting Mission")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 14)
        self.header_font = pygame.font.SysFont("Arial", 18, bold=True)
        
        # Trail memory
        self.trail = []

    def draw_cell(self, x, y, color, alpha=255):
        rect = pygame.Rect(
            MARGIN + x * (CELL_SIZE + MARGIN) + WIDTH_PADDING // 2,
            MARGIN + y * (CELL_SIZE + MARGIN) + WIDTH_PADDING // 2,
            CELL_SIZE,
            CELL_SIZE
        )
        surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
        surface.fill((*color, alpha))
        self.screen.blit(surface, rect)
        return rect.center

    def draw(self, env, fps=10):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        self.screen.fill(COLOR_BG)

        # Draw Map
        for x in range(self.grid_w):
            for y in range(self.grid_h):
                if (x, y) in env.no_fly: self.draw_cell(x, y, COLOR_NO_FLY, alpha=150)
                elif (x, y) in env.obstacles: self.draw_cell(x, y, COLOR_OBSTACLE)
                elif (x, y) == env.base: self.draw_cell(x, y, COLOR_BASE)
                else: self.draw_cell(x, y, COLOR_GRID)

        # Draw Waypoints
        for idx, wp in enumerate(env.waypoints):
            is_active = (env.remaining[idx] == 1.0)
            color = COLOR_WAYPOINT if is_active else COLOR_SCANNED
            alpha = 255 if is_active else 100
            center = self.draw_cell(wp[0], wp[1], color, alpha=alpha)
            if is_active:
                text = self.font.render(str(idx+1), True, (255, 255, 255))
                self.screen.blit(text, (center[0]-5, center[1]-8))

        for hx, hy in env.hotspots:
             pygame.draw.circle(self.screen, COLOR_HOTSPOT, 
                                (MARGIN + hx*(CELL_SIZE+MARGIN) + WIDTH_PADDING//2 + 5, 
                                 MARGIN + hy*(CELL_SIZE+MARGIN) + WIDTH_PADDING//2 + 5), 4)

        # Draw Drone & Trail
        dx, dy = env.x, env.y
        drone_x = MARGIN + dx * (CELL_SIZE + MARGIN) + WIDTH_PADDING // 2 + CELL_SIZE // 2
        drone_y = MARGIN + dy * (CELL_SIZE + MARGIN) + WIDTH_PADDING // 2 + CELL_SIZE // 2
        
        self.trail.append((drone_x, drone_y))
        if len(self.trail) > 30: self.trail.pop(0)
        if len(self.trail) > 1:
            pygame.draw.lines(self.screen, (0, 255, 255), False, self.trail, 2)

        pygame.draw.circle(self.screen, COLOR_DRONE, (drone_x, drone_y), 8)
        ring_radius = 12 + (env.alt * 6)
        pygame.draw.circle(self.screen, (0, 255, 255), (drone_x, drone_y), ring_radius, width=2)

        # HUD
        hud_y = self.height - HUD_HEIGHT + 10
        batt_color = (255, 50, 50) if env.battery < 0.2 else COLOR_TEXT
        self.screen.blit(self.header_font.render("Mission Status", True, COLOR_TEXT), (20, hud_y))
        self.screen.blit(self.font.render(f"Battery: {int(env.battery*100)}%", True, batt_color), (20, hud_y + 25))
        self.screen.blit(self.font.render(f"Altitude: Level {env.alt}", True, COLOR_TEXT), (20, hud_y + 45))
        self.screen.blit(self.font.render(f"Steps: {env.steps}", True, COLOR_TEXT), (20, hud_y + 65))

        self.screen.blit(self.header_font.render("Navigation", True, COLOR_TEXT), (200, hud_y))
        self.screen.blit(self.font.render(f"Wind: ({env.wind[0]:.2f}, {env.wind[1]:.2f})", True, (100, 200, 255)), (200, hud_y + 25))
        self.screen.blit(self.font.render(f"Targets Left: {int(env.remaining.sum())}", True, COLOR_TEXT), (200, hud_y + 45))
        
        self.screen.blit(self.header_font.render("Intel", True, COLOR_TEXT), (400, hud_y))
        self.screen.blit(self.font.render(f"Hotspots: {env.detected_hotspots}", True, COLOR_HOTSPOT), (400, hud_y + 25))
        return_color = (50, 255, 50) if env.return_intent else (100, 100, 100)
        self.screen.blit(self.font.render(f"Return Mode: {'ON' if env.return_intent else 'OFF'}", True, return_color), (400, hud_y + 45))

        pygame.display.flip()
        self.clock.tick(fps)

    def close(self):
        pygame.quit()