import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .rendering import DroneRenderer

class DroneScoutingEnv(gym.Env):
    """
    AgroInsightX Drone Environment (Complex 12x12 with Compass Navigation)
    """
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, grid_size=(12, 12), n_waypoints=5, max_steps=500, seed=None, render_mode=None):
        super().__init__()
        self.grid_w, self.grid_h = grid_size
        self.n_waypoints = n_waypoints
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)

        # Actions: 0-3:Move, 4:Asc, 5:Desc, 6:Scan, 7:Hover, 8:Return
        self.action_space = spaces.Discrete(9)
        
        # OBS: [x, y, alt, bat, wx, wy, ...waypoints..., nf_prox, hot_prox, ret_intent, dx_target, dy_target]
        obs_dim = 9 + n_waypoints + 2 # Added 2 for compass (dx, dy)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self.alt_levels = [0, 1, 2]
        self.scan_radius_by_alt = {0: 0, 1: 1, 2: 2}
        self.renderer = None

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def _place_entities(self):
        self.base = (0, 0)
        forbidden = {self.base}
        
        # 1. Obstacles (10% density - Full Complexity)
        self.obstacles = set()
        num_obs = int((self.grid_w * self.grid_h) * 0.1)
        while len(self.obstacles) < num_obs:
            cell = (self.rng.integers(0, self.grid_w), self.rng.integers(0, self.grid_h))
            if cell not in forbidden:
                self.obstacles.add(cell)
                forbidden.add(cell)

        # 2. No-Fly Zones (5% density)
        self.no_fly = set()
        num_nf = int((self.grid_w * self.grid_h) * 0.05)
        while len(self.no_fly) < num_nf:
            cell = (self.rng.integers(0, self.grid_w), self.rng.integers(0, self.grid_h))
            if cell not in forbidden:
                self.no_fly.add(cell)
                forbidden.add(cell)

        # 3. Waypoints
        self.waypoints = []
        while len(self.waypoints) < self.n_waypoints:
            cell = (self.rng.integers(0, self.grid_w), self.rng.integers(0, self.grid_h))
            if cell not in forbidden and cell not in self.obstacles:
                self.waypoints.append(cell)
                forbidden.add(cell)

        self.hotspots = set()
        for wp in self.waypoints:
            if self.rng.random() < 0.5:
                self.hotspots.add(wp)

    def reset(self, seed=None, options=None):
        if seed is not None: self.seed(seed)
        self._place_entities()
        self.x, self.y = self.base
        self.alt = 0
        self.battery = 1.0
        self.steps = 0
        self.return_intent = 0
        self.detected_hotspots = 0
        self.wind = self._sample_wind()
        self.remaining = np.ones(self.n_waypoints, dtype=np.float32)
        return self._get_obs(), {}

    def _sample_wind(self):
        wx = np.clip(self.rng.normal(0, 0.3), -1, 1)
        wy = np.clip(self.rng.normal(0, 0.3), -1, 1)
        return np.array([wx, wy], dtype=np.float32)

    def _get_target_vector(self):
        # Find nearest active waypoint
        nearest_dist = float('inf')
        target = self.base # Default to base if done
        
        if self.remaining.sum() > 0:
            for i, wp in enumerate(self.waypoints):
                if self.remaining[i] == 1.0:
                    dist = abs(self.x - wp[0]) + abs(self.y - wp[1])
                    if dist < nearest_dist:
                        nearest_dist = dist
                        target = wp
        
        # Calculate normalized vector
        dx = (target[0] - self.x) / self.grid_w
        dy = (target[1] - self.y) / self.grid_h
        return dx, dy

    def _get_obs(self):
        norm_x = self.x / (self.grid_w - 1)
        norm_y = self.y / (self.grid_h - 1)
        norm_alt = self.alt / 2.0
        
        if not self.no_fly: prox_nf = 0.0
        else:
            dists = [abs(nf[0]-self.x) + abs(nf[1]-self.y) for nf in self.no_fly]
            prox_nf = 1.0 - (min(dists) / (self.grid_w + self.grid_h))

        # Get Compass Data
        dx_target, dy_target = self._get_target_vector()
            
        obs = [norm_x, norm_y, norm_alt, self.battery, self.wind[0], self.wind[1]]
        obs.extend(self.remaining)
        obs.extend([prox_nf, self.detected_hotspots / 5.0, float(self.return_intent)])
        # Add Compass
        obs.extend([dx_target, dy_target])
        
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        self.steps += 1
        reward = -0.001 
        terminated = False
        truncated = False
        info = {}
        
        # Calculate distance BEFORE move
        dx_old, dy_old = self._get_target_vector()
        dist_old = abs(dx_old) + abs(dy_old)

        COST_HOVER = 0.005
        COST_MOVE = 0.01
        COST_ALT = 0.02
        COST_SCAN = 0.01

        dx, dy = 0, 0
        if action == 0: dy = -1 
        elif action == 1: dy = 1  
        elif action == 2: dx = 1  
        elif action == 3: dx = -1 
        elif action == 4: 
            if self.alt < 2: self.alt += 1; self.battery -= COST_ALT
        elif action == 5: 
            if self.alt > 0: self.alt -= 1; self.battery -= COST_HOVER
        elif action == 6: 
            r_scan, scanned = self._perform_scan()
            reward += r_scan
            if scanned: self.battery -= COST_SCAN
        elif action == 7: self.battery -= COST_HOVER
        elif action == 8: 
            self.return_intent = 1; self.battery -= COST_HOVER

        # Wind & Move
        if action <= 3:
            drift_x, drift_y = 0, 0
            # Wind drift logic
            if abs(self.wind[0]) > 0.5 and self.rng.random() < 0.3: drift_x = int(np.sign(self.wind[0]))
            if abs(self.wind[1]) > 0.5 and self.rng.random() < 0.3: drift_y = int(np.sign(self.wind[1]))
            nx = np.clip(self.x + dx + drift_x, 0, self.grid_w - 1)
            ny = np.clip(self.y + dy + drift_y, 0, self.grid_h - 1)
            self.x, self.y = nx, ny
            self.battery -= (COST_MOVE + (0.005 * self.alt))

        # Calculate distance AFTER move
        dx_new, dy_new = self._get_target_vector()
        dist_new = abs(dx_new) + abs(dy_new)

        # SHAPED REWARD: Did we get closer to the target?
        if dist_new < dist_old:
            reward += 0.1 # Breadcrumb reward
        
        # Collisions
        if (self.x, self.y) in self.obstacles and self.alt == 0:
            reward -= 1.0
            terminated = True
            info["outcome"] = "crashed_obstacle"

        if (self.x, self.y) in self.no_fly:
            reward -= 0.2

        # Mission Success
        if self.remaining.sum() == 0 and (self.x, self.y) == self.base and self.alt == 0:
            reward += 50.0 
            terminated = True
            info["outcome"] = "mission_success"

        if self.battery <= 0:
            reward -= 1.0
            terminated = True
            info["outcome"] = "battery_depleted"
            
        if self.steps >= self.max_steps:
            truncated = True
            info["outcome"] = "timeout"

        # Dynamic Wind
        if self.steps % 10 == 0:
            self.wind = np.clip(self.wind + self.rng.normal(0, 0.1, 2), -1, 1)

        return self._get_obs(), reward, terminated, truncated, info

    def _perform_scan(self):
        reward = 0
        scan_radius = self.scan_radius_by_alt[self.alt]
        scanned_something = False
        for i, wp in enumerate(self.waypoints):
            if self.remaining[i] == 1.0:
                dist = abs(wp[0] - self.x) + abs(wp[1] - self.y)
                if dist <= scan_radius:
                    self.remaining[i] = 0.0
                    reward += 5.0 
                    scanned_something = True
                    if wp in self.hotspots:
                        reward += 2.0
                        self.detected_hotspots += 1
        return reward, scanned_something

    def render(self):
        if self.render_mode == "human":
            if self.renderer is None: self.renderer = DroneRenderer(self.grid_w, self.grid_h)
            self.renderer.draw(self)

    def close(self):
        if self.renderer: self.renderer.close()