import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .rendering import DroneRenderer

class DroneScoutingEnv(gym.Env):
    """
    AgroInsightX Drone Environment
    Mission: Scan waypoints, detect hotspots, avoid no-fly zones, return to base.
    """
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, grid_size=(12, 12), n_waypoints=5, max_steps=400, seed=None, render_mode=None):
        super().__init__()
        self.grid_w, self.grid_h = grid_size
        self.n_waypoints = n_waypoints
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # RNG init
        self.rng = np.random.default_rng(seed)

        # Action Space: 9 Discrete Actions
        # 0-3: Move (N, S, E, W)
        # 4: Ascend
        # 5: Descend
        # 6: Scan
        # 7: Hover
        # 8: Return to Base (Intent)
        self.action_space = spaces.Discrete(9)

        # Observation Space: Concatenated Features
        # [x, y, alt, battery, wind_x, wind_y, waypoints_status(N), prox_no_fly, detected_count, return_flag]
        obs_dim = 9 + n_waypoints
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self.alt_levels = [0, 1, 2]
        self.scan_radius_by_alt = {0: 0, 1: 1, 2: 2}
        self.renderer = None

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def _place_entities(self):
        """Procedurally generate map elements based on seed."""
        self.base = (0, 0)
        forbidden = {self.base}
        
        # 1. Obstacles
        self.obstacles = set()
        num_obs = int((self.grid_w * self.grid_h) * 0.1)
        while len(self.obstacles) < num_obs:
            cell = (self.rng.integers(0, self.grid_w), self.rng.integers(0, self.grid_h))
            if cell not in forbidden:
                self.obstacles.add(cell)
                forbidden.add(cell)

        # 2. No-Fly Zones
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

        # 4. Hotspots (Subset of waypoints)
        self.hotspots = set()
        for wp in self.waypoints:
            if self.rng.random() < 0.4:
                self.hotspots.add(wp)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
            
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

    def _get_obs(self):
        norm_x = self.x / (self.grid_w - 1)
        norm_y = self.y / (self.grid_h - 1)
        norm_alt = self.alt / 2.0
        
        if not self.no_fly:
            prox_nf = 0.0
        else:
            dists = [abs(nf[0]-self.x) + abs(nf[1]-self.y) for nf in self.no_fly]
            prox_nf = 1.0 - (min(dists) / (self.grid_w + self.grid_h))
            
        obs = [norm_x, norm_y, norm_alt, self.battery, self.wind[0], self.wind[1]]
        obs.extend(self.remaining)
        obs.extend([prox_nf, self.detected_hotspots / 5.0, float(self.return_intent)])
        
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        self.steps += 1
        reward = -0.01 # Efficiency penalty
        terminated = False
        truncated = False
        info = {}

        # Costs
        COST_HOVER = 0.005
        COST_MOVE = 0.01
        COST_ALT = 0.02
        COST_SCAN = 0.01

        # Movement Delta
        dx, dy = 0, 0
        
        # --- Action Logic ---
        if action == 0: dy = -1 # N
        elif action == 1: dy = 1  # S
        elif action == 2: dx = 1  # E
        elif action == 3: dx = -1 # W
        
        elif action == 4: # Ascend
            if self.alt < 2:
                self.alt += 1
                reward -= 0.05
                self.battery -= COST_ALT
        elif action == 5: # Descend
            if self.alt > 0:
                self.alt -= 1
                self.battery -= COST_HOVER
        elif action == 6: # Scan
            r_scan, scanned = self._perform_scan()
            reward += r_scan
            if scanned: self.battery -= COST_SCAN
        elif action == 7: # Hover
            self.battery -= COST_HOVER
        elif action == 8: # Return
            self.return_intent = 1
            if self.remaining.sum() == 0:
                reward += 0.2 # Good boy reward for returning after work
            self.battery -= COST_HOVER

        # --- Physics & Movement ---
        if action <= 3:
            # Wind Drift (Probabilistic)
            drift_x = 0
            drift_y = 0
            if abs(self.wind[0]) > 0.5 and self.rng.random() < 0.3:
                drift_x = int(np.sign(self.wind[0]))
            if abs(self.wind[1]) > 0.5 and self.rng.random() < 0.3:
                drift_y = int(np.sign(self.wind[1]))

            nx = self.x + dx + drift_x
            ny = self.y + dy + drift_y
            nx = np.clip(nx, 0, self.grid_w - 1)
            ny = np.clip(ny, 0, self.grid_h - 1)
            
            self.x, self.y = nx, ny
            self.battery -= (COST_MOVE + (0.005 * self.alt))

        # --- Constraints & Terminals ---
        
        # Crash into obstacle (only at ground level)
        if (self.x, self.y) in self.obstacles and self.alt == 0:
            reward -= 2.0
            terminated = True
            info["outcome"] = "crashed_obstacle"

        # Enter No-Fly Zone
        if (self.x, self.y) in self.no_fly:
            reward -= 0.5 # Strong penalty

        # Mission Success (All scanned + at base + landed)
        if self.remaining.sum() == 0 and (self.x, self.y) == self.base and self.alt == 0:
            reward += 5.0
            terminated = True
            info["outcome"] = "mission_success"

        # Battery Death
        if self.battery <= 0:
            reward -= 1.0
            terminated = True
            info["outcome"] = "battery_depleted"
            
        if self.steps >= self.max_steps:
            truncated = True

        # Update Wind
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
                    reward += 1.0
                    scanned_something = True
                    if wp in self.hotspots:
                        reward += 0.5
                        self.detected_hotspots += 1
        
        if not scanned_something:
            reward -= 0.1 # Wasted energy
            
        return reward, scanned_something

    def render(self):
        if self.render_mode == "human":
            if self.renderer is None:
                self.renderer = DroneRenderer(self.grid_w, self.grid_h)
            self.renderer.draw(self)

    def close(self):
        if self.renderer:
            self.renderer.close()