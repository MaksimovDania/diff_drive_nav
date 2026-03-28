"""Differential Drive Navigation Environment.

A 2D navigation task where a differential-drive robot must reach a goal
while avoiding a circular obstacle at the origin.
"""

import math
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DiffDriveNavEnv(gym.Env):
    """Gymnasium environment for differential-drive obstacle avoidance navigation.

    The robot spawns in the bottom-left region and must reach a goal in the
    top-right region while avoiding a circle of radius 2 centered at the origin.

    Observation (8-dim, pre-normalized):
        0: dx to goal / 12
        1: dy to goal / 20
        2: dist to goal / 25
        3: cos(theta)
        4: sin(theta)
        5: cos(angle_to_goal)
        6: sin(angle_to_goal)
        7: dist to obstacle surface / 15

    Action (2-dim, continuous):
        0: v_left  in [-1, 1]
        1: v_right in [-1, 1]
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # --- constants ---
    DT = 0.1               # simulation timestep
    WHEELBASE = 0.5         # distance between wheels
    OBSTACLE_RADIUS = 2.0
    AGENT_RADIUS = 0.2
    GOAL_THRESHOLD = 0.3
    MAX_STEPS = 800

    # reward weights
    PROGRESS_SCALE = 5.0
    HEADING_SCALE = 0.1
    STEP_PENALTY = -0.01
    PROXIMITY_THRESHOLD = 1.0
    PROXIMITY_SCALE = -1.0
    GOAL_REWARD = 20.0
    COLLISION_PENALTY = -10.0

    # normalization constants (approximate world extents)
    NORM_DX = 12.0
    NORM_DY = 20.0
    NORM_DIST = 25.0
    NORM_OBS_DIST = 15.0

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.step_count = 0
        self._prev_dist = 0.0
        self._trajectory: list[tuple[float, float]] = []

        # pygame (lazy init)
        self._screen = None
        self._clock = None
        self._pygame_inited = False

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        rng = self.np_random

        # spawn agent in bottom-left region, ensuring safe distance from obstacle
        min_safe_dist = self.OBSTACLE_RADIUS + self.AGENT_RADIUS + 0.5
        while True:
            self.x = rng.uniform(-2.0, 0.0)
            self.y = rng.uniform(-10.0, 0.0)
            if math.sqrt(self.x ** 2 + self.y ** 2) >= min_safe_dist:
                break
        self.theta = rng.uniform(-math.pi, math.pi)

        # spawn goal in top-right region, ensuring it's outside obstacle
        while True:
            self.goal_x = rng.uniform(0.0, 2.0)
            self.goal_y = rng.uniform(0.0, 10.0)
            if math.sqrt(self.goal_x ** 2 + self.goal_y ** 2) >= min_safe_dist:
                break

        self.step_count = 0
        self._prev_dist = self._dist_to_goal()
        self._trajectory = [(self.x, self.y)]

        obs = self._get_obs()
        info = self._get_info(reward=0.0, terminated=False)
        return obs, info

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        v_left, v_right = float(action[0]), float(action[1])

        # differential drive kinematics
        v = (v_right + v_left) / 2.0
        omega = (v_right - v_left) / self.WHEELBASE

        self.x += v * math.cos(self.theta) * self.DT
        self.y += v * math.sin(self.theta) * self.DT
        self.theta += omega * self.DT
        # wrap theta to [-pi, pi]
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

        self.step_count += 1
        self._trajectory.append((self.x, self.y))

        # --- check termination ---
        dist_to_goal = self._dist_to_goal()
        dist_to_obs = self._dist_to_obstacle_surface()
        collision = dist_to_obs <= 0.0
        success = dist_to_goal < self.GOAL_THRESHOLD
        truncated = self.step_count >= self.MAX_STEPS
        terminated = collision or success

        # --- reward ---
        reward = 0.0

        # progress shaping
        reward += (self._prev_dist - dist_to_goal) * self.PROGRESS_SCALE
        self._prev_dist = dist_to_goal

        # heading alignment bonus
        angle_to_goal = math.atan2(self.goal_y - self.y, self.goal_x - self.x)
        heading_error = abs(self._angle_diff(angle_to_goal, self.theta))
        reward += (1.0 - heading_error / math.pi) * self.HEADING_SCALE

        # step penalty
        reward += self.STEP_PENALTY

        # obstacle proximity penalty (scales from 0 at threshold to full at surface)
        if dist_to_obs < self.PROXIMITY_THRESHOLD:
            reward += self.PROXIMITY_SCALE * (1.0 - dist_to_obs / self.PROXIMITY_THRESHOLD)

        # terminal rewards
        if success:
            reward += self.GOAL_REWARD
        if collision:
            reward += self.COLLISION_PENALTY

        obs = self._get_obs()
        info = self._get_info(reward=reward, terminated=terminated, success=success, collision=collision)

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None
        return self._render_pygame()

    def close(self):
        if self._pygame_inited:
            import pygame
            pygame.quit()
            self._pygame_inited = False
            self._screen = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        dx = self.goal_x - self.x
        dy = self.goal_y - self.y
        dist = math.sqrt(dx * dx + dy * dy)
        angle_to_goal = math.atan2(dy, dx)
        dist_to_obs = self._dist_to_obstacle_surface()

        # angle from agent to obstacle center
        angle_to_obs = math.atan2(-self.y, -self.x)
        # relative angle (obstacle direction in agent's frame)
        rel_angle_obs = self._angle_diff(angle_to_obs, self.theta)

        obs = np.array([
            dx / self.NORM_DX,
            dy / self.NORM_DY,
            dist / self.NORM_DIST,
            math.cos(self.theta),
            math.sin(self.theta),
            math.cos(angle_to_goal),
            math.sin(angle_to_goal),
            max(dist_to_obs, 0.0) / 5.0,
            math.cos(rel_angle_obs),    # obstacle direction relative to heading
            math.sin(rel_angle_obs),
            self.x / 5.0,              # agent position (helps learn spatial policy)
            self.y / 12.0,
        ], dtype=np.float32)
        return obs

    def _get_info(self, reward=0.0, terminated=False, success=False, collision=False):
        return {
            "dist_to_goal": self._dist_to_goal(),
            "dist_to_obstacle": self._dist_to_obstacle_surface(),
            "success": success,
            "collision": collision,
            "step": self.step_count,
        }

    def _dist_to_goal(self) -> float:
        return math.sqrt((self.x - self.goal_x) ** 2 + (self.y - self.goal_y) ** 2)

    def _dist_to_obstacle_surface(self) -> float:
        """Distance from agent boundary to obstacle boundary (negative = inside)."""
        dist_to_center = math.sqrt(self.x ** 2 + self.y ** 2)
        return dist_to_center - self.OBSTACLE_RADIUS - self.AGENT_RADIUS

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """Signed difference a - b, wrapped to [-pi, pi]."""
        d = a - b
        return math.atan2(math.sin(d), math.cos(d))

    # ------------------------------------------------------------------
    # Pygame rendering
    # ------------------------------------------------------------------

    _WINDOW_SIZE = 600
    _WORLD_RANGE = 12.0  # world coords go from -12 to +12

    def _render_pygame(self):
        import pygame

        if not self._pygame_inited:
            pygame.init()
            self._pygame_inited = True
            self._clock = pygame.time.Clock()
            if self.render_mode == "human":
                self._screen = pygame.display.set_mode(
                    (self._WINDOW_SIZE, self._WINDOW_SIZE)
                )
                pygame.display.set_caption("Diff Drive Nav")
            else:
                self._screen = pygame.Surface(
                    (self._WINDOW_SIZE, self._WINDOW_SIZE)
                )

        surf = self._screen
        surf.fill((255, 255, 255))

        def w2s(wx, wy):
            """World coords to screen coords."""
            sx = int((wx + self._WORLD_RANGE) / (2 * self._WORLD_RANGE) * self._WINDOW_SIZE)
            sy = int((self._WORLD_RANGE - wy) / (2 * self._WORLD_RANGE) * self._WINDOW_SIZE)
            return sx, sy

        def w2r(wr):
            """World radius to screen radius."""
            return max(1, int(wr / (2 * self._WORLD_RANGE) * self._WINDOW_SIZE))

        # grid lines
        for i in range(-int(self._WORLD_RANGE), int(self._WORLD_RANGE) + 1, 2):
            sx0, sy0 = w2s(i, -self._WORLD_RANGE)
            sx1, sy1 = w2s(i, self._WORLD_RANGE)
            pygame.draw.line(surf, (230, 230, 230), (sx0, sy0), (sx1, sy1), 1)
            sx0, sy0 = w2s(-self._WORLD_RANGE, i)
            sx1, sy1 = w2s(self._WORLD_RANGE, i)
            pygame.draw.line(surf, (230, 230, 230), (sx0, sy0), (sx1, sy1), 1)

        # axes
        ax0, ay0 = w2s(-self._WORLD_RANGE, 0)
        ax1, ay1 = w2s(self._WORLD_RANGE, 0)
        pygame.draw.line(surf, (200, 200, 200), (ax0, ay0), (ax1, ay1), 2)
        ax0, ay0 = w2s(0, -self._WORLD_RANGE)
        ax1, ay1 = w2s(0, self._WORLD_RANGE)
        pygame.draw.line(surf, (200, 200, 200), (ax0, ay0), (ax1, ay1), 2)

        # obstacle
        obs_pos = w2s(0, 0)
        obs_r = w2r(self.OBSTACLE_RADIUS)
        pygame.draw.circle(surf, (180, 180, 180), obs_pos, obs_r)
        pygame.draw.circle(surf, (100, 100, 100), obs_pos, obs_r, 2)

        # trajectory trail
        if len(self._trajectory) > 1:
            n = len(self._trajectory)
            for i in range(1, n):
                alpha = int(80 + 175 * i / n)
                color = (alpha, alpha, 255)
                p0 = w2s(*self._trajectory[i - 1])
                p1 = w2s(*self._trajectory[i])
                pygame.draw.line(surf, color, p0, p1, 2)

        # goal
        gx, gy = w2s(self.goal_x, self.goal_y)
        pygame.draw.circle(surf, (0, 200, 0), (gx, gy), 8)
        pygame.draw.circle(surf, (0, 150, 0), (gx, gy), 8, 2)

        # agent (triangle oriented by theta)
        size = 8
        cx, cy = w2s(self.x, self.y)
        # note: screen y is flipped, so negate sin for y
        cos_t, sin_t = math.cos(self.theta), math.sin(self.theta)
        tip = (cx + int(size * 1.5 * cos_t), cy - int(size * 1.5 * sin_t))
        left = (cx + int(size * (-cos_t + sin_t)), cy - int(size * (-sin_t - cos_t)))
        right = (cx + int(size * (-cos_t - sin_t)), cy - int(size * (-sin_t + cos_t)))
        pygame.draw.polygon(surf, (30, 100, 255), [tip, left, right])
        pygame.draw.polygon(surf, (0, 50, 200), [tip, left, right], 2)

        if self.render_mode == "human":
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
            # process events to keep window responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
            return None
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(surf)), axes=(1, 0, 2)
            )
