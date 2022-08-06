
from os import path
from typing import Optional
import math
import numpy as np
import pygame as pg
# from pygame import gfxdraw

import gym
from gym import spaces
from enum import Enum

# Bird Eye View Env
class BEVEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    class actions(Enum):
        move_forward = 0
        move_back = 1
        turn_left = 2
        turn_right = 3

    def __init__(self, g=10.0):
        self.screen = None
        self.screen_dim = 1024
        self.clock = None
        self.img = None
        self.step_count = 0
        self.max_episode_steps = 0

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        self.obs_img_shape = (160, 120, 3)
        self.observation_space = spaces.Box(0., 255., (self.obs_img_shape[1], self.obs_img_shape[0], 3))
        self.pos = (100, 100)

    def step(self, u):
        costs = 0
        self.step_count += 1
        return self._get_obs(), -costs, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ):
        super().reset(seed=seed)

        self.step_count = 0

        if not return_info:
            return self._get_obs()
        else:
            return self._get_obs(), {}

    def _get_obs(self):
        # return np.array([0,0], dtype=np.float32)
        return self.observation_space.sample()

    def arrow(self, screen, lcolor, tricolor, start, alen, rotation, trirad, thickness=2):
        end = (
            start[0] + alen * math.cos(rotation),
            start[1] + alen * math.sin(rotation)
        )
        pg.draw.line(screen, lcolor, start, end, thickness)
        rotation = (math.atan2(start[1] - end[1], end[0] - start[0])) + math.pi/2
        rad = np.pi/180
        pg.draw.polygon(screen, tricolor, ((end[0] + trirad * math.sin(rotation),
                                            end[1] + trirad * math.cos(rotation)),
                                        (end[0] + trirad * math.sin(rotation - 120*rad),
                                            end[1] + trirad * math.cos(rotation - 120*rad)),
                                        (end[0] + trirad * math.sin(rotation + 120*rad),
                                            end[1] + trirad * math.cos(rotation + 120*rad))))

    def render(self, mode="human"):
        if self.screen is None:
            pg.init()
            pg.display.init()
            self.screen = pg.display.set_mode((self.screen_dim, self.screen_dim))

        if self.clock is None:
            self.clock = pg.time.Clock()

        self.surf = pg.Surface((self.screen_dim, self.screen_dim))
        # self.surf.fill((255, 255, 255))
        if self.img is None:
            fname = "img_BEV_0_1658066355209776600.png"
            self.img = pg.image.load(fname)

        self.surf.blit(self.img,(0,0))
        self.arrow(
            screen=self.surf,
            lcolor=(255, 255, 255),
            tricolor=(255, 255, 255),
            start=self.pos,
            alen=100,
            rotation=np.pi/8,
            trirad=20
            )

        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            pg.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pg.display.flip()

        if mode == "rgb_array":
            return np.transpose(
                np.array(pg.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        return self.screen is not None

    def close(self):
        if self.screen is not None:
            pg.display.quit()
            pg.quit()
            self.screen = None


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi
