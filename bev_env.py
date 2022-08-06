
from os import path
from typing import Optional
import math
import numpy as np
import pygame as pg
# from pygame import gfxdraw

import gym
from gym import spaces
from enum import Enum
import cv2

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
        self.screen_dim = (1024, 768)
        self.clock = None
        self.img_0_small = None
        self.step_count = 0
        self.max_episode_steps = 0

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        self.obs_img_shape = (160, 120, 3)
        self.observation_space = spaces.Box(0., 255., (self.obs_img_shape[1], self.obs_img_shape[0], 3), dtype=np.uint8)
        self.pos = [100.0, 100.0]
        self.direction = 0.0
        self.step_size = 10.0
        self.turn_size_rad = np.pi/180.0

        self.img_0 = cv2.imread("montage_0.jpg")
        self.img_0_small = cv2.resize(self.img_0, self.screen_dim)
        self.img_0_small = self.cvimage_to_pygame(self.img_0_small)
        self.img_5 = cv2.imread("montage_5.jpg")
        self.img_5 = self.cvimage_to_pygame(self.img_5)

    def set_step_size(self, ss):
        self.step_size = np.clip(ss, 1.0, 100)
        self.turn_size_rad = np.clip(self.step_size/2, 1.0, 30.0) # degrees
        self.turn_size_rad = np.radians(self.turn_size_rad)

    def step(self, u):
        costs = 0
        self.step_count += 1
        min_pos = self.observation_space.shape[1]/2, self.observation_space.shape[0]/2
        max_pos = self.img_0.shape[1] - self.observation_space.shape[1]/2, self.img_0.shape[0] - self.observation_space.shape[0]/2

        ppos = list(self.pos)
        pdir = self.direction

        if u == self.actions.move_back:
            self.pos[0] = np.clip(self.pos[0] - self.step_size * np.cos(self.direction), min_pos[0], max_pos[0])
            self.pos[1] = np.clip(self.pos[1] - self.step_size * np.sin(self.direction), min_pos[1], max_pos[1])
            if not self.obs_in_rect():
                print("refused to go outside image bounds")
                self.pos = ppos

        elif u == self.actions.move_forward:
            self.pos[0] = np.clip(self.pos[0] + self.step_size * np.cos(self.direction), min_pos[0], max_pos[0])
            self.pos[1] = np.clip(self.pos[1] + self.step_size * np.sin(self.direction), min_pos[1], max_pos[1])
            if not self.obs_in_rect():
                print("refused to go outside image bounds")
                self.pos = ppos

        elif u == self.actions.turn_left:
            self.direction -= self.turn_size_rad
            if not self.obs_in_rect():
                print("cant turn outside image bounds")
                self.direction = pdir

        elif u == self.actions.turn_right:
            self.direction += self.turn_size_rad
            if not self.obs_in_rect():
                print("cant turn outside image bounds")
                self.direction = pdir

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

    def obs_in_rect(self):
        num_rows = self.img_0.shape[0]
        num_cols = self.img_0.shape[1]
        width, height = self.observation_space.shape[:2][::-1]
        rect = (self.pos, (width, height), np.degrees(self.direction + np.pi/2))
        return inside_rect(rect = rect, num_cols = num_cols, num_rows = num_rows)


    def _get_obs(self):
        # return np.array([0,0], dtype=np.float32)
        # return self.observation_space.sample()
        width, height = self.observation_space.shape[:2][::-1]
        rect = (self.pos, (width, height), np.degrees(self.direction + np.pi/2))
        crop = crop_rotated_rectangle(image = self.img_0, rect = rect)

        # h, w = self.observation_space.shape[:2]
        # x1, y1 = int(self.pos[0] - w/2), int(self.pos[0] - h/2)
        # x2, y2 = x1 + w, y1 + h

        # crop = self.img_0[y1:y2, x1:x2]
        return crop

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

    @staticmethod
    def cvimage_to_pygame(image):
        """Convert cvimage into a pygame image"""
        return pg.image.frombuffer(image.tostring(), image.shape[1::-1], "BGR")

    def render(self, mode="human"):
        if self.screen is None:
            pg.init()
            pg.display.init()
            self.screen = pg.display.set_mode(self.screen_dim)

        if self.clock is None:
            self.clock = pg.time.Clock()

        self.surf = pg.Surface(self.screen_dim)
        # self.surf.fill((255, 255, 255))

        self.surf.blit(self.img_0_small,(0,0))
        self.arrow(
            screen=self.surf,
            lcolor=(255, 255, 255),
            tricolor=(255, 255, 255),
            start= (self.pos[0] * self.screen_dim[0]/ self.img_0.shape[1], self.pos[1] * self.screen_dim[1]/ self.img_0.shape[0]),
            alen= np.clip(self.step_size, 50, 200),
            rotation=self.direction,
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


def inside_rect(rect, num_cols, num_rows):
    # Determine if the four corners of the rectangle are inside the rectangle with width and height
    # rect tuple
    # center (x,y), (width, height), angle of rotation (to the row)
    # center  The rectangle mass center.
    # center tuple (x, y): x is regarding to the width (number of columns) of the image, y is regarding to the height (number of rows) of the image.
    # size    Width and height of the rectangle.
    # angle   The rotation angle in a clockwise direction. When the angle is 0, 90, 180, 270 etc., the rectangle becomes an up-right rectangle.
    # Return:
    # True: if the rotated sub rectangle is side the up-right rectange
    # False: else

    rect_center = rect[0]
    rect_center_x = rect_center[0]
    rect_center_y = rect_center[1]

    rect_width, rect_height = rect[1]

    rect_angle = rect[2]

    if (rect_center_x < 0) or (rect_center_x > num_cols):
        return False
    if (rect_center_y < 0) or (rect_center_y > num_rows):
        return False

    # https://docs.opencv.org/3.0-beta/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html
    box = cv2.boxPoints(rect)

    x_max = int(np.max(box[:,0]))
    x_min = int(np.min(box[:,0]))
    y_max = int(np.max(box[:,1]))
    y_min = int(np.min(box[:,1]))

    if (x_max <= num_cols) and (x_min >= 0) and (y_max <= num_rows) and (y_min >= 0):
        return True
    else:
        return False


def rect_bbx(rect):
    # Rectangle bounding box for rotated rectangle
    # Example:
    # rotated rectangle: height 4, width 4, center (10, 10), angle 45 degree
    # bounding box for this rotated rectangle, height 4*sqrt(2), width 4*sqrt(2), center (10, 10), angle 0 degree

    box = cv2.boxPoints(rect)

    x_max = int(np.max(box[:,0]))
    x_min = int(np.min(box[:,0]))
    y_max = int(np.max(box[:,1]))
    y_min = int(np.min(box[:,1]))

    # Top-left
    # (x_min, y_min)
    # Top-right
    # (x_min, y_max)
    # Bottom-left
    #  (x_max, y_min)
    # Bottom-right
    # (x_max, y_max)
    # Width
    # y_max - y_min
    # Height
    # x_max - x_min
    # Center
    # (x_min + x_max) // 2, (y_min + y_max) // 2

    center = (int((x_min + x_max) // 2), int((y_min + y_max) // 2))
    width = int(x_max - x_min)
    height = int(y_max - y_min)
    angle = 0

    return (center, (width, height), angle)


def image_rotate_without_crop(mat, angle):
    # https://stackoverflow.com/questions/22041699/rotate-an-image-without-cropping-in-opencv-in-c
    # angle in degrees

    height, width = mat.shape[:2]
    image_center = (width/2, height/2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))

    return rotated_mat

def crop_rectangle(image, rect):
    # rect has to be upright

    num_rows = image.shape[0]
    num_cols = image.shape[1]

    if not inside_rect(rect = rect, num_cols = num_cols, num_rows = num_rows):
        print("Proposed rectangle is not fully in the image.")
        return None

    rect_center = rect[0]
    rect_center_x = rect_center[0]
    rect_center_y = rect_center[1]
    rect_width = rect[1][0]
    rect_height = rect[1][1]


    return image[rect_center_y-rect_height//2:rect_center_y+rect_height-rect_height//2, rect_center_x-rect_width//2:rect_center_x+rect_width-rect_width//2]

def crop_rotated_rectangle(image, rect):
    # Crop a rotated rectangle from a image

    num_rows = image.shape[0]
    num_cols = image.shape[1]

    if not inside_rect(rect = rect, num_cols = num_cols, num_rows = num_rows):
        print("Proposed rectangle is not fully in the image.")
        return None

    rotated_angle = rect[2]

    rect_bbx_upright = rect_bbx(rect = rect)
    rect_bbx_upright_image = crop_rectangle(image = image, rect = rect_bbx_upright)

    rotated_rect_bbx_upright_image = image_rotate_without_crop(mat = rect_bbx_upright_image, angle = rotated_angle)

    rect_width = rect[1][0]
    rect_height = rect[1][1]

    crop_center = (rotated_rect_bbx_upright_image.shape[1]//2, rotated_rect_bbx_upright_image.shape[0]//2)

    return rotated_rect_bbx_upright_image[crop_center[1]-rect_height//2 : crop_center[1]+(rect_height-rect_height//2), crop_center[0]-rect_width//2 : crop_center[0]+(rect_width-rect_width//2)]

