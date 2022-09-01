from typing import Optional
import math
import numpy as np
import pygame as pg

import gym
from gym import spaces
import cv2
import time
from PIL import ImageColor
import pydantic
import typing
import logging

class BEVConfig(pydantic.BaseModel, extra=pydantic.Extra.forbid):
    bg_rgb_img_path: str = pydantic.Field(
        description="path to RGB mosaic image",
        default="montage_0.jpg"
    )
    bg_segm_img_path: str = pydantic.Field(
        description="path to SEGM mosaic image",
        default="montage_5.jpg"
    )
    grass_colors: typing.List[str] = pydantic.Field(
        description="Allowable Area Colors",
        default=["#B4469C", "#BC469C"]
    )
    obs_img_shape: typing.Tuple[int, int] = pydantic.Field(
        description="Observation Image Shape, (cols, rows)",
        default=(160, 120)
    )
    mosaic_size_m: typing.Tuple[float, float] = pydantic.Field(
        description="mosaic size (width, height), m",
        default=(5*8, 40)
    )
    initial_pos: typing.Tuple[float, float] = pydantic.Field(
        description="Initial position, (x,y), m",
        default=(1.0, 1.0)
    )
    initial_rot: float = pydantic.Field(
        description="Initial rotation, clockwise positive, rad",
        default=np.pi/4
    )
    twist_only: bool = pydantic.Field(
        default=True,
        description="use 1d continious action space for turns, no velocity control"
    )
    default_speed_1d: float = pydantic.Field(
        default=1.0,
        description="cart speed in 1d mode, m/s"
    )
    render_in_step: bool = pydantic.Field(
        description="call render() inside step(). Useful while training.",
        default=False
    )
    segm_in_obs: bool = pydantic.Field(
        description="return segmentation image in observation.",
        default=True
    )
    random_pos: bool = pydantic.Field(
        description="random initial position & orientation.",
        default=False
    )
    init_logging: bool = pydantic.Field(
        description="initialize logging.",
        default=False
    )
    obstacle_done: bool = pydantic.Field(
        description="return Done on collision.",
        default=True
    )
    timeout_done: bool = pydantic.Field(
        description="return Done on timeout/max steps reached.",
        default=True
    )
    obstacle_cost: float = pydantic.Field(
        default=500.0,
        description="cost of colliding with obstacle, reward = -cost"
    )
    render_fps: int = pydantic.Field(
        description="rendering fps",
        default=30
    )
    const_dt: float = pydantic.Field(
        description="use constant dt in step()",
        default=None
    )
    max_episode_steps: int = pydantic.Field(
        description="max episode steps",
        default=0
    )

class BEVEnv(gym.Env):
    "Bird Eye View Env with geometry_msgs/Twist as action"

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    MAX_SPEED = 4.0

    def __init__(self, **kwargs):
        self.config = BEVConfig(**kwargs)
        if self.config.init_logging:
            logging.basicConfig(level=logging.DEBUG)
        logging.info(self.config)

        self.grass_colors = [ImageColor.getcolor(grass_color, "RGB")[::-1] for grass_color in self.config.grass_colors]
        self.screen = None
        self.screen_dim = (1024, 768) # pygame screen size
        self.clock = None
        self.img_0_small = None

        # action space: geometry_msgs/Twist
        # linear.x - forward velocity, m/s
        # angular.z - yaw speed, rad/s
        if self.config.twist_only:
            self.action_space = spaces.Box(low=-np.pi/2, high=np.pi/2, shape=(1,), dtype=np.float32)
        else:
            shape_12 = (2,)
            self.action_space = spaces.Box(
                low=np.array([-self.MAX_SPEED, -np.pi/2]).reshape(shape_12),
                high=np.array([self.MAX_SPEED, np.pi/2]).reshape(shape_12),
                shape=shape_12
            )

        # observation space: RGB BEV segmentation image
        obs_img_shape = self.config.obs_img_shape + (3,)
        self.observation_space = spaces.Box(0, 255, (obs_img_shape[1], obs_img_shape[0], 3), dtype=np.uint8)

        self.img_0 = cv2.imread(self.config.bg_rgb_img_path)
        self.img_5 = cv2.imread(self.config.bg_segm_img_path)

        self.last_time_sec = time.time()
        self.m_to_pix = (
            self.img_0.shape[0] / self.config.mosaic_size_m[0],
            self.img_0.shape[1] / self.config.mosaic_size_m[1]
        ) # 5m/tile, 8x8 tiles
        self._reset()

    def _reset(self):
        self.step_count = 0
        self.total_reward = 0
        self.last_time_sec = time.time()
        self.current_position_m = list(self.config.initial_pos)
        self.abs_direction_rad = self.config.initial_rot  # clockwise positive
        if self.config.random_pos:
            tryouts = 100
            while tryouts > 0:
                self.current_position_m = np.random.uniform(low=1.0, high=39.0, size=(2,)).tolist()
                self.abs_direction_rad = np.random.uniform(low=-np.pi/2, high=np.pi/2, size=(1,)).item()
                if self.staying_on_the_grass() and not self.only_grass_in_obs():
                    break
            if not tryouts:
                raise Exception("cannot randomly place cart in allowed area")

        if self.config.twist_only:
            self.last_vel = (self.config.default_speed_1d, 0)
        else:
            self.last_vel = (0, 0)
        logging.info(f"initial location {self.current_position_m} direction {self.abs_direction_rad} speed {self.last_vel}")

    def position_in_pixels(self):
        return (
            self.current_position_m[0] * self.m_to_pix[0],
            self.current_position_m[1] * self.m_to_pix[1]
        )

    def staying_on_the_grass(self):
        pos_pix = tuple(map(int, self.position_in_pixels()))[::-1]
        color_at_pos = tuple(map(int, self.img_5[pos_pix]))
        return color_at_pos in self.grass_colors

    def only_grass_in_obs(self, obs=None):
        if obs is None:
            obs = self._get_obs()
        colors, counts = np.unique(obs.reshape(-1, obs.shape[-1]), axis=0, return_counts=True)
        # remove colors with counts < 1% of obs area in pix
        thd = 0.1 * np.prod(obs.shape[:2])
        colors = [tuple(col) for col, cnt in zip(colors, counts) if cnt > thd]
        return set(colors) - set(self.grass_colors) == set()

    def step(self, action):
        action = np.array(action).reshape(self.action_space.shape).astype(self.action_space.dtype)
        assert self.action_space.contains(action), f"Invalid Action {action}, space={self.action_space}"
        if len(action.shape) == 1 and action.shape[0] == 1:
            action = action.item()
        if self.config.twist_only:
            self.last_vel = (self.last_vel[0], action)
        else:
            self.last_vel = action

        velocity_ms, yaw_rads = self.last_vel
        self.step_count += 1
        time_sec = time.time()
        time_diff = (time_sec - self.last_time_sec)
        if self.config.const_dt is not None:
            time_diff = self.config.const_dt
        pos_increment_m = velocity_ms * time_diff
        yaw_increment_rad = yaw_rads  * time_diff
        self.last_time_sec = time_sec

        ppos = list(self.current_position_m)
        pyaw = self.abs_direction_rad

        self.abs_direction_rad += yaw_increment_rad
        self.current_position_m[0] += pos_increment_m * np.cos(self.abs_direction_rad)
        self.current_position_m[1] += pos_increment_m * np.sin(self.abs_direction_rad)

        # wall bouncing logic
        collided_with_boundary = False
        if not self.obs_in_rect():
            logging.info(f"at step {self.step_count} refused to go outside image bounds at location {self.current_position_m} direction {self.abs_direction_rad}")
            self.current_position_m = list(ppos)
            self.abs_direction_rad = np.pi - pyaw

            self.current_position_m[0] += pos_increment_m * np.cos(self.abs_direction_rad)
            self.current_position_m[1] += pos_increment_m * np.sin(self.abs_direction_rad)
            if not self.obs_in_rect():
                self.abs_direction_rad = -pyaw

            self.current_position_m = ppos
            collided_with_boundary = True

        # cost calculation
        hit_obstacle = not self.staying_on_the_grass()
        if hit_obstacle:
            logging.warning(f"at step {self.step_count} hit an obstacle at location {self.current_position_m} direction {self.abs_direction_rad}")
        reward = 1.0 * pos_increment_m
        if hit_obstacle:
            reward -= self.config.obstacle_cost
        obs = self._get_obs()
        only_grass_around = self.only_grass_in_obs(obs)
        if only_grass_around:
            reward += np.pi/8 - abs(yaw_increment_rad)
        self.total_reward += reward
        info = {
            "step_count": self.step_count,
            "max_episode_steps": self.config.max_episode_steps,
            "last_action": action,
            "pos_increment_m": pos_increment_m,
            "yaw_increment_rad": yaw_increment_rad,
            "current_position_m": self.current_position_m,
            "current_position_pix": self.position_in_pixels(),
            "abs_direction_rad": self.abs_direction_rad,
            "reward": reward,
            "total_reward": self.total_reward,
            "collided_with_boundary": collided_with_boundary,
            "only_grass_around": only_grass_around,
        }
        if self.step_count % 100 == 0:
            logging.info(info)

        if self.config.render_in_step:
            self.render()
        done = False
        info["TimeLimit.truncated"] = False
        if self.config.max_episode_steps > 0:
            if self.step_count >= self.config.max_episode_steps:
                logging.info(f"max episodes at step {self.config.max_episode_steps} reached")
                if self.config.timeout_done:
                    done = True
                else:
                    info["TimeLimit.truncated"] = True
        if self.config.obstacle_done and hit_obstacle:
            done = True
        if done:
            info["TimeLimit.truncated"] = False
            logging.info(f"rollout done at step {self.step_count}, total_reward={self.total_reward}")
        return obs, reward, done, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ):
        # super().reset(seed=seed)

        self._reset()

        if not return_info:
            return self._get_obs()
        else:
            return self._get_obs(), {}

    def obs_in_rect(self):
        num_rows = self.img_0.shape[0]
        num_cols = self.img_0.shape[1]
        width, height = self.observation_space.shape[:2][::-1]
        rect = (self.position_in_pixels(), (width, height), np.degrees(self.abs_direction_rad + np.pi/2))
        return inside_rect(rect = rect, num_cols = num_cols, num_rows = num_rows)


    def _get_obs(self):
        # return np.array([0,0], dtype=np.float32)
        # return self.observation_space.sample()
        height, width = self.observation_space.shape[:2]
        rect = (self.position_in_pixels(), (width, height), np.degrees(self.abs_direction_rad + np.pi/2))
        img = self.img_0
        if self.config.segm_in_obs:
            img = self.img_5
        crop = crop_rotated_rectangle(image = img, rect = rect)

        # h, w = self.observation_space.shape[:2]
        # x1, y1 = int(self.current_position_m[0] - w/2), int(self.current_position_m[0] - h/2)
        # x2, y2 = x1 + w, y1 + h

        # crop = self.img_0[y1:y2, x1:x2]
        return crop

    def arrow(self, screen, lcolor, tricolor, start, alen, rotation, trirad, thickness=2):
        end = (
            start[0] + alen * math.cos(rotation),
            start[1] + alen * math.sin(rotation)
        )
        start = (
            start[0] * self.screen_dim[0]/ self.img_0.shape[1],
            start[1] * self.screen_dim[1]/ self.img_0.shape[0]
        )
        end = (
            end[0] * self.screen_dim[0]/ self.img_0.shape[1],
            end[1] * self.screen_dim[1]/ self.img_0.shape[0]
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
            self.img_0_small = cv2.resize(self.img_0, self.screen_dim)
            self.img_0_small = self.cvimage_to_pygame(self.img_0_small)

        if self.clock is None:
            self.clock = pg.time.Clock()


        self.surf = pg.Surface(self.screen_dim)
        # self.surf.fill((255, 255, 255))

        self.surf.blit(self.img_0_small,(0,0))
        self.arrow(
            screen=self.surf,
            lcolor=(255, 255, 255),
            tricolor=(255, 255, 255),
            start= self.position_in_pixels(),
            alen= np.clip(self.last_vel[0] * 400, 200, 800),
            rotation=self.abs_direction_rad,
            trirad=20
            )

        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            pg.event.pump()
            self.clock.tick(self.config.render_fps)
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
        logging.info("Proposed rectangle is not fully in the image.")
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
        logging.info("Proposed rectangle is not fully in the image.")
        return None

    rotated_angle = rect[2]

    rect_bbx_upright = rect_bbx(rect = rect)
    rect_bbx_upright_image = crop_rectangle(image = image, rect = rect_bbx_upright)

    rotated_rect_bbx_upright_image = image_rotate_without_crop(mat = rect_bbx_upright_image, angle = rotated_angle)

    rect_width = rect[1][0]
    rect_height = rect[1][1]

    crop_center = (rotated_rect_bbx_upright_image.shape[1]//2, rotated_rect_bbx_upright_image.shape[0]//2)

    return rotated_rect_bbx_upright_image[crop_center[1]-rect_height//2 : crop_center[1]+(rect_height-rect_height//2), crop_center[0]-rect_width//2 : crop_center[0]+(rect_width-rect_width//2)]

