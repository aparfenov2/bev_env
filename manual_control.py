#!/usr/bin/env python3

"""
This script allows you to manually control the simulator
using the keyboard arrows.
"""

import sys
import argparse
# import pyglet
import math
# from pyglet.window import key
# from pyglet import clock
import pygame
import numpy as np
import gym
# import gym_miniworld
import cv2
import json
import time
import bev_env

# from gym.envs.registration import register
# from bev_env import BEVEnv
# register(id='BEVEnv-v1',entry_point='bev_env:BEVEnv',)

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default='BEVEnv-v1')
    return parser

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

VIEW_MODE = 'human'
MIN_VEL  = 0
MAX_VEL  = 4.0
MIN_RVEL = -np.pi/2
MAX_RVEL = np.pi/2

class Main:
    def __init__(self, args):
        self.args = args
        self.vel_inc = 1.0  # m/s
        self.rad_inc = MAX_RVEL  # rad/s
        self.cur_vel = [0, 0]

    def main(self):
        self.env = gym.make(self.args.env_name, segm_in_obs=True, twist_only=False)

        obs = self.env.reset()
        # Create the display window
        self.env.render(mode=VIEW_MODE)

        cv2.imshow("winname", obs)
        cv2.waitKey(1)

        # Enter main event loop
        exit_game = False
        vel = [0,0]
        while not exit_game:
            for event in pygame.event.get():  # For Loop
                if cv2.waitKey(1) & 0xff == 27:
                    exit_game = True
                    break
                if event.type == pygame.QUIT:
                    exit_game = True
                    break
                if event.type == pygame.KEYUP:
                    self.on_key_release(event.key)
                if event.type == pygame.KEYDOWN:
                    ret = self.on_key_press(event.key)
                    if not ret:
                        exit_game = True
                        break
            self.step()
            pygame.time.wait(0)
        self.env.close()

    def step(self):
        action = self.cur_vel
        if len(self.env.action_space.shape) == 1 and self.env.action_space.shape[0] == 1:
            action = self.cur_vel[1]
        obs, reward, done, info = self.env.step(action)
        cv2.imshow("winname", obs)
        cv2.waitKey(1)

        # if done:
        #     print('done!')
        #     self.env.reset()

        info.update({
            "cur_vel": self.cur_vel,
            "vel_inc": self.vel_inc,
            "rad_inc": self.rad_inc,
        })
        print("----------------------------------------------")
        # print(json.dumps(json.loads(json.dumps(info), parse_float=lambda x: round(float(x), 3)), indent=4))
        print(json.dumps(info, indent=4, cls=NumpyEncoder))

        self.env.render(mode=VIEW_MODE)

        if info["collided_with_boundary"]:
            self.cur_vel[1] = 0


    def on_key_press(self, symbol):
        """
        This handler processes keyboard commands that
        control the simulation
        """

        if symbol == pygame.K_BACKSPACE or symbol == pygame.K_SLASH:
            print('RESET')
            self.env.reset()
            self.env.render( mode=VIEW_MODE)
            self.cur_vel = [0,0]
            return True

        if symbol == pygame.K_ESCAPE:
            return False

        if symbol == pygame.K_UP:
            self.cur_vel[0] = min(MAX_VEL, self.cur_vel[0] + self.vel_inc)
            self.cur_vel[1] = 0

        elif symbol == pygame.K_DOWN:
            self.cur_vel[0] = max(MIN_VEL, self.cur_vel[0] - self.vel_inc)
            self.cur_vel[1] = 0

        elif symbol == pygame.K_PAGEUP:
            self.vel_inc += 0.1

        elif symbol == pygame.K_PAGEDOWN:
            self.vel_inc = max(0, self.vel_inc - 0.1)

        elif symbol == pygame.K_LEFT:
            self.cur_vel[1] = max(MIN_RVEL, self.cur_vel[1] - self.rad_inc)

        elif symbol == pygame.K_RIGHT:
            self.cur_vel[1] = min(MAX_RVEL, self.cur_vel[1] + self.rad_inc)
        return True

    def on_key_release(self, symbol):
        if symbol == pygame.K_LEFT:
            self.cur_vel[1] = 0
        if symbol == pygame.K_RIGHT:
            self.cur_vel[1] = 0

if __name__ == '__main__':
    Main(make_parser().parse_args()).main()