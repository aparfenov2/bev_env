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

from gym.envs.registration import register
# from bev_env import BEVEnv
register(id='BEVEnv-v1',entry_point='bev_env:BEVEnv',)

def make_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--env-name', default='MiniWorld-Hallway-v0')
    parser.add_argument('--env-name', default='BEVEnv-v1')
    parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
    parser.add_argument('--no-time-limit', action='store_true', help='ignore time step limits')
    parser.add_argument('--top_view', action='store_true', help='show the top view instead of the agent view')
    return parser

VIEW_MODE = 'human'

class Main:
    def __init__(self, args):
        self.args = args
        self.vel_inc = 0.1  # m/s
        self.rad_inc = np.pi/180  # rad/s
        self.cur_vel = [0, 0]

    def main(self):
        self.env = gym.make(self.args.env_name)

        if self.args.no_time_limit:
            self.env.max_episode_steps = math.inf
        if self.args.domain_rand:
            self.env.domain_rand = True

        # VIEW_MODE = 'top' if self.args.top_view else 'agent'

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
                if event.type == pygame.KEYDOWN:
                    ret = self.on_key_press(event.key)
                    if not ret:
                        exit_game = True
                        break
            self.step()
            pygame.time.wait(0)
        self.env.close()

    def step(self):
        obs, reward, done, info = self.env.step(self.cur_vel)
        cv2.imshow("winname", obs)
        cv2.waitKey(1)

        if done:
            print('done!')
            self.env.reset()

        info.update({
            "cur_vel": self.cur_vel,
            "vel_inc": self.vel_inc,
            "rad_inc": self.rad_inc,
        })
        print("----------------------------------------------")
        print(json.dumps(json.loads(json.dumps(info), parse_float=lambda x: round(float(x), 3)), indent=4))
        # print(json.dumps(info, indent=4))

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
            self.cur_vel[0] += self.vel_inc
            self.cur_vel[1] = 0

        elif symbol == pygame.K_DOWN:
            self.cur_vel[0] = max(0, self.cur_vel[0] - self.vel_inc)
            self.cur_vel[1] = 0

        elif symbol == pygame.K_PAGEUP:
            self.vel_inc += 0.1

        elif symbol == pygame.K_PAGEDOWN:
            self.vel_inc = max(0, self.vel_inc - 0.1)

        elif symbol == pygame.K_LEFT:
            self.cur_vel[1] = max(-np.pi/2, self.cur_vel[1] - self.rad_inc)

        elif symbol == pygame.K_RIGHT:
            self.cur_vel[1] = min(np.pi/2, self.cur_vel[1] + self.rad_inc)
        return True

if __name__ == '__main__':
    Main(make_parser().parse_args()).main()