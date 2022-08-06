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
            pygame.time.wait(0)
        self.env.close()

    def print_state(self, action=None):
        if action is None:
            action_str = ""
        else:
            action_str = f"action {self.env.actions(action).name}"

        print(f"step {self.env.step_count}/{self.env.max_episode_steps} {action_str} \
step_size {self.env.step_size:3.3f} turn_size_rad {np.degrees(self.env.turn_size_rad):3.3f} \
pos {self.env.pos[0]:3.3f}, {self.env.pos[1]:3.3f} dir {np.degrees(self.env.direction):3.3f}")

    def step(self, action):
        self.print_state(action)
        obs, reward, done, info = self.env.step(action)
        cv2.imshow("winname", obs)
        cv2.waitKey(1)

        if reward > 0:
            print('reward={:.2f}'.format(reward))

        if done:
            print('done!')
            self.env.reset()

        self.env.render(mode=VIEW_MODE)

    def on_key_press(self, symbol):
        """
        This handler processes keyboard commands that
        control the simulation
        """

        if symbol == pygame.K_BACKSPACE or symbol == pygame.K_SLASH:
            print('RESET')
            self.env.reset()
            self.env.render( mode=VIEW_MODE)
            return True

        if symbol == pygame.K_ESCAPE:
            return False

        if symbol == pygame.K_UP:
            self.step(self.env.actions.move_forward)

        elif symbol == pygame.K_DOWN:
            self.step(self.env.actions.move_back)

        elif symbol == pygame.K_PAGEUP:
            self.env.set_step_size(self.env.step_size + 1)
            self.env.render(mode=VIEW_MODE)
            self.print_state()

        elif symbol == pygame.K_PAGEDOWN:
            self.env.set_step_size(self.env.step_size - 1)
            self.env.render(mode=VIEW_MODE)
            self.print_state()

        elif symbol == pygame.K_LEFT:
            self.step(self.env.actions.turn_left)

        elif symbol == pygame.K_RIGHT:
            self.step(self.env.actions.turn_right)
        return True

if __name__ == '__main__':
    Main(make_parser().parse_args()).main()