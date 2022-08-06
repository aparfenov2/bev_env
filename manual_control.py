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

parser = argparse.ArgumentParser()
# parser.add_argument('--env-name', default='MiniWorld-Hallway-v0')
parser.add_argument('--env-name', default='BEVEnv-v1')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--no-time-limit', action='store_true', help='ignore time step limits')
parser.add_argument('--top_view', action='store_true', help='show the top view instead of the agent view')
args = parser.parse_args()

env = gym.make(args.env_name)

if args.no_time_limit:
    env.max_episode_steps = math.inf
if args.domain_rand:
    env.domain_rand = True

# view_mode = 'top' if args.top_view else 'agent'
view_mode = 'human'

env.reset()

# Create the display window
env.render(mode=view_mode)

def step(action):
    print('step {}/{}: {}'.format(env.step_count+1, env.max_episode_steps, env.actions(action).name))

    obs, reward, done, info = env.step(action)
    cv2.imshow("winname", obs)
    cv2.waitKey(1)


    if reward > 0:
        print('reward={:.2f}'.format(reward))

    if done:
        print('done!')
        env.reset()

    env.render(mode=view_mode)


def on_key_press(symbol):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == pygame.K_BACKSPACE or symbol == pygame.K_SLASH:
        print('RESET')
        env.reset()
        env.render( mode=view_mode)
        return True

    if symbol == pygame.K_ESCAPE:
        return False

    if symbol == pygame.K_UP:
        step(env.actions.move_forward)
    elif symbol == pygame.K_DOWN:
        step(env.actions.move_back)

    elif symbol == pygame.K_LEFT:
        step(env.actions.turn_left)
    elif symbol == pygame.K_RIGHT:
        step(env.actions.turn_right)
    return True

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
            ret = on_key_press(event.key)
            if not ret:
                exit_game = True
                break
    pygame.time.wait(0)
env.close()
