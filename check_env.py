import gym
import bev_env
from stable_baselines3.common.env_checker import check_env

env = gym.make("BEVEnv-discrete-v1", segm_in_obs=True)
check_env(env)