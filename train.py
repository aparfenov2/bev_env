import gym
from gym.envs.registration import register
register(id='BEVEnv-v1',entry_point='bev_env:BEVEnv',)

# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines import PPO2
from stable_baselines3 import PPO

env = gym.make('BEVEnv-v1',
    twist_only=True,
    # render_in_step=True,
    const_dt=0.1,
    random_pos=True,
    obstacle_done=True
    )
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

# model = PPO2(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=10000)

model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)
print("Learning done.")
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
