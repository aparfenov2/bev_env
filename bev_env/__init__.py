from gym.envs.registration import register
from space_wrappers import DiscretizedActionWrapper, RescaledActionWrapper
from .bev_env import BEVEnv

register(id='BEVEnv-v1',entry_point='bev_env.bev_env:BEVEnv',)

def _dicrete_ctor(*args, **kwargs):
    env = BEVEnv(*args, **kwargs)
    env = DiscretizedActionWrapper(env, 3)
    return env

register(id='BEVEnv-discrete-v1',entry_point='bev_env:_dicrete_ctor',)
