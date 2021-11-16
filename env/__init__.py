import pybullet_envs
import sys

from .ant import AntEnv
from .humanoid import HumanoidEnv
from .gym_locomotion_envs import HopperBulletEnv
from .gym_locomotion_envs import Walker2DBulletEnv
from .gym_locomotion_envs import HalfCheetahBulletEnv

env_overwrite = {'Ant': AntEnv,'Humanoid':HumanoidEnv,'HopperBulletEnv':HopperBulletEnv,
                 'Walker2DBulletEnv':Walker2DBulletEnv,'HalfCheetahBulletEnv':HalfCheetahBulletEnv}

sys.modules[__name__] = env_overwrite