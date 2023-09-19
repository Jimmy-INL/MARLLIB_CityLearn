import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Dict as GymDict, Box
from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
import time

# importing CityLearn
from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import StableBaselines3Wrapper
from citylearn.wrappers import NormalizedObservationWrapper
from pathlib import Path

from marllib.envs.base_env import ENV_REGISTRY
from add_citylearn_env import RLlibCityLearnGym

if __name__ == '__main__':
    ENV_REGISTRY["CityLearnGym"] = RLlibCityLearnGym
    env = marl.make_env(environment_name="CityLearnGym", map_name="CityLearn")


    # write MASAC
    happo = marl.algos.happo(hyperparam_source="common")
    model = marl.build_model(env, happo, {"core_arch": "mlp", "encode_layer": "128-128"})
    happo.fit(env, model, stop={'timesteps_total': 8759}, local_mode=True, num_gpus=1,
              num_workers=2, share_policy='all', checkpoint_freq=500,
              checkpoint_end=True)
    
    # after fit the model - 