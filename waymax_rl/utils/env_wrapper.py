import os
import random
import time
import numpy as np
import jax
import jax.numpy as jnp
import gym
from gym import spaces
from gym.vector import utils
from typing import ClassVar, Optional
import dataclasses
from rl_games.common import vecenv, env_configurations
from rl_games.envs.brax import BraxEnv

from brax.envs.wrappers import VectorWrapper

from waymax import config as _config
from waymax import dataloader, dynamics, datatypes
from waymax.env import planning_agent_environment
from waymax.config import LinearCombinationRewardConfig
from waymax.env.wrappers import brax_wrapper


class VecWaymaxEnv(gym.vector.VectorEnv):
    _gym_disable_underscore_compat: ClassVar[bool] = True
    def __init__(self, **cfg):
        self.cfg = cfg
        data_path = cfg['data_cfg']['data_path']  
        if os.path.isdir(data_path):
            data_list = os.listdir(data_path)
            if cfg['data_cfg']['data_type'] == 'pkl':
                data_list = [i for i in data_list if i[-3:] == 'pkl']
            self.data_list = [os.path.join(data_path, i) for i in data_list]
        elif os.path.isfile(data_path):
            self.data_list = [data_path]
        else:
            raise OSError(data_path)
        
        self.data_file_name = random.choice(self.data_list)     

        if cfg['data_cfg']['data_type'] == 'tfrecord':
            self.dataset_config = dataclasses.replace(_config.WOD_1_1_0_TRAINING,
                                                      batch_dims=(cfg['env_nums'],),
                                                      max_num_objects=cfg['max_num_objects'],
                                                      path=self.data_file_name,
                                                    #   include_sdc_paths=True,
                                                      num_paths=1,
                                                      num_points_per_path=200)

            self.dataset_iter = dataloader.simulator_state_generator(self.dataset_config)

        # update env_config
        reward_cfg = cfg['reward_cfg']
        reward_cfg = {k: v for k, v in reward_cfg.items() if int(v) != 0}
        reward_cfg = LinearCombinationRewardConfig(reward_cfg)
        self.env_config = _config.EnvironmentConfig(init_steps=11, max_num_objects=cfg['max_num_objects'], rewards=reward_cfg)

        # select dynamics
        dynamics_model = None  # dynamics.DeltaGlobal()
        act_space_cfg = cfg['action_space']
        if act_space_cfg['steering_acc']:
            # normalize to [-1, 1]
            if not act_space_cfg['is_discrete']:
                dynamics_model = dynamics.InvertibleBicycleModel(normalize_actions=True)
                self.single_action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=float)

        self.single_stateless_env = (
            planning_agent_environment.PlanningAgentEnvironment(
                dynamics_model=dynamics_model, config=self.env_config
            )
        )
        self._env = brax_wrapper.BraxWrapper(self.single_stateless_env)
        self._env = VmapWrapper(self._env, cfg['env_nums'])
        self._env = AutoResetWrapper(self._env)
        # self.env = VectorGymWrapper(self.env, cfg['seed'], backend=cfg['backend'])

        # gym setting
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': 10
        }     
        if not hasattr(self._env, 'batch_size'):
            raise ValueError('underlying env must be batched')          

        self.num_envs = self._env.batch_size
        self.seed(cfg['seed'])
        self.backend = cfg['backend']
        self._state = None

        single_obs = np.inf * np.ones((10, 6), dtype='float32')
        self.single_observation_space = spaces.Box(-single_obs, single_obs, dtype='float32')
        self.observation_space = utils.batch_space(self.single_observation_space, self.num_envs)

        # self.observation_space = spaces.Dict({
        #     'ego_obs': spaces.Box(low=-np.inf, high=np.inf, shape=(10, 6), dtype=np.float32),
        #     'agent_obs': spaces.Box(low=-np.inf, high=np.inf, shape=(12, 3, 6), dtype=np.float32),
        #     'routing_obs': spaces.Box(low=-np.inf, high=np.inf, shape=(5, 20, 3), dtype=np.float32),
        #     'lanes_obs': spaces.Box(low=-np.inf, high=np.inf, shape=(50, 20, 7), dtype=np.float32),
        # })
        # self.observation_space = spaces.Box(low=-1, high=1, shape=(10, 6), dtype=float)

        self.action_space = utils.batch_space(self.single_action_space, self.num_envs)

        # super speed setting
        def reset(scenarios):
            init_state = self._env.reset(scenarios)
            return init_state
        self._reset = jax.jit(reset, backend=self.backend)

        def step(state, action, init_state):
            now_state = self._env.step(state, action, init_state)
            return now_state
        
        self._step = jax.jit(step, backend=self.backend)

    def reset(self):
        self.now_scenarios = next(self.dataset_iter)
        self.init_state = self.now_state = self._reset(self.now_scenarios)
        print('*' * 1000)

        obs = jnp.zeros((self.cfg['env_nums'], 10, 6), dtype=jnp.float32)

        return obs
    
    def step(self, action):
        action = datatypes.Action(
            data=action, valid=jnp.ones_like(action[..., 0:1], dtype=jnp.bool_)
        )

        self.now_state = self._step(self.now_state, action, self.init_state)

        obs = jnp.zeros((self.cfg['env_nums'], 10, 6), dtype=jnp.float32)
        rew = jnp.zeros((self.cfg['env_nums']), dtype=jnp.float32)
        # done = jnp.bool_([False] * self.cfg['env_nums'])
        done = self.now_state.done
        return obs, rew, done, {}


class WaymaxEnvWrapper(BraxEnv):
    '''A Gym wrapper for VecEnv, with interfaces for converting between JAX and Torch.
    '''
    def __init__(self, config_name, num_actors, **kwargs):

        kwargs['env_nums'] = num_actors
        self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)

        self.observation_space = self.env.single_observation_space
        self.action_space = self.env.single_action_space


class BaseWrapper:
    def __init__(self, env):
        self.env = env

    def reset(self, state):
        return self.env.reset(state)
    
    def __getattr__(self, name):
        if name == '__setstate__':
            raise AttributeError(name)
        return getattr(self.env, name)


class VmapWrapper(BaseWrapper):
    def __init__(self, env, batch_size):
        super().__init__(env)
        self.batch_size = batch_size
    
    def reset(self, state):
        return jax.vmap(self.env.reset)(state)
    
    def step(self, state, action):
        return jax.vmap(self.env.step)(state, action)


class AutoResetWrapper(BaseWrapper):

    def reset(self, orig_state):
        state = self.env.reset(orig_state)
        return state

    def step(self, state, action, init_state):
        # state = state.replace(done=jnp.zeros_like(state.done))
        state = self.env.step(state, action)

        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
            return jnp.where(done, x, y)

        obs = jax.tree_util.tree_map(where_done, init_state.observation, state.observation)
        orig_state = jax.tree_util.tree_map(where_done, init_state.state, state.state)
        state = state.replace(observation=obs, state=orig_state)      
        return state