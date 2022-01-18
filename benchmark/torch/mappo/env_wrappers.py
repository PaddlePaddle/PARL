"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import numpy as np
from mpe.environment import MultiAgentEnv
from mpe.scenarios import load


def MPEEnv(scenario_name, rank, seed):
    # load scenario from script
    scenario = load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                        scenario.observation, scenario.info)
    env.seed(seed + rank * 1000)
    return env


class ParallelEnv(object):
    """
    Creates a simple vectorized wrapper for multiple environments, calling each environment in sequence on the current
    Python process. This is useful for computationally simple environment such as ``cartpole-v1``,
    as the overhead of multiprocess or multithread outweighs the environment computation time.
    This can also be used for RL methods that
    require a vectorized environment, but that you want a single environments to train with.
    """
    closed = False
    viewer = None

    def __init__(self, scenario_name, env_num, seed):
        self.envs = [
            MPEEnv(scenario_name, rank, seed) for rank in range(env_num)
        ]
        self.actions = None
        self.num_envs = env_num
        self.observation_space = self.envs[0].observation_space
        self.share_observation_space = self.envs[0].share_observation_space
        self.action_space = self.envs[0].action_space

    def reset(self):
        obs = [env.reset() for env in self.envs]
        return np.array(obs)

    def step(self, actions):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        temp_actions_env = []
        for agent_id in range(len(self.observation_space)):
            action = actions[agent_id]
            if self.action_space[
                    agent_id].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.action_space[agent_id].shape):
                    uc_action_env = np.eye(
                        self.action_space[agent_id].high[i] + 1)[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate(
                            (action_env, uc_action_env), axis=1)
            elif self.action_space[agent_id].__class__.__name__ == 'Discrete':
                action_env = np.squeeze(
                    np.eye(self.action_space[agent_id].n)[action], 1)
            else:
                raise NotImplementedError

            temp_actions_env.append(action_env)

        actions_env = []
        for i in range(self.num_envs):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
            actions_env.append(one_hot_action_env)

        self.actions = actions_env

        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))

        for (i, done) in enumerate(dones):
            if 'bool' in done.__class__.__name__:
                if done:
                    obs[i] = self.envs[i].reset()
            else:
                if np.all(done):
                    obs[i] = self.envs[i].reset()
        self.actions = None

        return obs, rews, dones, infos

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.closed = True
        for env in self.envs:
            env.close()
