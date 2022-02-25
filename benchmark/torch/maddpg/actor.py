import parl
from simple_model import MAModel
from simple_agent import MAAgent
from parl.algorithms import MADDPG
from parl.env.multiagent_simple_env import MAenv
from parl.utils import logger

CRITIC_LR = 0.01  # learning rate for the critic model
ACTOR_LR = 0.01  # learning rate of the actor model
GAMMA = 0.95  # reward discount factor
TAU = 0.01  # soft update
BATCH_SIZE = 1024
MAX_STEP_PER_EPISODE = 25  # maximum step per episode

@parl.remote_class(wait=False)
class Actor(object):
    def __init__(self, args):
        env = MAenv(args.env)

        from gym import spaces
        from multiagent.multi_discrete import MultiDiscrete
        for space in env.action_space:
            assert (isinstance(space, spaces.Discrete)
                    or isinstance(space, MultiDiscrete))

        critic_in_dim = sum(env.obs_shape_n) + sum(env.act_shape_n)
        logger.info('critic_in_dim: {}'.format(critic_in_dim))

        agents = []
        for i in range(env.n):
            model = MAModel(env.obs_shape_n[i], env.act_shape_n[i], critic_in_dim)
            algorithm = MADDPG(
                model,
                agent_index=i,
                act_space=env.action_space,
                gamma=GAMMA,
                tau=TAU,
                critic_lr=CRITIC_LR,
                actor_lr=ACTOR_LR)
            agent = MAAgent(
                algorithm,
                agent_index=i,
                obs_dim_n=env.obs_shape_n,
                act_dim_n=env.act_shape_n,
                batch_size=BATCH_SIZE,
                speedup=(not args.restore))
            agents.append(agent)

        self.env = env
        self.agents = agents
    
    def set_weights(self, weights):
        for i, agent in enumerate(self.agents):
            agent.set_weights(weights[i])

    def run_episode(self):
        obs_n = self.env.reset()
        total_reward = 0
        agents_reward = [0 for _ in range(self.env.n)]
        steps = 0

        experience = []
        while True:
            steps += 1
            action_n = [agent.predict(obs) for agent, obs in zip(self.agents, obs_n)]
            next_obs_n, reward_n, done_n, _ = self.env.step(action_n)
            done = all(done_n)
            terminal = (steps >= MAX_STEP_PER_EPISODE)

            # store experience
            experience.append((obs_n, action_n, reward_n,
                                     next_obs_n, done_n))

            # compute reward of every agent
            obs_n = next_obs_n
            for i, reward in enumerate(reward_n):
                total_reward += reward
                agents_reward[i] += reward

            # check the end of an episode
            if done or terminal:
                break
            
        return experience, total_reward, agents_reward, steps


