import parl
import numpy as np
from osim.env import L2M2019Env
from env_wrapper import FrameSkip, ActionScale, OfficialObs, FinalReward, FirstTarget


@parl.remote_class
class Actor(object):
    def __init__(self,
                 difficulty,
                 vel_penalty_coeff,
                 muscle_penalty_coeff,
                 penalty_coeff,
                 only_first_target=False):

        random_seed = np.random.randint(int(1e9))

        env = L2M2019Env(
            difficulty=difficulty, visualize=False, seed=random_seed)
        max_timelimit = env.time_limit

        env = FinalReward(
            env,
            max_timelimit=max_timelimit,
            vel_penalty_coeff=vel_penalty_coeff,
            muscle_penalty_coeff=muscle_penalty_coeff,
            penalty_coeff=penalty_coeff)

        if only_first_target:
            assert difficulty == 3, "argument `only_first_target` is available only in `difficulty=3`."
            env = FirstTarget(env)

        env = FrameSkip(env)
        env = ActionScale(env)
        self.env = OfficialObs(env, max_timelimit=max_timelimit)

    def reset(self):
        observation = self.env.reset(project=False)
        return observation

    def step(self, action):
        return self.env.step(action, project=False)
