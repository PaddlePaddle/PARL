'''waymax_rl train script
'''
from utils.init_set import init_set
init_set()

import hydra
from omegaconf import DictConfig, OmegaConf

from rl_games.common import env_configurations, vecenv

from utils.build_runner import build_alg_runner
from utils.algo_observer import PPOAlgoObserver
from utils.env_wrapper import VecWaymaxEnv, WaymaxEnvWrapper


vecenv.register('WAYMAX', lambda config_name, num_actors, **kwargs: WaymaxEnvWrapper(config_name, num_actors, **kwargs))
env_configurations.register('waymax', {
    'env_creator': lambda **kwargs: VecWaymaxEnv(**kwargs),
    'vecenv_type': 'WAYMAX'})


@hydra.main(version_base=None, config_path="conf", config_name="ppo_config")
def train(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)

    algo_observer = PPOAlgoObserver()
    runner = build_alg_runner(algo_observer)
    runner.load(cfg)
    runner.reset()
    runner.run({
        'train': True,
    })


if __name__ == '__main__':
    train()