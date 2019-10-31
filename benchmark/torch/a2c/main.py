import time
from learner import Learner
import torch
import os
import argparse
from parl.utils import logger


def main(config):
    cuda = torch.cuda.is_available()
    learner = Learner(config, cuda)
    assert config['log_metrics_interval_s'] > 0

    while not learner.should_stop():
        start = time.time()
        while time.time() - start < config['log_metrics_interval_s']:
            learner.step()
        learner.log_metrics()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=3e-4, help='learning_rate')
    parser.add_argument(
        '--vf_loss_coeff',
        default=0.5,
        help='hyper-parameter for the value function loss')
    args = parser.parse_args()
    from a2c_config import config
    logger.set_dir(
        os.path.join('./train_log', 'lr_{}_vf_{}_norm'.format(
            args.lr, args.vf_loss_coeff)))
    config['start_lr'] = float(args.lr)
    config['vf_loss_coeff'] = float(args.vf_loss_coeff)
    main(config)
