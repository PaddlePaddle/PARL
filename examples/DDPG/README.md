## Reproduce DDPG with PARL
Based on PARL, the DDPG algorithm of deep reinforcement learning has been reproduced, reaching the same level of indicators as the paper in Mujoco benchmarks.

> Paper: DDPG in [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)

### Mujoco games introduction
PARL currently supports the open-source version of Mujoco provided by DeepMind, so users do not need to download binaries of Mujoco as well as install mujoco-py and get license. For more details, please visit [Mujoco](https://github.com/deepmind/mujoco).

### Benchmark result

<img src="https://github.com/benchmarking-rl/PARL-experiments/blob/master/DDPG/paddle/result.png" width="600" alt="DDPG_results"/>
+ Each experiment was run three times with different seeds

## How to use
### Dependencies:
+ python3.7+
+ [parl>=2.1.1](https://github.com/PaddlePaddle/PARL)
+ [paddlepaddle>=2.0.0](https://github.com/PaddlePaddle/Paddle)
+ gym>=0.26.0
+ mujoco>=2.2.2

### Start Training:
```
# To train an agent for HalfCheetah-v4 game
# python train.py

# To train for other game
# python train.py --env [ENV_NAME]
