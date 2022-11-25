## Reproduce SAC with PARL
Based on PARL, the SAC algorithm of deep reinforcement learning has been reproduced, reaching the same level of indicators as the paper in Mujoco benchmarks.

> Paper: SAC in [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)

### Mujoco games introduction
PARL currently supports the open-source version of Mujoco provided by DeepMind, so users do not need to download binaries of Mujoco as well as install mujoco-py and get license. For more details, please visit [Mujoco](https://github.com/deepmind/mujoco).

### Benchmark result

<img src="https://github.com/benchmarking-rl/PARL-experiments/blob/master/SAC/torch/result.png" alt="SAC_results"/>

+ Each experiment was run three times with different seeds

## How to use
### Dependencies:
+ python3.7+
+ [parl>=2.1.1](https://github.com/PaddlePaddle/PARL)
+ gym>=0.26.0
+ torch
+ mujoco>=2.2.2

### Start Training:
#### Train
```
# To train for HalfCheetah-v2(default),Hopper-v4,Walker2d-v4,Ant-v4
# --alpha 0.2(default)
python train.py --env [ENV_NAME]

# To reproduce the performance of Humanoid-v4
python train.py --env Humanoid-v4 --alpha 0.05
