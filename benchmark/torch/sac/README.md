## Reproduce SAC with PARL
Based on PARL, the SAC algorithm of deep reinforcement learning has been reproduced, reaching the same level of indicators as the paper in Mujoco benchmarks.

> Paper: SAC in [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)

### Mujoco games introduction
Please see [here](https://github.com/openai/mujoco-py) to know more about Mujoco games.

### Benchmark result

<img src="https://github.com/ljy2222/PARL-experiments/blob/master/SAC/torch/result.png" width = "800" height ="400" alt="SAC_results"/>

+ Each experiment was run three times with different seeds

## How to use
### Dependencies:
+ python3.5+
+ [parl](https://github.com/PaddlePaddle/PARL)
+ gym==0.21.0
+ torch
+ mujoco-py==2.1.2.14

### Start Training:
#### Train
```
# To train for HalfCheetah-v2(default),Hopper-v2,Walker2d-v2,Ant-v2
# --alpha 0.2(default)
python train.py --env [ENV_NAME]

# To reproduce the performance of Humanoid-v1
python train.py --env Humanoid-v1 --alpha 0.05
