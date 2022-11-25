## Reproduce TD3 with PARL
Based on PARL, the TD3 algorithm of deep reinforcement learning has been reproduced, reaching the same level of indicators as the paper in Mujoco benchmarks.

Include following improvements:
+ Clipped Double Q-learning
+ Target Networks and Delayed Policy Update
+ Target Policy Smoothing Regularization

> TD3 in
[Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)

### Mujoco games introduction
PARL currently supports the open-source version of Mujoco provided by DeepMind, so users do not need to download binaries of Mujoco as well as install mujoco-py and get license. For more details, please visit [Mujoco](https://github.com/deepmind/mujoco)

### Benchmark result

<img src="https://github.com/benchmarking-rl/PARL-experiments/blob/master/TD3/paddle/result.png" alt="TD3_results"/>
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
python train.py

# To train for different game
python train.py --env [ENV_NAME]
