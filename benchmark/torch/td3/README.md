## Reproduce TD3 with PARL
Based on PARL, the TD3 algorithm of deep reinforcement learning has been reproduced, reaching the same level of indicators as the paper in Mujoco benchmarks.

Include following approaches:
+ Clipped Double Q-learning
+ Target Networks and Delayed Policy Update
+ Target Policy Smoothing Regularization

> TD3 in
[Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)

### Mujoco games introduction
Please see [here](https://github.com/openai/mujoco-py) to know more about Mujoco games.

### Benchmark result

<img src="https://github.com/ljy2222/PARL-experiments/blob/master/TD3/torch/result.png" alt="Performance" />

## How to use
### Dependencies:
+ python
+ [parl](https://github.com/PaddlePaddle/PARL)
+ gym==0.21.0
+ torch
+ mujoco-py==2.1.2.14

### Start Training:
```
# To train an agent for HalfCheetah-v2 game
python train.py

# To train for different game and different loss type
# python train.py --env [ENV_NAME]
