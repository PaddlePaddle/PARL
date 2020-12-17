## Reproduce DDPG with PARL
Based on PARL, the DDPG algorithm of deep reinforcement learning has been reproduced, reaching the same level of indicators as the paper in Mujoco benchmarks.

> **Continuous control with Deep Reinforcement Learning**
>
> [Continuous control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971)

About DDPG:

+ A model-free, off-policy algorithm.
+ Can only be used for environment with continuous action spaces.
+ An continuous thought of DQN.

Tricks:

+ Replay memory

+ Target network: **soft** target updates

+ **Batch normalization**(Not implement)

+ Noise for exploration

  

### Mujoco games introduction

Please see [here](https://github.com/openai/mujoco-py) to know more about Mujoco games.

### Benchmark result

<img src=".benchmark/train.png" width = "800" height ="300" alt="Performance" />

<img src=".benchmark/evaluation.png" width = "800" height ="300" alt="Performance" />

## How to use
### Dependencies:
+ python
+ [parl](https://github.com/PaddlePaddle/PARL)
+ gym
+ torch
+ mujoco-py>=1.50.1.0

### Start Training:
```
# To train an agent for HalfCheetah-v1 game
python train.py

# To train for different game and different loss type
# python train.py --env [ENV_NAME]

```