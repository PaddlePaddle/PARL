## Reproduce DDPG with PARL
Based on PARL, the DDPG algorithm of deep reinforcement learning has been reproduced, reaching the same level of indicators as the paper in Mujoco benchmarks.

> Paper: DDPG in [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)

### Mujoco games introduction
Please see [here](https://github.com/openai/mujoco-py) to know more about Mujoco games.

### Benchmark result

<img src=".benchmark/HalfCheetac-v1.png" width = "800" height ="300" alt="DDPG_HalfCheetah-v1"/><img src=".benchmark/Hopper-v1.png" width = "800" height ="300" alt="DDPG_Hopper-v1"/>

## How to use
### Dependencies:
+ python3.5+
+ [parl](https://github.com/PaddlePaddle/PARL)
+ gym
+ torch
+ mujoco-py>=1.50.1.0

### Start Training:
```
# To train an agent for HalfCheetah-v1 game
python train.py

# To train for other game
# python train.py --env [ENV_NAME]
