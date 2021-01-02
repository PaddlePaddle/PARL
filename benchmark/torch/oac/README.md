## Reproduce OAC with PARL
Based on PARL, the OAC algorithm of deep reinforcement learning has been reproduced, reaching the same level of indicators as the paper in Mujoco benchmarks.

> Paper: OAC in [Better Exploration with Optimistic Actor-Critic](https://arxiv.org/abs/1910.12807)

### Mujoco games introduction
Please see [here](https://github.com/openai/mujoco-py) to know more about Mujoco games.

### Benchmark result

<img src=".benchmark/OAC_results.png" width = "800" height ="400" alt="OAC_results"/>

## How to use
### Dependencies:
+ python3.5+
+ [parl](https://github.com/PaddlePaddle/PARL)
+ gym
+ torch
+ mujoco-py>=1.50.1.0

### Start Training:
#### Train

```
# To train an agent for HalfCheetah-v1 game
python train.py

# To train for other game & for automatic entropy tuning
# python train.py --env [ENV_NAME] ---automatic_entropy_tuning True
