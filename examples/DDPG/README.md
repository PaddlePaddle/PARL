## Reproduce DDPG with PARL
Based on PARL, the DDPG algorithm of deep reinforcement learning has been reproduced, reaching the same level of indicators as the paper in Mujoco benchmarks.

> Paper: DDPG in [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)

### Mujoco games introduction
Please see [here](https://github.com/openai/mujoco-py) to know more about Mujoco games.

### Benchmark result

<img src=".benchmark/DDPG_results.png" width = "800" height ="400" alt="DDPG_results"/>
+ Each experiment was run three times with different seeds

## How to use
### Dependencies:
+ python3.5+
+ [parl>=2.0.0](https://github.com/PaddlePaddle/PARL)
+ [paddlepaddle>=2.0.0](https://github.com/PaddlePaddle/Paddle)
+ gym==0.9.1
+ mujoco-py==0.5.7

### Start Training:
```
# To train an agent for HalfCheetah-v1 game
# python train.py

# To train for other game
# python train.py --env [ENV_NAME]
