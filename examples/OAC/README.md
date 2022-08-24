## Reproduce OAC with PARL
Based on PARL, the OAC algorithm of deep reinforcement learning has been reproduced, reaching the same level of indicators as the paper in Mujoco benchmarks.

> Paper: OAC in [Better Exploration with Optimistic Actor-Critic](https://arxiv.org/abs/1910.12807)

### Mujoco games introduction
Please see [here](https://github.com/openai/mujoco-py) to know more about Mujoco games.

### Benchmark result

<img src="https://github.com/benchmarking-rl/PARL-experiments/blob/master/OAC/paddle/result.png" width="600" alt="OAC_results"/>

## How to use
### Dependencies:
+ python3.5+
+ [parl>=2.0.0](https://github.com/PaddlePaddle/PARL)
+ [paddlepaddle>=2.0.0](https://github.com/PaddlePaddle/Paddle)
+ gym==0.21.0
+ mujoco-py==2.1.2.14

### Start Training:
```
# To train an agent for Humanoid-v2 game
python train.py

# To train for other game
python train.py --env [ENV_NAME]
