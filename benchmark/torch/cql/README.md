## Reproduce CQL with PARL
Based on PARL, the CQL algorithm of deep reinforcement learning has been reproduced, reaching the same level of indicators as the paper on the MuJoCo environments with D4RL Dataset.

> Paper: CQL in [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/abs/2006.04779)

### Environment and dataset introduction
+ Mujoco games: Please see [here](https://github.com/openai/mujoco-py) to know more about Mujoco games.
+ D4RL datasets: Please see [here](https://sites.google.com/view/d4rl/home) to know more about D4RL datasets.

### Benchmark result

![learning curve](https://github.com/benchmarking-rl/PARL-experiments/blob/master/CQL/torch/result.png)

+ Each experiment was run three times with different seeds

## How to use
### Dependencies:
+ python3.5+
+ [parl>2.0.2](https://github.com/PaddlePaddle/PARL)
+ [paddlepaddle>=2.0.0](https://github.com/PaddlePaddle/Paddle)
+ gym==0.20.0
+ mujoco-py==2.0.2.13
+ [d4rl](https://github.com/rail-berkeley/d4rl) (install from source)

### Start Training:
#### Train
```
# To train for halfcheetah-medium-expert-v0(default), or [halfcheetah/hopper/walker]-[random/medium/expert/medium-expert]-v0
python train.py --env [ENV_NAME]

# To reproduce the performance of halfcheetah-medium-expert-v0
python train.py --env halfcheetah-medium-expert-v0 --with_automatic_entropy_tuning
