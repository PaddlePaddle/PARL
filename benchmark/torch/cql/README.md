## Reproduce CQL with PARL

Different from Online RL which gather new data directly from the environment, Offline RL algorithms learn a policy or Q-function from static dataset, without additional interaction with the environment.

Based on PARL, the CQL algorithm of deep reinforcement learning has been reproduced, reaching the same level of indicators as the paper on continuous control datasets from the D4RL benchmark.

> Paper: CQL in [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/abs/2006.04779)

### Env and dataset introduction
+ D4RL datasets: Please see [here](https://sites.google.com/view/d4rl/home) to know more about D4RL datasets. D4RL require Mujoco as a dependency. For more D4RL usage methods, please refer to its [guide](https://github.com/rail-berkeley/d4rl#using-d4rl).
+ Mujoco simulator: Please see [here](http://mujoco.org/) to know more about Mujoco simulator and obtain a license.

### Benchmark result

![learning curve](https://github.com/benchmarking-rl/PARL-experiments/blob/master/CQL/torch/result.png)

+ Each experiment was run three times with different seeds

## How to use
### Dependencies:
+ python3.5+
+ [parl>2.0.2](https://github.com/PaddlePaddle/PARL)
+ torch
+ gym==0.20.0
+ mujoco-py==2.0.2.13
+ [d4rl](https://github.com/rail-berkeley/d4rl) (install from source)

### Start Training:
#### Train
```
# To train for halfcheetah-medium-expert-v0(default), or [halfcheetah/hopper/walker/ant]-[random/medium/expert/medium-expert/medium-replay]-[v0/v2]
python train.py --env [ENV_NAME]

# To reproduce the performance of halfcheetah-medium-expert-v0
python train.py --env halfcheetah-medium-expert-v0 --with_automatic_entropy_tuning
