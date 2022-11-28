## Reproduce OAC with PARL
Based on PARL, the OAC algorithm of deep reinforcement learning has been reproduced, reaching the same level of indicators as the paper in Mujoco benchmarks.

> Paper: OAC in [Better Exploration with Optimistic Actor-Critic](https://arxiv.org/abs/1910.12807)

### Mujoco games introduction
PARL currently supports the open-source version of Mujoco provided by DeepMind, so users do not need to download binaries of Mujoco as well as install mujoco-py and get license. For more details, please visit [Mujoco](https://github.com/deepmind/mujoco).

### Benchmark result

<img src="https://github.com/benchmarking-rl/PARL-experiments/blob/master/OAC/torch/result.png" width="600" alt="OAC_results"/>

+ Each experiment was run three times with different seeds

## How to use
### Dependencies:
+ python3.7+
+ [parl>=2.1.1](https://github.com/PaddlePaddle/PARL)
+ gym>=0.26.0
+ torch
+ mujoco>=2.2.2

### Start Training:
```train
# To train an agent for HalfCheetah-v4 game
python train.py

# To train for other game & params
python train.py --env [ENV_NAME] --alpha [float] --beta [float] --delta [float]
```

### Reference
+ [microsoft/oac-explore](https://github.com/microsoft/oac-explore)
