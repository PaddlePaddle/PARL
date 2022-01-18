## Reproduce MAPPO with PARL

Based on PARL, the MAPPO algorithm of deep reinforcement learning has been reproduced.

> Paper: MAPPO in [ The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955)

### Multi-agent particle environment introduction

A simple multi-agent particle world based on gym. Please see [here](https://github.com/caoxixiya/mappo_mpe) to install the environment, which is a light [MPE](https://github.com/openai/multiagent-particle-envs) env modified from [MAPPO](https://github.com/marlbenchmark/on-policy/tree/main/onpolicy/envs) official code.

## How to use

### Dependencies:

+ python>=3.8
+ [parl>=2.0.4](https://github.com/PaddlePaddle/PARL)
+ [mappo-mpe](https://github.com/caoxixiya/mappo_mpe)
+ numpy==1.18.5
+ torch==1.5.1
+ seaborn

### Start Training:

```python
# To train an agent for simple_speaker_listener scenario
python train.py

# To train for other scenario
python train.py --scenario_name [ENV_NAME]
```