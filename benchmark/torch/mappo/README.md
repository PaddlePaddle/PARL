## Reproduce MAPPO with PARL

Based on PARL, the MAPPO algorithm of deep reinforcement learning has been reproduced.

> Paper: MAPPO in [ The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955)

### Multi-agent particle environment introduction

A simple multi-agent particle world based on gym. Please see [here](https://github.com/benchmarking-rl/PARL-experiments/tree/master/MAPPO/env) to install the environment, which is a modified [MPE](https://github.com/openai/multiagent-particle-envs) env extracted from [MAPPO](https://github.com/marlbenchmark/on-policy/tree/main/onpolicy/envs) official code.

### Benchmark result

Mean episode reward (every 128 episodes) in training process.

<p align="center">
<img src="https://github.com/benchmarking-rl/PARL-experiments/blob/master/MAPPO/pc/result.png" alt="result"/>
</p>

### Experiments result

<table align="center">
<tr>
<td>
simple_reference<br>
<img src="https://github.com/benchmarking-rl/PARL-experiments/blob/master/MAPPO/pc/simple_reference.gif"                  width = "200" height = "200" alt="MAPPO_simple_reference"/>
</td>
<td>
simple_speaker_listener<br>
<img src="https://github.com/benchmarking-rl/PARL-experiments/blob/master/MAPPO/pc/simple_speaker_listener.gif"        width = "200" height = "200" alt="MAPPO_simple_speaker_listener"/>
</td>
<td>
simple_spread<br>
<img src="https://github.com/benchmarking-rl/PARL-experiments/blob/master/MAPPO/pc/simple_spread.gif"             width = "200" height = "200" alt="MAPPO_simple_spread"/>
</td>
</tr>
</table>

## How to use

### Dependencies:

+ python>=3.6
+ [parl>=2.0.4](https://github.com/PaddlePaddle/PARL)
+ [mappo-mpe](https://github.com/benchmarking-rl/PARL-experiments/tree/master/MAPPO/env)
+ numpy==1.18.5
+ torch==1.5.1
+ seaborn

### Start Training:

```
# To train an agent for simple_speaker_listener scenario
python train.py

# To train for other scenario
# python train.py --env_name [ENV_NAME]
```
