## Reproduce PPO with PARL
Based on PARL, the PPO algorithm of deep reinforcement learning has been reproduced, reaching the same level of indicators as the paper in mujoco benchmarks.

> Paper: PPO in [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

### Mujoco/Atari games introduction
Please see [mujoco-py](https://github.com/openai/mujoco-py) to know more about Mujoco games or [atari](https://gym.openai.com/envs/#atari) to know more about Atari games.

### Benchmark result
#### 1. Mujoco games results
<p align="center">
<img src="https://github.com/benchmarking-rl/PARL-experiments/blob/master/PPO/torch/mujoco_result.png" alt="mujoco-result"/>
</p>

#### 2. Atari games results
<p align="center">
<img src="https://github.com/benchmarking-rl/PARL-experiments/blob/master/PPO/torch/atari_result.png" alt="atari-result"/>
</p>

+ Each experiment was run three times with different seeds

## How to use
### Dependencies:
+ python>=3.6.2
+ [parl>2.0.5](https://github.com/PaddlePaddle/PARL)
+ pytorch
+ gym==0.21.0
+ mujoco-py==2.1.2.14

### Local Training:

```
# To train an agent for discrete action game (Atari: PongNoFrameskip-v4 by default)
python train.py

# To train an agent for continuous action game (Mujoco)
python train.py --env 'HalfCheetah-v2' --continuous_action
```

### Distributed Training
Accelerate training process when `env_num > 1`.     
At first, we can start a local cluster with 8 CPUs:

```
xparl start --port 8010 --cpu_num 8
```

Note that if you have started a master before, you don't have to run the above
command. For more information about the cluster, please refer to our
[documentation](https://parl.readthedocs.io/en/latest/parallel_training/setup.html).

Then we can start the distributed training by running:

```
# To train an agent distributedly
# for discrete action game (Atari games)
python train.py --env "PongNoFrameskip-v4" --env_num 8 --xparl_addr 'localhost:8010'
# for continuous action game (Mujoco games)
python train.py --env 'HalfCheetah-v2' --continuous_action --env_num 5 --xparl_addr 'localhost:8010'
```
#### Training time Comparison

Training time comparison for 10M steps in Atari games.
|  Environment         | env_num  | Time/h (local, distributed) |
|----|----|----|
|  PongNoFrameskip-v4  | 8  | 61.67, 29.05 |
| BreakoutNoFrameskip-v4  | 8 | 578.05,  476.2|
