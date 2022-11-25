## Reproduce PPO with PARL
Based on PARL, the PPO algorithm of deep reinforcement learning has been reproduced, reaching the same level of indicators as the paper in mujoco benchmarks.

> Paper: PPO in [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

### Mujoco/Atari games introduction
Please see [atari](https://gym.openai.com/envs/#atari) to know more about Atari games.
PARL currently supports the open-source version of Mujoco provided by DeepMind, so users do not need to download binaries of Mujoco as well as install mujoco-py and get license. For more details, please visit [Mujoco](https://github.com/deepmind/mujoco).

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
### Mujoco-Dependencies:
+ python>=3.7+
+ [parl>=2.1.1](https://github.com/PaddlePaddle/PARL)
+ pytorch
+ gym>=0.26.0
+ mujoco>=2.2.2

### Atari-Dependencies:
+ python>=3.7+
+ [parl>=2.1.1](https://github.com/PaddlePaddle/PARL)
+ pytorch
+ gym==0.18.0
+ atari-py==0.2.6
+ opencv-python

### Training:

```
# To train an agent for discrete action game (Atari: PongNoFrameskip-v4 by default)
python train.py

# To train an agent for continuous action game (Mujoco)
python train.py --env 'HalfCheetah-v4' --continuous_action
```

### Distributed Training
Accelerate training process by setting `xparl_addr` and `env_num > 1` when environment simulation running very slow.        
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
python train.py --env 'HalfCheetah-v4' --continuous_action --env_num 5 --xparl_addr 'localhost:8010'
```
