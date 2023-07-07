## Reproduce PPO with PARL
Based on PARL, the PPO algorithm of deep reinforcement learning has been reproduced, reaching the same level of indicators as the paper in mujoco benchmarks.

> Paper: PPO in [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

### Mujoco/Atari games introduction
PARL currently supports the open-source version of Mujoco provided by DeepMind, so users do not need to download binaries of Mujoco as well as install [mujoco-py](https://github.com/openai/mujoco-py#install-mujoco). For more details, please visit [Mujoco](https://github.com/deepmind/mujoco).

### Benchmark result
#### 1. Mujoco games results
The horizontal axis represents the number of episodes.
<p align="center">
<img src="https://github.com/benchmarking-rl/PARL-experiments/blob/master/PPO/paddle/mujoco_result.png" alt="mujoco-result"/>
</p>

#### 2. Atari games results
The horizontal axis represents the number of steps.
<p align="center">
<img src="https://github.com/benchmarking-rl/PARL-experiments/blob/master/PPO/paddle/atari_result.png" alt="atari-result"/>
</p>

+ Each experiment was run three times with different seeds

## How to use
### Mujoco-Dependencies:
+ python3.7+
+ [paddle>=2.3.1](https://github.com/PaddlePaddle/Paddle)
+ [parl>=2.2.2](https://github.com/PaddlePaddle/PARL)
+ gym==0.18.0
+ mujoco>=2.2.2
+ mujoco-py==2.1.2.14

### Atari-Dependencies:
+ [paddle>=2.3.1](https://github.com/PaddlePaddle/Paddle)
+ [parl>=2.2.2](https://github.com/PaddlePaddle/PARL)
+ gym==0.18.0
+ atari-py==0.2.6
+ opencv-python


### Training Mujoco Distributedly
Accelerate training process by setting `xparl_addr` and `env_num > 1` when environment simulation running very slowly.        
At first, we can start a local cluster with 8 CPUs:

```
xparl start --port 8010 --cpu_num 8
```

Note that if you have started a master before, you don't have to run the above
command. For more information about the cluster, please refer to our
[documentation](https://parl.readthedocs.io/en/latest/parallel_training/setup.html).

Then we can start the distributed training for mujoco games by running:

```
cd mujoco

python train.py --env 'HalfCheetah-v2' --train_total_episodes 100000 --env_num 5
```


### Training Atari
To train an agent for discrete action game (Atari: PongNoFrameskip-v4 by default):

```
cd atari

# Local training
python train.py

# Distributed training
xparl start --port 8010 --cpu_num 8
python train.py --env "PongNoFrameskip-v4" --env_num 8 --xparl_addr 'localhost:8010'
```
