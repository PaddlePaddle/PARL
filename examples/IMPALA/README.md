## Reproduce IMPALA with PARL
Based on PARL, the IMPALA algorithm of deep reinforcement learning is reproduced, and the same level of indicators of the paper is reproduced in the classic Atari game.

> Paper: IMPALA in [Impala: Scalable distributed deep-rl with importance weighted actor-learner architectures](https://arxiv.org/abs/1802.01561)

### Atari games introduction
Please see [here](https://gym.openai.com/envs/#atari) to know more about Atari games.

### Benchmark result
Learning curve with one learner (in a P40 GPU) and 32 actors (in 32 CPUs).
+ PongNoFrameskip-v4: mean_episode_rewards can reach 18-19 score in about 10 minutes.
<img src="https://github.com/benchmarking-rl/PARL-experiments/raw/master/IMPALA/Pong.png" width = "400" height ="300" alt="IMPALA_Pong" />

+ Learning curves (mean_episode_rewards) of other games in an hour.

<img src="https://github.com/benchmarking-rl/PARL-experiments/raw/master/IMPALA/FourEnvs.png" width = "800" height ="600" alt="IMPALA_others" /> 

## How to use
### Dependencies
+ [paddlepaddle>=2.0.0](https://github.com/PaddlePaddle/Paddle)
+ [parl>=2.1.1](https://github.com/PaddlePaddle/PARL)
+ gym==0.12.1
+ atari-py==0.1.7
+ opencv-python

### Distributed Training:

At first, We can start a local cluster with 32 CPUs:

```bash
xparl start --port 8010 --cpu_num 32
```

Note that it is not necessary to run the command each time before training. 
We can reuse the xparl cluster for distributed training if we have started it before.
[documentation](https://parl.readthedocs.io/en/latest/parallel_training/setup.html)

Then we can start the distributed training by running:

```bash
python train.py
```

### Reference
+ [Parl Cluster Setup](https://parl.readthedocs.io/en/latest/parallel_training/setup.html).
+ [deepmind/scalable_agent](https://github.com/deepmind/scalable_agent)
+ [Ray](https://github.com/ray-project/ray)
