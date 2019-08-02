## Reproduce IMPALA with PARL
Based on PARL, the IMPALA algorithm of deep reinforcement learning is reproduced, and the same level of indicators of the paper is reproduced in the classic Atari game.

+ IMPALA in
[Impala: Scalable distributed deep-rl with importance weighted actor-learner architectures](https://arxiv.org/abs/1802.01561)

### Atari games introduction
Please see [here](https://gym.openai.com/envs/#atari) to know more about Atari games.

### Benchmark result
Result with one learner (in a P40 GPU) and 32 actors (in 32 CPUs).
+ PongNoFrameskip-v4: mean_episode_rewards can reach 18-19 score in about 7~10 minutes.
<img src=".benchmark/IMPALA_Pong.jpg" width = "400" height ="300" alt="IMPALA_Pong" />

+ Results of other games in an hour.

<img src=".benchmark/IMPALA_Breakout.jpg" width = "400" height ="300" alt="IMPALA_Breakout" /> <img src=".benchmark/IMPALA_BeamRider.jpg" width = "400" height ="300" alt="IMPALA_BeamRider"/>
<br>
<img src=".benchmark/IMPALA_Qbert.jpg" width = "400" height ="300" alt="IMPALA_Qbert" /> <img src=".benchmark/IMPALA_SpaceInvaders.jpg" width = "400" height ="300" alt="IMPALA_SpaceInvaders"/>

## How to use
### Dependencies
+ [paddlepaddle>=1.5.1](https://github.com/PaddlePaddle/Paddle)
+ [parl](https://github.com/PaddlePaddle/PARL)
+ gym
+ atari-py


### Distributed Training:

We can start a local cluster with 8 CPUs by executing the `xparl start`
command:

```bash
xparl start --port 8010 --cpu_num 8
```

After the cluster is started, we can add more computation resources to our
cluster with the `xparl connect` command at any time and on any machine.

```bash
xparl connect --address master_address
```

Then we can start the distributed training by running `train.py`.

```bash
python train.py
```

If we have an existing cluster running at `cluster_address`, we can start a new
training task with this cluster by setting `'master_address' = cluster_address`
in the `impala_config.py`.

Training result will be saved in `log_dir/train/log.log` and the cluster logs
will be saved in `~/.parl_data/`. For more detailed information about the
usage of the parl cluster, please refer to our official document
[Parl Cluster Setup](https://parl.readthedocs.io/en/latest/parallel_training/setup.html).

### Reference
+ [Parl Cluster Setup](https://parl.readthedocs.io/en/latest/parallel_training/setup.html).
+ [deepmind/scalable_agent](https://github.com/deepmind/scalable_agent)
+ [Ray](https://github.com/ray-project/ray)
