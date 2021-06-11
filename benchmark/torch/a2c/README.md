## Reproduce A2C with PARL
Based on PARL, the A2C algorithm of deep reinforcement learning has been reproduced, reaching the same level of indicators as the paper in Atari benchmarks.

### Atari game introduction
Please see [here](https://gym.openai.com/envs/#atari) to know more about Atari games.

### Benchmark result
Mean episode reward of A2C on some atari games in training process after 10 million sample steps.

<p align="center">
<img src=".result/result_a2c_torch0.png" alt="result"/>
</p>
<p align="center">
<img src=".result/result_a2c_torch1.png" alt="result"/>
</p>


## How to use
### Dependencies
+ [parl>=1.4.3](https://github.com/PaddlePaddle/PARL)
+ gym==0.18.0
+ atari-py==0.2.6
+ torch==1.7.1


### Distributed Training

At first, we can start a local cluster with 5 CPUs:

```bash
xparl start --port 8010 --cpu_num 5
```

Note that if you have started a master before, you don't have to run the above
command. For more information about the cluster, please refer to our
[documentation](https://parl.readthedocs.io/en/latest/parallel_training/setup.html)

Then we can start the distributed training by running:

```bash
python train.py
```

### Reference
+ [Parl](https://parl.readthedocs.io/en/latest/parallel_training/setup.html)
+ [Ray](https://github.com/ray-project/ray)
+ [OpenAI Baselines: ACKTR & A2C](https://openai.com/blog/baselines-acktr-a2c/)
