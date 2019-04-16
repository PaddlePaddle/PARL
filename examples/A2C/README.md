## Reproduce A2C with PARL
Based on PARL, the A2C algorithm of deep reinforcement learning is reproduced, and the same level of indicators of the paper is reproduced in the classic Atari game.

A2C is a synchronous, deterministic variant of [Asynchronous Advantage Actor Critic (A3C)](https://arxiv.org/abs/1602.01783)

### Atari games introduction
Please see [here](https://gym.openai.com/envs/#atari) to know more about Atari game.

### Benchmark result
Results with one learner (in P40 GPU) and 5 actors in 10 million sample steps.
<img src=".benchmark/A2C_Pong.jpg" width = "400" height ="300" alt="A2C_Pong" /> <img src=".benchmark/A2C_Breakout.jpg" width = "400" height ="300" alt="A2C_Breakout"/>

## How to use
### Dependencies
+ python2.7 or python3.5+
+ [paddlepaddle>=1.3.0](https://github.com/PaddlePaddle/Paddle)
+ [parl](https://github.com/PaddlePaddle/PARL)
+ gym
+ atari_py


### Distributed Training

#### Learner
```sh
python train.py 
```

#### Actors (Suggest: 5 actors in 5 CPUs)
```sh
for index in {1..5}; do
    python actor.py &
done;
wait
```

You can change training settings (e.g. `env_name`, `server_ip`) in `a2c_config.py`.
Training result will be saved in `log_dir/train/result.csv`.

### Reference
+ [Ray](https://github.com/ray-project/ray)
