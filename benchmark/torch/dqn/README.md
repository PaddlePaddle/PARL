## Reproduce DQN with PARL
Based on PARL, the DQN algorithm of deep reinforcement learning has been reproduced, reaching the same level of indicators as the paper in Atari benchmarks.

+ Papers: 

> DQN in [Human-level Control Through Deep Reinforcement Learning](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html)

> DDQN in [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)

> Dueling DQN in [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)

### Atari games introduction
Please see [here](https://gym.openai.com/envs/#atari) to know more about Atari games.

### Benchmark results

*Benchmark results are obtained using different random seeds.*

Performance of **DQN** on various environments:

<p align="center">
<img src=".benchmark/dqn.png" alt="result"/>
</p>

## How to use
### Dependencies:
+ python>=3.6.2
+ [pytorch==1.7.1](https://pytorch.org/get-started/previous-versions/)
+ [parl](https://github.com/PaddlePaddle/PARL)
+ gym==0.18.0
+ tqdm
+ atari-py==0.2.6

### Start Training:
```
# To train a model for Pong game
python train.py

# For more customized arguments
python train.py --help
```
