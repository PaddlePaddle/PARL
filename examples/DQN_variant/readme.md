## Reproduce DQN with PARL
Based on PARL, the DQN algorithm of deep reinforcement learning has been reproduced, reaching the same level of indicators as the paper in Atari benchmarks.

+ Papers: 

> DQN in [Human-level Control Through Deep Reinforcement Learning](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html)

> DDQN in [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)

> Dueling DQN in [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)

### Atari games introduction
Please see [here](https://gym.openai.com/envs/#atari) to know more about Atari games.

### Benchmark results

<<<<<<< HEAD
*Benchmark results are obtained using different random seeds.*
=======
Performance of DQN on various environments

<p align="center">
<img src=".benchmark/DQN.png" alt="result"/>
</p>
>>>>>>> b4e37d07564f19d3bad864687f2abc0dda380e25

Performance of DDQN on various environments

<p align="center">
<img src=".benchmark/DDQN.png" alt="result" width="1200"/>
</p>

Performance of Dueling DQN on various environments

<p align="center">
<<<<<<< HEAD
<img src=".benchmark/DQN.png" alt="result"/>
</p>

Performance of DDQN on various environments

<p align="center">
<img src=".benchmark/DDQN.png" alt="result"/>
</p>

Performance of Dueling DQN on various environments

<p align="center">
<img src=".benchmark/Dueling DQN.png" alt="result"/>
=======
<img src=".benchmark/Dueling DQN.png" alt="result" width="1200"/>
>>>>>>> b4e37d07564f19d3bad864687f2abc0dda380e25
</p>

## How to use
### Dependencies:
+ [paddlepaddle>=2.0.0](https://github.com/PaddlePaddle/Paddle)
+ [parl](https://github.com/PaddlePaddle/PARL)
+ gym
+ tqdm
+ atari-py
+ [ale_python_interface](https://github.com/mgbellemare/Arcade-Learning-Environment)


### Start Training:
```
# To train a model for Pong game
python train.py --rom ./rom_files/pong.bin

# For more customized arguments
python train.py --help
```

> To train more games, you can install more rom files from [here](https://github.com/openai/atari-py/tree/master/atari_py/atari_roms).
