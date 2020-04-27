## Reproduce DQN with PARL
Based on PARL, we provide a simple demonstration of DQN.

+ DQN in
[Human-level Control Through Deep Reinforcement Learning](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html)

### Result

Performance of DQN playing CartPole-v1

<p align="left">
<img src="https://parl.readthedocs.io/en/latest/_images/performance1.gif" alt="result" width="450"/>
</p>
<p align="left">
<img src="cartpole.jpg" alt="result" width="450"/>
</p>

## How to use
### Dependencies:
+ [paddlepaddle>=1.6.1](https://github.com/PaddlePaddle/Paddle)
+ [parl](https://github.com/PaddlePaddle/PARL)
+ gym
+ tqdm


### Start Training:
```
# To train a model for CartPole-v1 game
python train.py
```

## DQN-Variants

For DQN variants such as Double DQN and Dueling DQN, please check [here](https://github.com/PaddlePaddle/PARL/tree/develop/examples/DQN_variant)
