## Reproduce DQN with PARL
Based on PARL, we provide a simple example of the DQN algorithm.

+ Paper: DQN in [Human-level Control Through Deep Reinforcement Learning](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html)

### Result

Performance of DQN playing CartPole-v0

<p align="left">
<img src="../QuickStart/performance.gif" alt="result" height="175"/>
<img src="cartpole.jpg" alt="result" height="175"/>
</p>

## How to use
### Dependencies:
+ [paddlepaddle>=2.0.0](https://github.com/PaddlePaddle/Paddle)
+ [parl>=2.0.0](https://github.com/PaddlePaddle/PARL)
+ gym


### Start Training:
```
# To train a model for CartPole-v0 game
python train.py
```

## DQN-Variants

For DQN variants such as Double DQN and Dueling DQN, please check [here](https://github.com/PaddlePaddle/PARL/tree/develop/examples/DQN_variant)
