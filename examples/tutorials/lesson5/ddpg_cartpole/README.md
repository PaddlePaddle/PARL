## Reproduce DDPG with PARL
Based on PARL, we provide a simple demonstration of DDPG.

+ DDPG in
[Continuous Control with Deep Reinforcement Learning](https://arxiv.org/pdf/1509.02971.pdf)

### Result

Performance of DDPG playing continuous CartPole-v0

<p align="left">
<img src="result.jpg" alt="result" height="200"/>
</p>

## How to use
### Dependencies:
+ [paddlepaddle>=1.6.1](https://github.com/PaddlePaddle/Paddle)
+ [parl](https://github.com/PaddlePaddle/PARL)
+ gym


### Start Training:
```
# To train a model for continuous CartPole-v0 game
python train.py
```
