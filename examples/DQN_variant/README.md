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

Performance of **Dueling DQN** on various environments:

<p align="center">
<img src=".benchmark/Dueling DQN.png" alt="result"/>
</p>

Performance of **Dueling DQN** on 55 Atari environments:

|                     |                      |                      |                    |                 |
|---------------------|----------------------|----------------------|--------------------|-----------------|
|Alien (5977)         | Amidar (364)         | Assault (9676)       |Asterix (23800)     | Asteroids (657)  |
|Atlantis (85633)     | WizardOfWor (2767)   | BankHeist (1143)     |BattleZone (37667)  | BeamRider (13570)|
|Berzerk (827)        | Bowling (47)         | Boxing (100)         |Breakout (409)      | Centipede (5103) |
|ChopperCommand (1300)| CrazyClimber (118733)| DemonAttack (167200) |DoubleDunk (-1)     | Enduro (4153)    |
|FishingDerby (-64)   | Freeway (22)         | Frostbite (5273)     |Gopher (11187)      | Gravitar (0)     | 
|Hero (14613)         | IceHockey (2)        | Jamesbond (767)      |Kangaroo (4133)     | Krull (8856)     |
|KungFuMaster (19933) | MontezumaRevenge (0) | MsPacman (4013)      |NameThisGame (10327)| Phoenix (7333)   |
|Pitfall (0)          | Pong (21)            | PrivateEye (49)      |Qbert (15275)       | Riverraid (13410)|
|RoadRunner (47167)   | Robotank (27)        | Seaquest (16573)     |Skiing (-14409)     | Solaris (53)     |
|SpaceInvaders (2797) | StarGunner (59367)   | Tennis (0)           |TimePilot (8200)    | Tutankham (235)  |
|UpNDown (18153)      | Venture (0)          | VideoPinball (745800)|YarsRevenge (34346) | Zaxxon (13233)   |

## How to use
### Dependencies:
+ [paddlepaddle>=2.0.0](https://github.com/PaddlePaddle/Paddle)
+ [parl>=2.0.2](https://github.com/PaddlePaddle/PARL)
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
