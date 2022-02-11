<p align="center">
<img src=".github/PARL-logo.png" alt="PARL" width="500"/>
</p>

English | [简体中文](./README.cn.md)

[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://parl.readthedocs.io/en/latest/index.html) [![Documentation Status](https://img.shields.io/badge/中文文档-最新-brightgreen.svg)](https://parl.readthedocs.io/zh_CN/latest/) [![Documentation Status](https://img.shields.io/badge/手册-中文-brightgreen.svg)](./docs/zh_CN/Overview.md) [![Release](https://img.shields.io/badge/release-v2.0.3-blue.svg)](https://github.com/PaddlePaddle/PARL/releases)

> PARL is a flexible and high-efficient reinforcement learning framework.

<!-- toc -->

- [About PARL](#about-parl)
  - [Features](#features)
  - [Abstractions](#abstractions)
  - [Parallelization](#parallelization)
- [Install](#install)
- [Getting Started](#getting-started)
- [Examples](#examples)

# About PARL
## Features
**Reproducible**. We provide algorithms that stably reproduce the result of many influential reinforcement learning algorithms.

**Large Scale**. Ability to support high-performance parallelization of training with thousands of CPUs and multi-GPUs.

**Reusable**.  Algorithms provided in the repository could be directly adapted to a new task by defining a forward network and training mechanism will be built automatically.

**Extensible**. Build new algorithms quickly by inheriting the abstract class in the framework.


## Abstractions
<img src=".github/abstractions.png" alt="abstractions" width="400"/>
PARL aims to build an agent for training algorithms to perform complex tasks.   
The main abstractions introduced by PARL that are used to build an agent recursively are the following:

### Model
`Model` is abstracted to construct the forward network which defines a policy network or critic network given state as input.

### Algorithm
`Algorithm` describes the mechanism to update parameters in `Model` and often contains at least one model.

### Agent
`Agent`, a data bridge between the environment and the algorithm, is responsible for data I/O with the outside environment and describes data preprocessing before feeding data into the training process.  

Note: For more information about base classes, please visit our [tutorial](https://parl.readthedocs.io/en/latest/tutorial/getting_started.html) and [API documentation](https://parl.readthedocs.io/en/latest/apis/model.html).

## Parallelization
PARL provides a compact API for distributed training, allowing users to transfer the code into a parallelized version by simply adding a decorator. For more information about our APIs for parallel training, please visit our [documentation](https://parl.readthedocs.io/en/latest/parallel_training/setup.html).  
Here is a `Hello World` example to demonstrate how easy it is to leverage outer computation resources.
```python
#============Agent.py=================
@parl.remote_class
class Agent(object):

    def say_hello(self):
        print("Hello World!")

    def sum(self, a, b):
        return a+b

parl.connect('localhost:8037')
agent = Agent()
agent.say_hello()
ans = agent.sum(1,5) # it runs remotely, without consuming any local computation resources
```
Two steps to use outer computation resources:
1. use the `parl.remote_class` to decorate a class at first, after which it is transferred to be a new class that can run in other CPUs or machines.
2. call `parl.connect` to initialize parallel communication before creating an object. Calling any function of the objects **does not** consume local computation resources since they are executed elsewhere.

<img src=".github/decorator.png" alt="PARL" width="450"/>
As shown in the above figure, real actors (orange circle) are running at the cpu cluster, while the learner (blue circle) is running at the local gpu with several remote actors (yellow circle with dotted edge).  

For users, they can write code in a simple way, just like writing multi-thread code, but with actors consuming remote resources. We have also provided examples of parallized algorithms like [IMPALA](benchmark/fluid/IMPALA/), [A2C](examples/A2C/). For more details in usage please refer to these examples.  


# Install:
### Dependencies
- Python 3.6+(Python 3.8+ is preferable for distributed training). 
- [paddlepaddle>=2.0](https://github.com/PaddlePaddle/Paddle) (**Optional**, if you only want to use APIs related to parallelization alone)  


```
pip install parl
```

# Getting Started
Several-points to get you started:
- [Tutorial](https://parl.readthedocs.io/en/latest/tutorial/getting_started.html) : How to solve cartpole problem.
- [Xparl Usage](https://parl.readthedocs.io/en/latest/parallel_training/setup.html) : How to set up a cluster with `xparl` and compute in parallel.
- [Advanced Tutorial](https://parl.readthedocs.io/en/latest/implementations/new_alg.html) : Create customized algorithms.
- [API documentation](https://parl.readthedocs.io/en/latest/apis/model.html)

For absolute beginners, we also provide an introductory course on reinforcement learning (RL) : ( [Video](https://www.bilibili.com/video/BV1yv411i7xd) | [Code](examples/tutorials/) )

# Examples
- [QuickStart](examples/QuickStart/)
- [DQN](examples/DQN/)
- [ES](examples/ES/)
- [DDPG](examples/DDPG/)
- [A2C](examples/A2C/)
- [TD3](examples/TD3/)
- [SAC](examples/SAC/)
- [QMIX](examples/QMIX/)
- [MADDPG](examples/MADDPG/)
- [PPO](examples/PPO/)
- [CQL](examples/CQL/)
- [Winning Solution for NIPS2018: AI for Prosthetics Challenge](examples/NeurIPS2018-AI-for-Prosthetics-Challenge/)
- [Winning Solution for NIPS2019: Learn to Move Challenge](examples/NeurIPS2019-Learn-to-Move-Challenge/)
- [Winning Solution for NIPS2020: Learning to Run a Power Network Challenge](examples/NeurIPS2020-Learning-to-Run-a-Power-Network-Challenge/)

<img src="examples/NeurIPS2019-Learn-to-Move-Challenge/image/performance.gif" width = "280" height ="200" alt="NeurlIPS2018"/> <img src=".github/Half-Cheetah.gif" width = "280" height ="200" alt="Half-Cheetah"/> <img src=".github/Breakout.gif" width = "195" height ="200" alt="Breakout"/>
<br>
<img src=".github/Aircraft.gif"  width = "762" height ="300"  alt="NeurlIPS2018"/>
