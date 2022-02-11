<p align="center">
<img src=".github/PARL-logo.png" alt="PARL" width="500"/>
</p>

[English](./README.md) | 简体中文

[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://parl.readthedocs.io/en/latest/index.html) [![Documentation Status](https://img.shields.io/badge/中文文档-最新-brightgreen.svg)](https://parl.readthedocs.io/zh_CN/latest/) [![Documentation Status](https://img.shields.io/badge/手册-中文-brightgreen.svg)](./docs/zh_CN/Overview.md) [![Release](https://img.shields.io/badge/release-v2.0.3-blue.svg)](https://github.com/PaddlePaddle/PARL/releases)


> PARL 是一个高性能、灵活的强化学习框架。

<!-- toc -->

- [概览](#概览)
  - [特点](#特点)
  - [框架结构](#框架结构)
  - [简易高效的并行接口](#简易高效的并行接口)
- [安装](#安装)
- [快速开始](#快速开始)
- [算法示例](#算法示例)

# 概览
## 特点
**可复现性保证**。我们提供了高质量的主流强化学习算法实现，严格地复现了论文对应的指标。

**大规模并行支持**。框架最高可支持上万个CPU的同时并发计算，并且支持多GPU强化学习模型的训练。

**可复用性强**。用户无需自己重新实现算法，通过复用框架提供的算法可以轻松地把经典强化学习算法应用到具体的场景中。

**良好扩展性**。当用户想调研新的算法时，可以通过继承我们提供的基类可以快速实现自己的强化学习算法。


## 框架结构
<img src=".github/abstractions.png" alt="abstractions" width="400"/>  
PARL的目标是构建一个可以完成复杂任务的智能体。以下是用户在逐步构建一个智能体的过程中需要了解到的结构：

### Model
`Model` 用来定义前向 (`Forward`)网络，这通常是一个策略网络 (`Policy Network`)或者一个值函数网络 (`Value Function`)，输入是当前环境状态 (`State`)。

### Algorithm
`Algorithm` 定义了具体的算法来更新前向网络 (`Model`)，也就是通过定义损失函数来更新`Model`。一个`Algorithm`包含至少一个`Model`。

### Agent
`Agent` 负责算法与环境的交互，在交互过程中把生成的数据提供给`Algorithm`来更新模型 (`Model`)，数据的预处理流程也一般定义在这里。

提示： 请访问[教程](https://parl.readthedocs.io/zh_CN/latest/tutorial/getting_started.html) and [API 文档](https://parl.readthedocs.io/zh_CN/latest/apis/model.html)以获取更多关于基础类的信息。

## 简易高效的并行接口
在PARL中，一个**修饰符**(parl.remote_class)就可以帮助用户实现自己的并行算法。
以下我们通过`Hello World`的例子来说明如何简单地通过PARL来调度外部的计算资源实现并行计算。 请访问我们的[教程文档](https://parl.readthedocs.io/zh_CN/latest/parallel_training/setup.html)以获取更多的并行训练信息。
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
ans = agent.sum(1,5) # run remotely and not comsume any local computation resources 
```
两步调度外部的计算资源：
1. 使用`parl.remote_class`修饰一个类，之后这个类就被转化为可以运行在其他CPU或者机器上的类。
2. 调用`parl.connect`函数来初始化并行通讯，通过这种方式获取到的实例和原来的类是有同样的函数的。由于这些类是在别的计算资源上运行的，执行这些函数**不再消耗当前线程计算资源**。

<img src=".github/decorator.png" alt="PARL" width="450"/>

如上图所示，真实的actor（橙色圆圈）运行在CPU集群，learner（蓝色圆圈）和remote actor（黄色圆圈）运行在本地的GPU上。对于用户而言，完全可以像写多线程代码一样来实现并行算法，相当简单，但是这些多线程的运算利用了外部的计算资源。我们也提供了并行算法示例，更多细节请参考[IMPALA](benchmark/fluid/IMPALA/), [A2C](examples/A2C/)。


# 安装:
### 依赖
- Python 3.6+. (Python 3.8+ 更适合用于并行训练)
- [paddlepaddle>=2.0](https://github.com/PaddlePaddle/Paddle) (**非必须的**，如果你只用并行部分的接口不需要安装paddle) 


```
pip install parl
```

# 快速开始
请查看一下几个教程帮助您快速上手PARL:
- [教程](https://parl.readthedocs.io/zh_CN/latest/tutorial/getting_started.html) : 解决经典的 CartPole 问题。
- [Xparl用法](https://parl.readthedocs.io/zh_CN/latest/parallel_training/setup.html) : 如何使用`xparl`设置集群，实现并行运算。
- [进阶教程](https://parl.readthedocs.io/zh_CN/latest/implementations/new_alg.html) : 自定义新算法。
- [API 文档](https://parl.readthedocs.io/zh_CN/latest/apis/model.html)

同时，我们还为零基础开发者提供强化学习入门课程 : ( [视频](https://www.bilibili.com/video/BV1yv411i7xd) | [代码](examples/tutorials/) )

# 算法示例
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
- [冠军解决方案：NIPS2018强化学习假肢挑战赛](examples/NeurIPS2018-AI-for-Prosthetics-Challenge/)
- [冠军解决方案：NIPS2019强化学习仿生人控制赛事](examples/NeurIPS2019-Learn-to-Move-Challenge/)
- [冠军解决方案：NIPS2020强化学习电网调度赛事](examples/NeurIPS2020-Learning-to-Run-a-Power-Network-Challenge/)

<img src="examples/NeurIPS2019-Learn-to-Move-Challenge/image/performance.gif" width = "300" height ="200" alt="NeurlIPS2018"/> <img src=".github/Half-Cheetah.gif" width = "300" height ="200" alt="Half-Cheetah"/> <img src=".github/Breakout.gif" width = "200" height ="200" alt="Breakout"/> 
<br>
<img src=".github/Aircraft.gif"  width = "808" height ="300"  alt="NeurlIPS2018"/>
