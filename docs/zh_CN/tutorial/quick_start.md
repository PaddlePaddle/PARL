# **教程：使用PARL解决Cartpole问题**

本教程会使用 [示例](https://github.com/PaddlePaddle/PARL/tree/develop/examples/QuickStart)中的代码来解释任何通过PARL构建智能体解决经典的Cartpole问题。

本教程的目标：
- 熟悉PARL构建智能体过程中需要用到的子模块。

## CartPole 介绍

CartPole又叫倒立摆。小车上放了一根杆，杆会因重力而倒下。为了不让杆倒下，我们要通过移动小车，来保持其是直立的。如下图所示。 在每一个时间步，模型的输入是一个4维的向量,表示当前小车和杆的状态，模型输出的信号用于控制小车往左或者右移动。当杆没有倒下的时候，每个时间步，环境会给1分的奖励；当杆倒下后，环境不会给任何的奖励，游戏结束。

<p align="center">
<img src="../../../examples/QuickStart/performance.gif" width="300"/>
</p>

## Model

**Model** 主要定义前向网络，这通常是一个策略网络(Policy Network)或者一个值函数网络(Value Function)，输入是当前环境状态(State)。

首先，我们构建一个包含2个全连接层的前向网络。

```python
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import parl

class CartpoleModel(parl.Model):
    def __init__(self, obs_dim, act_dim):
        super(CartpoleModel, self).__init__()
        hid1_size = act_dim * 10
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size, act_dim)

    def forward(self, x):
        out = paddle.tanh(self.fc1(x))
        prob = F.softmax(self.fc2(out), axis=-1)
        return prob
```

定义前向网络的主要三个步骤：
- 继承`parl.Model`类
- 构造函数`__init__`中声明要用到的中间层
- 在`forward`函数中搭建网络

在上面的代码中，我们先在构造函数中声明了两个全连接层以及其激活函数，然后在`forward`函数中定义了网络的前向计算方式：输入一个状态，然后经过两层FC以及softmax，得到了每个action的概率分布预测。

## Algorithm

**Algorithm** 定义了具体的算法来更新前向网络(Model)中的参数，也就是通过定义损失函数以及使用optimizer更新Model。一个Algorithm包含至少一个Model。在这个教程中，我们将使用经典的PolicyGradient算法来解决问题。由于PARL仓库已经实现了这个算法，我们只需要直接import来使用即可。

```python
model = CartpoleModel(act_dim=2)
algorithm = parl.algorithms.PolicyGradient(model, lr=1e-3)
```
在实例化了Model之后，我们把它传给algorithm。

## Agent
**Agent** 负责算法与环境的交互，在交互过程中把生成的数据提供给Algorithm来更新模型(Model)，也就是数据和算法的交互一般定义在这里。

我们得要继承`parl.Agent`这个类来实现自己的Agent，下面先把Agent的代码抛出来，再按照函数解释：
```python
class CartpoleAgent(parl.Agent):

    def __init__(self, algorithm):
        super(CartpoleAgent, self).__init__(algorithm)

    def sample(self, obs):

        obs = paddle.to_tensor(obs, dtype='float32')
        prob = self.alg.predict(obs)
        prob = prob.numpy()
        act = np.random.choice(len(prob), 1, p=prob)[0]
        return act

    def predict(self, obs):
        obs = paddle.to_tensor(obs, dtype='float32')
        prob = self.alg.predict(obs)
        act = prob.argmax().numpy()[0]
        return act

    def learn(self, obs, act, reward):
        act = np.expand_dims(act, axis=-1)
        reward = np.expand_dims(reward, axis=-1)
        obs = paddle.to_tensor(obs, dtype='float32')
        act = paddle.to_tensor(act, dtype='int32')
        reward = paddle.to_tensor(reward, dtype='float32')

        loss = self.alg.learn(obs, act, reward)
        return loss.numpy()[0]
```
一般情况下，用户必须实现以下几个函数：
- 构造函数：
把前面定义好的algorithm传进来，作为agent的一个成员变量，用于后续的数据交互。需要注意的是，这里必须得要初始化父类：super(CartpoleAgent, self).\_\_init\_\_(algorithm)。
- predict： 根据环境状态返回预测动作（action），一般用于评估和部署agent。
- sample：根据环境状态返回动作（action），一般用于训练时候采样action进行探索。
- learn： 输入训练数据，更新智能体的相关参数。

## 开始训练
首先，我们来定一个智能体。逐步定义model|algorithm|agent，然后得到一个可以和环境进行交互的智能体。
```python
model = CartpoleModel(obs_dim=obs_dim, act_dim=act_dim)
alg = parl.algorithms.PolicyGradient(model, lr=LEARNING_RATE)
agent = CartpoleAgent(alg)

```
然后我们用这个agent和环境进行交互，训练模型，1000个episode之后，agent就可以很好地解决Cartpole问题，拿到满分（200）。
```python
def run_train_episode(agent, env):
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset()
    while True:
        obs_list.append(obs)
        action = agent.sample(obs)
        action_list.append(action)

        obs, reward, done, info = env.step(action)
        reward_list.append(reward)

        if done:
            break
    return obs_list, action_list, reward_list

env = gym.make("CartPole-v0")
for i in range(1000):
      obs_list, action_list, reward_list = run_episode(env, agent)
      if i % 10 == 0:
          logger.info("Episode {}, Reward Sum {}.".format(i, sum(reward_list)))

      batch_obs = np.array(obs_list)
      batch_action = np.array(action_list)
      batch_reward = calc_reward_to_go(reward_list)

      agent.learn(batch_obs, batch_action, batch_reward)
      if (i + 1) % 100 == 0:
          _, _, reward_list = run_episode(env, agent, train_or_test='test')
          total_reward = np.sum(reward_list)
          logger.info('Test reward: {}'.format(total_reward))
```

## 总结


<img src="../../../examples/QuickStart/performance.gif" width="300"/><img src="../../images/quickstart.png" width="300"/>

在这个教程，我们展示了如何一步步地构建强化学习智能体，用于解决经典的Cartpole问题。完整的训练代码可以在这个[文件夹](https://github.com/PaddlePaddle/PARL/tree/develop/examples/QuickStart)中找到。赶紧执行下面的代码运行尝试下吧：）

```shell
# Install dependencies
pip install paddlepaddle

pip install gym
git clone https://github.com/PaddlePaddle/PARL.git
cd PARL
pip install .

# Train model
cd examples/QuickStart/
python train.py
```
