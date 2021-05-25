# **教程：子模块说明**
<p align="center">
<img src="../../../.github/abstractions.png" width="300"/>
</p>
在上一个教程中，我们快速地展示了如果通过PARL的三个基础模块：Model Algorithm, Agent 来搭建智能体和环境进行交互的。在这个教程中，我们将详细介绍每个模块的具体定位，以及使用规范。


## Model
- 定义：`Model` 用来定义前向(Forward)网络，这通常是一个策略网络(Policy Network)或者一个值函数网络(Value Function)，输入是当前环境状态(State)。
- **⚠️注意事项**：用户得要继承`parl.Model`这个类来构建自己的Model。
- 需要实现的函数：
    - forward: 根据在初始化函数中声明的计算层来搭建前向网络。
- 备注：在PARL中，实现强化学习常用的target很方便的，直接通过deepcopy即可。
- 示例：
```python
import parl
from parl import layers

class CartpoleModel(parl.Model):
    def __init__(self):
        self.fc1 = layers.fc(size=10, act='tanh')
        self.fc2 = layers.fc(size=2, act='softmax')

    def forward(self, obs):
        out = self.fc1(obs)
        out = self.fc2(out)
        return out

if __name__ == '__main__:
    model = CartpoleModel()
    target_model = deepcopy.copy(model)
```


## Algorithm
- 定义：`Algorithm` 定义了具体的算法来更新前向网络(Model)，也就是通过定义损失函数来更新Model。一个Algorithm包含至少一个Model。
- **⚠️注意事项**：一般不自己开发，推荐直接import 仓库中已经实现好的算法。
- 需要实现的函数（`开发新算法才需要`）：
    - learn: 根据训练数据（观测量和输入的reward），定义损失函数，用于更新Model中的参数。
    - predict: 根据当前的观测量，给出动作概率或者Q函数的预估值。
- 示例：
```python
model = CartpoleModel()
alg = parl.algorithms.PolicyGradient(model, lr=1e-3)
```


## Agent
- 定义：`Agent` 负责算法与环境的交互，在交互过程中把生成的数据提供给Algorithm来更新模型(Model)，数据的预处理流程也一般定义在这里。
- **⚠️注意事项**：需要继承`parl.Agent`来使用，要在构造函数中调用父类的构造函数。
- 需要实现的函数：
    - build_program: 定义paddle的program。用户通常在这里定义两个program：一个用于训练（在learn函数中调用），一个用于预测（用于sample、predict函数中）注意⚠️：这个函数会自动被调用，用户无需关注。
    - learn: 根据输入的训练数据，更新模型参数。
    - predict: 根据输入的观测量，返回要执行的动作（action）
    - sample: 根据输入的观测量，返回要执行的动作，这个一般是添加了噪声的，用于探索用的。
- 示例：
```python
class CartpoleAgent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(CartpoleAgent, self).__init__(algorithm)

    def build_program(self):
        # 这里我们定义了两个program。这个函数会被PARL自动调用，用户只需在这里定义program就好。
        self.pred_program = fluid.Program()
        self.train_program = fluid.Program()

        # 
        with fluid.program_guard(self.pred_program):    
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.act_prob = self.alg.predict(obs)

        with fluid.program_guard(self.train_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            act = layers.data(name='act', shape=[1], dtype='int64')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            self.cost = self.alg.learn(obs, act, reward)

    def learn(self, obs, act, reward):
        # 在这里对数据进行预处理，同时准备好数据feed给上面定义的program
        act = np.expand_dims(act, axis=-1)
        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('int64'),
            'reward': reward.astype('float32')
        }
        cost = self.fluid_executor.run(
            self.train_program, feed=feed, fetch_list=[self.cost])[0]
        return cost

    def predict(self, obs):
        # 在这里对数据进行预处理，同时准备好数据feed给上面定义的program
        obs = np.expand_dims(obs, axis=0)
        act_prob = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.act_prob])[0]
        act_prob = np.squeeze(act_prob, axis=0)
        act = np.argmax(act_prob)
        return act

    def sample(self, obs):
        # 输入观测量，返回用于探索的action
        obs = np.expand_dims(obs, axis=0)
        act_prob = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.act_prob])[0]
        # 按照概率采样，输出探索性的动作
        act_prob = np.squeeze(act_prob, axis=0)
        act = np.random.choice(range(self.act_dim), p=act_prob)
        return act

if __name__ == '__main__':
    model = CartpoleModel()
    alg = parl.algorithms.PolicyGradient(model, lr=1e-3)
    agent = CartpoleAgent(alg, obs_dim=4, act_dim=2)
```
