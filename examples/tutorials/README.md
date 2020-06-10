## 《PARL强化学习入门实践》课程示例

针对强化学习初学者，PARL提供了[入门课程](https://aistudio.baidu.com/aistudio/course/introduce/1335)，展示最基础的5个强化学习算法代码示例。

## 课程大纲
+ 一、强化学习(RL)初印象
    + RL概述、入门路线
    + 实践：环境搭建（[lesson1](lesson1/gridworld.py) 的代码提供了格子环境世界的渲染封装）
+ 二、基于表格型方法求解RL
    + MDP、状态价值、Q表格
    + 实践： [Sarsa](lesson2/sarsa)、[Q-learning](lesson2/q_learning)
+ 三、基于神经网络方法求解RL
    + 函数逼近方法
    + 实践：[DQN](lesson3/dqn)
+ 四、基于策略梯度求解RL
    + 策略近似、策略梯度
    + 实践：[Policy Gradient](lesson4/policy_gradient)
+ 五、连续动作空间上求解RL
    + 实战：[DDPG](lesson5/ddpg)



## 使用说明

### 安装依赖

+ [paddlepaddle==1.6.3](https://github.com/PaddlePaddle/Paddle)
+ [parl==1.3.1](https://github.com/PaddlePaddle/PARL)
+ gym


### 运行示例

进入每个示例对应的代码文件夹中，运行
```
python train.py
```
