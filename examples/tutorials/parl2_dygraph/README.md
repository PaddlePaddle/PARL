## 《PARL强化学习入门实践》课程示例（动态图版本）
+ 应广大学员的要求，我们提供了课程配套代码的（lesson3-lesson5）的**动态图框架**版本， lesson1-lesson2不涉及神经网络，可沿用上级目录中的代码。

## 代码大纲
+ `lesson3`：基于神经网络方法求解RL
    + dqn：使用 DQN 算法解决 CartPole 问题。
    + homework：使用 DQN 算法解决 MountainCar 问题。
+ `lesson4`：基于策略梯度求解RL
    + policy_gradient：使用 PG 算法解决 CartPole 问题。
    + homework： 使用 PG 算法解决 Atari 游戏里的 Pong 环境。
+ `lesson5`：连续动作空间上求解RL
    + ddpg：使用 DDPG 算法解决连续动作版本的 CartPole 问题。
    + homework：使用 DDPG 算法解决四轴飞行器的悬停问题。


## 使用说明

### 安装依赖（注意：请务必安装对应的版本）

+ Python 3.6/3.7/3.8
+ [paddlepaddle](https://github.com/PaddlePaddle/Paddle)==2.2.0
+ [parl](https://github.com/PaddlePaddle/PARL)==2.0.3
+ gym==0.18.0
+ atari-py==0.2.6 (仅 lesson4 的 homework 需要安装)
+ rlschool==0.3.1 (仅 lesson5 的 homework 需要安装)

可以直接安装本目录下的 `requirements.txt` 来完成以上依赖版本的适配。
```
pip install -r requirements.txt
```

### 运行示例

进入每个示例对应的代码文件夹中，运行
```
python train.py
```
