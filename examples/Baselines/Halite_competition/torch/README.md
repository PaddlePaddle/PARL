# 浦发赛事强化学习基线方案
基于PARL框架，我们提供了一个PPO算法的基线方案

## 目录
* config.py : 参数配置
* train.py : 训练脚本
* test.py : 测试脚本
* submission.py : 提交示例
* rl_trainer
  * model.py : 定义 actor 和 critic 网络架构
  * agent.py : 负责算法和环境交互，包括将数据提供给算法训练
  * algorithm.py : PPO算法实现
  * controller.py : 跟踪每艘飞船的状态，设计奖励并收集训练数据
  * policy.py : 定义基于规则的策略以用于控制基地
  * obs_parser.py : 设计每艘飞船的状态
  

## 基线设计
我们使用PPO算法来控制每艘飞船，其中所有飞船都共享同一个模型参数。
每艘飞船的目标是尽可能快地收集K个单位的砂金(K为超参数)，飞船采集完成后则返航到基地，将砂金放置基地后则开始新一轮的采集过程，另外当交互过程即将结束(达到最大的交互步数)，飞船也会被强制返航。换而言之，飞船采集的过程是由模型来控制，其余过程则由规则控制。对于基地，我们则使用规则来控制，基地的目标是尽可能快地生产M艘飞船(M为超参数)。


## 快速开始
创建并激活一个虚拟python环境
```shell
conda create -n halite python==3.6

source activate halite
```

安装依赖
```shell
pip install -r requirements.txt
```

## 训练
在 config.py 文件中修改超参数后并运行以下命令:
```shell
python train.py
```

## 测试
当训练完成后，在 test.py 中修改你的模型加载路径后运行脚本来测试你的模型效果。
```shell
python test.py
```

需要注意的是，此测试脚本使用了一个内置的 random agent 作为对手。如果你需要对比其他智能体的话则需要修改 “random” 为对应的智能体方法。当你提交模型和方案到平台前，也可以使用此脚本来测试你的代码中是否无误。


## 结果
以下图片展示了PPO算法的学习效果。目前，我们只在某个固定种子下训练模型，并且选取了一个随机智能体作为对手。为了在赛事中得到较好的名次，选手应该训练出一个更为鲁棒的模型(如应对不同砂金分布的环境和 1vs1, 1vs3场景)。
![learning curve](https://github.com/benchmarking-rl/PARL-experiments/blob/master/Baselines/Halite_Competition/torch/learning_curve.jpg?raw=true)

## 可视化
如果你想查看经渲染后的对战效果，首先需要激活Jupyter Notebook环境并打开test.ipynb，随后运行其中代码即可看到动画效果。
![animation](https://github.com/benchmarking-rl/PARL-experiments/blob/master/Baselines/Halite_Competition/torch/animation.gif?raw=true)

## 提交
目前选手们只能提交一个文件到平台上，因此选手需要将需要用到的函数和模型都放置到同一个文件中。为了在文件中加载模型，选手需要先将模型编码成字节串然后放到文件中，在需要加载模型的地方将字节串解码。选手可以参考 encode_model.py 查看如何编码模型，参考 submission.py 文件查看提交范例和加载模型。

需要注意的是，评分系统只会调用提交文件的最后一个函数方法。因此选手需要将智能体给出动作的函数方法放在提交文件的最后，此方法接收 observation 和 configuration作为输入，给出每艘飞船和基地的动作，具体选手可查看 submission.py 文件。
