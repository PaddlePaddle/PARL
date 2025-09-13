# waymax_RL

> **Full-GPU RL for Waymax GPU autonomous driving simulation**


## 简介
**waymax-RL** 是我们为 **Waymax**（GPU 自动驾驶仿真）开发的 **Full-GPU 强化学习训练框架**。

- 与传统RL框架相比，断档领先的FULL-GPU RL自动驾驶训练
- Colab 上直接运行
- 基于DlPack与仿真数据直接显存交互，可兼容torch、paddle、tensorflow、jax多种框架, 目前提供了基于torch的训练流程。

**快速体验**：在 Colab 上打开并运行示例：[[COLAB_LINK]](https://colab.research.google.com/drive/1l7TxIeM8Qd-THscwMoTcJS1Dfz8TeT5u?usp=sharing)

---

## 目录

- [waymax\_RL](#waymax_rl)
  - [简介](#简介)
  - [目录](#目录)
  - [什么是 Waymax？](#什么是-waymax)
  - [为什么要开源 Waymax\_RL？](#为什么要开源-waymax_rl)
  - [安装（本地）](#安装本地)

---

## 什么是 Waymax？

Waymax 是由 Waymo 与 DeepMind 在 2023 年底联合推出的、**完全运行在 GPU 上** 的自动驾驶结构化仿真。通过把仿真逻辑完全搬到 GPU 上，提升并行仿真与数据生产效率至万倍以上实时速度，为大规模 RL 训练提供了新的可能性。

继 Waymax 之后，2025 年苹果推出了 GIGAFLOW 等相关研究，也展示了在自动驾驶领域以纯模拟数据和大规模 GPU RL 训练达成 SOTA 的潜力。

---

## 为什么要开源 Waymax_RL？

- Waymax 官方开源了 GPU 仿真器本身，但 **未提供配套的 Full-GPU RL 训练框架**， 详见该issue： https://github.com/waymo-research/waymax/issues/11。
- 仅将仿真从CPU转到GPU，仍使用rllib、parl等传统cpu分布式框架，无法发挥GPU仿真的惊人效率：每步的cpu数据交换和计算则会形成新的瓶颈， 详见图1。

因此我们开源 **Waymax_RL**，提供与 GPU 仿真紧密耦合的训练框架，帮助研究者和工程师更方便地在 GPU 上高效训练自动驾驶智能体。

> PS: waymax-RL的内部版本很早就经过实车测试，证明了大规模高效RL训练的先进性


## 安装（本地）
若是本地安装，与colab运行步骤类似：
1. 创建conda环境 conda create -n waymax_rl python=3.10
2. pip install -U "jax[cuda12]"
3. git clone https://github.com/waymo-research/waymax.git
4. 安装waymax： cd waymax && git checkout 71c2be9 && pip install -e .
5. 卸载gpu版tensorflow: pip uninstall -y tensorflow
6. cd waymax_rl, pip install -r requirements.txt

```

## 快速开始

**训练示例**：

```bash
cd waymax_rl
python train.py --config-name=ppo_config
```

**评估/可视化示例**：

```bash
python eval.py --checkpoint runs/exp1/checkpoint.pt --render
```


---


如果你觉得这个研究不错，欢迎 ⭐️ 支持我们！