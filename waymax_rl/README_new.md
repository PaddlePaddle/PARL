# waymax-RL

> **End-to-End GPU Reinforcement Learning for Waymax Autonomous Driving Simulation**

---

## 简介

**waymax-RL** 是我们为 **Waymax**（GPU 自动驾驶仿真）开发的 **全 GPU 强化学习训练框架**。

- 与传统 RL 框架相比，**断档领先**的全 GPU 自动驾驶训练  
- 支持在 **Colab 上直接运行**  
- 基于 **DLPack** 与仿真数据直接显存交互，可兼容 **PyTorch、Paddle、TensorFlow、JAX** 等多种框架  
- 当前提供了 **基于 PyTorch 的训练流程**  

👉 **快速体验**：在 Colab 上打开并运行示例 [[Colab Demo]](https://colab.research.google.com/drive/1l7TxIeM8Qd-THscwMoTcJS1Dfz8TeT5u?usp=sharing)

---

## 目录

- [waymax-RL](#waymax-rl)
  - [简介](#简介)
  - [目录](#目录)
  - [什么是 Waymax？](#什么是-waymax)
  - [为什么要开源 waymax-RL？](#为什么要开源-waymax-rl)
  - [安装（本地）](#安装本地)
  - [快速开始](#快速开始)
    - [训练示例](#训练示例)
    - [评估 / 可视化示例](#评估--可视化示例)
  - [致谢 \& 支持](#致谢--支持)

---

## 什么是 Waymax？

**Waymax** 是由 Waymo 与 DeepMind 在 2023 年底联合推出的 **完全运行在 GPU 上** 的自动驾驶结构化仿真引擎。  

通过将仿真逻辑完全搬到 GPU 上，它可以将并行仿真与数据生产效率提升至 **万倍以上实时速度**，为大规模 RL 训练提供了新的可能性。  

继 Waymax 之后，2025 年苹果推出了 **GIGAFLOW** 等相关研究，也展示了在自动驾驶领域以纯模拟数据和大规模 GPU RL 训练达成 **SOTA** 的潜力。  

---

## 为什么要开源 waymax-RL？

- Waymax 官方开源了 GPU 仿真器本身，但 **未提供配套的全 GPU RL 训练框架**，参见 [issue](https://github.com/waymo-research/waymax/issues/11)  
- 仅将仿真迁移到 GPU，而继续使用 **rllib、parl 等传统 CPU 分布式框架**，会导致 CPU ↔ GPU 的数据交换成为新的瓶颈（详见图 1）  

因此我们开源 **waymax-RL**，提供与 GPU 仿真紧密耦合的 RL 训练框架，帮助研究者和工程师更方便地在 GPU 上高效训练自动驾驶智能体。  

> 💡 内部版本的 waymax-RL 曾经过实车测试，证明了大规模高效 RL 训练的先进性。  

---

## 安装（本地）

与 Colab 步骤类似，本地安装流程如下：

```bash
# 1. 创建 conda 环境
conda create -n waymax_rl python=3.10

# 2. 安装 jax (CUDA 12)
pip install -U "jax[cuda12]"

# 3. 克隆 Waymax
git clone https://github.com/waymo-research/waymax.git

# 4. 安装指定版本的 waymax
cd waymax
git checkout 71c2be9
pip install -e .

# 5. 卸载 GPU 版 tensorflow (避免冲突)
pip uninstall -y tensorflow

# 6. 安装 waymax-RL
cd ../waymax_rl
pip install -r requirements.txt
```

---

## 快速开始

### 训练示例

```bash
cd waymax_rl
python train.py --config-name=ppo_config
```

### 评估 / 可视化示例

```bash
python eval.py --checkpoint runs/exp1/checkpoint.pt --render
```

---

## 致谢 & 支持

如果你觉得这个研究有帮助，欢迎点个 ⭐️ 支持我们！  

---
