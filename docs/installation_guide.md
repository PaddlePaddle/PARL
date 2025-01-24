# Installation Guide

## 详细安装步骤

1. **环境准备**
    - 已测试的 Python 版本范围为：3.7 - 3.10（Linux 系统下）

2. **dl框架安装**
   - 若安装 CPU 版，正常运行以下命令即可：
     ```bash
     pip install paddlepaddle
     ```
   - 若安装 GPU 版：
     - 在 Linux 下，支持的最大版本号为 2.5：
       ```bash
       pip install paddlepaddle-gpu==2.5
       ```
     - 在 Windows 下，支持的最大版本号为 2.2.1：
       ```bash
       pip install paddlepaddle-gpu==2.2.1
       ```

3. **Numpy 版本调整**
   - 若安装paddle后，当前 numpy 版本高于 1.23.5，需要重新安装：
     ```bash
     pip install numpy==1.23.5
     ```

4. **安装 PARL 和 Gym**
   - 运行以下命令安装最新版 PARL 和 Gym：
     ```bash
     pip install parl gym
     ```

5. **测试安装**
   - 使用以下命令运行快速启动测试脚本：
     ```bash
     python examples/QuickStart/train.py
     ```

---