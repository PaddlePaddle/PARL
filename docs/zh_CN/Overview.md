<p align="center">
<img src="../../.github/PARL-logo.png" width="500"/>
<img src="../images/bar.png"/>
</p>

<br>**PARL**是一个主打高性能、稳定复现、轻量级的强化学习框架。<br>


## 使用场景
- 想要在**实际任务中**尝试使用强化学习解决问题
- 想快速调研下**不同强化学习算法**在同一个问题上的效果
- 强化学习算法训练速度太慢，想搭建**分布式**强化学习训练平台
- python的GIL全局锁限制了多线程加速，想**加速python代码**


## PARL文档全览
<table>
  <tbody>
    <tr align="center" valign="bottom">
    <td>
      </td>
      <td>
        <b>构建智能体（基础）</b>
        <img src="../images/bar.png"/>
      </td>
      <td>
        <b>开源算法库</b>
        <img src="../images/bar.png"/>
      </td>
      <td>
        <b>并行训练（进阶）</b>
        <img src="../images/bar.png"/>
      </td>
    </tr>
    </tr>
    <tr valign="top">
    <td align="center" valign="middle">
      </td>
      <td>
        <ul>
        <li><b>教程</b></li>
           <ul>
          <li><a href="tutorial/quick_start.md">入门：解决cartpole问题</a></li>
          <li><a href="tutorial/module.md">子模块说明</a></li>
          <li><a href="tutorial/param.md">参数保存与加载</a></li>
          <li><a href="tutorial/summary.md">绘制训练曲线</a></li>
          <li><a href="tutorial/csv_logger.md">表格输出实验数据</a></li>
           </ul>
        </ul>
      </td>
      <td align="left" >
        <ul>
          <li><b>前沿算法</b></li>
            <ul>
              <li><a href="https://github.com/PaddlePaddle/PARL/tree/develop/examples/SAC">SAC</a></li>
              <li><a href="https://github.com/PaddlePaddle/PARL/tree/develop/examples/TD3">TD3</a></li>
              <li><a href="https://github.com/PaddlePaddle/PARL/tree/develop/examples/MADDPG">MADDPG</a></li>
            </ul>
          <li><b>经典算法</b></li>
            <ul>
              <li><a href="https://github.com/PaddlePaddle/PARL/tree/develop/examples/QuickStart">PolicyGradient</a></li>
              <li><a href="https://github.com/PaddlePaddle/PARL/tree/develop/examples/DQN">DQN</a></li>
            <li><a href="https://github.com/PaddlePaddle/PARL/tree/develop/examples/DDPG">DDPG</a></li>
            <li><a href="https://github.com/PaddlePaddle/PARL/tree/develop/examples/PPO">PPO</a></li>
            </ul>
          <li><b>并行算法</b></li>
            <ul>
              <li><a href="https://github.com/PaddlePaddle/PARL/tree/develop/examples/A2C">A2C</a></li>
              <li><a href="https://github.com/PaddlePaddle/PARL/tree/develop/examples/ES">ES</a></li>
            </ul>
        </ul>
      </td>
      <td>
      <ul>
        <li><b>教程</b></li>
            <ul><li><a href="xparl/introduction.md">XPARL并行介绍</a></li>
            <li><a href="xparl/tutorial.md">使用教程</a></li>
            <li><a href="xparl/example.md">加速案例</a></li>
            <li><a href="xparl/debug.md">如何debug</a></li>
            <li><a href="xparl/distribute_files.md">分发本地文件</a></li>
            <li><a href="xparl/serialize.md">序列化加速（非必须）</a></li>
            </ul>
      </td>
    </tr>
  </tbody>
  
</table>

## **安装**

### **安装**
PARL 支持并在 Ubuntu >= 16.04, macOS >= 10.14.1, 和 Windows 10通过了测试。 目前在Windows上**仅支持**python3.5+以上的版本，要求是64位的python。

```shell
pip install parl --upgrade
```
如果遇到网络问题导致的下载较慢，建议使用清华源解决:
```shell
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple parl --upgrade
```

如果想试试最新代码，可从源代码安装。
```shell
git clone https://github.com/PaddlePaddle/PARL
cd PARL
pip install .
```
如果遇到网络问题导致的下载较慢，建议使用清华源解决（参考上面的命令）。<br>
遇到git clone如果较慢的问题，建议使用我们托管在国内码云平台的仓库。
```shell
git clone https://gitee.com/paddlepaddle/PARL.git
```

### **关于并行**

如果只是想使用PARL的并行功能的话，是无需安装任何深度学习框架的。


## 贡献
本项目欢迎任何贡献和建议。 大多数贡献都需要你同意参与者许可协议（CLA），来声明你有权，并实际上授予我们有权使用你的贡献。
### 代码贡献规范
- 代码风格规范<br>
PARL使用yapf工具进行代码风格的统一，使用方法如下：
```shell
pip install yapf==0.24.0
yapf -i modified_file.py
```
- 持续集成测试<br>
当增加代码时候，需要增加测试代码覆盖所添加的代码，测试代码得放在相关代码文件的`tests`文件夹下，以`_test.py`结尾（这样持续集成测试会自动拉取代码跑）。附：[测试代码示例](../../parl/tests/import_test.py)
- 本地运行单元测试（非必要）<br>
如果你希望在自己的机器运行单测代码，可先在本地机器上安装Docker，再按以下步骤执行单测任务。
```
cd PARL
docker build -t parl/parl-test:unittest  .teamcity/
nvidia-docker run -i --rm -v $PWD:/work -w /work parl/parl-test:unittest .teamcity/build.sh test
```

## 反馈
- 在 GitHub 上[提交问题](https://github.com/PaddlePaddle/PARL/issues)
