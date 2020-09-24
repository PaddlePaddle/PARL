# EvoKit
EvoKit 是一个集合了多种进化算法、兼容多种类预测框架的进化算法库，主打快速上线验证 。
<p align="center">
<img src="DeepES.gif" alt="PARL" width="500"/>
</p>

## 使用示范
```c++
//实例化一个预测，根据配置文件加载模型，采样方式（Gaussian\CMA sampling..)、更新方式(SGD\Adam)等
auto agent = ESAgent(config); 

for (int i = 0; i < 10; ++i) {
   auto sampling_agnet = agent->clone(); // clone出一个sampling agent
   SamplingInfo info;
   sampling_agent->add_noise(info); // 参数扰动，同时保存随机种子到info中
   int reward = evaluate(env, sampling_agent); //评估参数
   noisy_info.push_back(info); // 记录随机噪声对应种子
   noisy_rewards.push_back(reward); // 记录评估结果
}
//根据评估结果、随机种子更新参数，然后重复以上过程，直到收敛。
agent->update(noisy_info, noisy_rewards);
```

## 一键运行demo列表
- sh ./scripts/build.sh

## 相关依赖:
- Protobuf2
- OpenMP
- [glog](https://github.com/gflags/gflags/blob/master/INSTALL.md)
- [gflag](https://github.com/google/glog)

## 额外依赖：

### 使用PaddleLite
下载PaddleLite的X86预编译库，或者编译PaddleLite源码，得到inference_lite_lib文件夹，放在当前目录中。(可参考：[PaddleLite使用X86预测部署](https://paddle-lite.readthedocs.io/zh/latest/demo_guides/x86.html))

### 使用torch 
下载[libtorch](https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.4.0%2Bcpu.zip)或者编译torch源码，得到libtorch文件夹，放在当前目录中。
