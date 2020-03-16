# DeepES工具
DeepES是一个支持**快速验证**ES效果、**兼容多个框架**的C++库。
<p align="center">
<img src="DeepES.gif" alt="PARL" width="500"/>
</p>

## 使用示范
```c++
//实例化一个预测，根据配置文件加载模型，采样方式（Gaussian\CMA sampling..)、更新方式(SGD\Adam)等
auto predictor = Predicotr(config); 

for (int i = 0; i < 100; ++i) {
   auto noisy_predictor = predictor->clone(); // copy 一份参数
   int key = noisy_predictor->add_noise(); // 参数扰动，同时保存随机种子
   int reward = evaluate(env, noisiy_predictor); //评估参数
   noisy_keys.push_back(key); // 记录随机噪声对应种子
   noisy_rewards.push_back(reward); // 记录评估结果
}
//根据评估结果、随机种子更新参数，然后重复以上过程，直到收敛。
predictor->update(noisy_keys, noisy_rewards);
```

## 一键运行demo列表
- **Torch**: sh [build.sh](./build.sh)
- **Paddle**: 
- **裸写网络**：

## 相关依赖:
- Protobuf >= 2.4.2
- glog
- gflag

## 额外依赖：

### 使用torch 
下载[libtorch](https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.4.0%2Bcpu.zip)或者编译torch源码，得到libtorch文件夹，放在当前目录中。
