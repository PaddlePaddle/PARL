## Introduction
Based on PARL, we provide the implementation of decision transformer, with the same performance as reported in the original paper.

> Paper: [Decision Transformer: Reinforcement
Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345)

### Dataset for RL
Follow the installation instruction in [D4RL](https://github.com/Farama-Foundation/D4RL) to install D4RL.
Then run the scripts in `data` directory to download dataset for traininig.
```shell
python download_d4rl_datasets.py
```


### Benchmark result
#### 1. Mujoco results
<p align="center">
<img src="https://github.com/benchmarking-rl/PARL-experiments/blob/master/DT/torch/mujoco_result.png" alt="mujoco-result"/>
</p>

+ Each experiment was run three times with different random seeds

## How to use
### Dependencies:
+ [D4RL](//github.com/Farama-Foundation/D4RL)
+ [parl>=2.1.1](https://github.com/PaddlePaddle/PARL)
+ pytorch
+ gym==0.18.3
+ mujoco-py==2.0.2.13
+ transformers==4.5.1


### Training:

```shell
# To train an agent for `hooper` environment with `medium` dataset
python train.py --env hopper --dataset medium

# To train an agent for `hooper` environment with `expert` dataset
python train.py --env hopper --dataset expert
```


### Reference

https://github.com/kzl/decision-transformer
