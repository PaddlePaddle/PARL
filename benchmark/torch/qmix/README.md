## QMIX based on Pytorch
We reproduce the QMIX based on **PARL** and **Pytorch**, reaching the same level of indicators as the paper in StarCraftII benchmarks.
### QMIX
QMIX is a **value-based** multi-agent reinforcement learning algorithm.  
See more information about QMIX in: [QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
### StarCraftII Environment
Paper: [The StarCraft Multi-Agent Challenge](https://arxiv.org/pdf/1902.04043)  
Github Repositories: [smac](https://github.com/oxwhirl/smac)  
## Benchmark Results
<img src="images/torch-qmix-result.png" width = "1200"  alt="Performance" />  

- We traineded our model in 5 different scenarios: *"3m", "8m", "2s_3z", "3s_5z"* and *"1c_3s_5z"*.  
- The **difficulty** in all scenarios are set to be "7" (very difficult).  
- We trainined the model 3 times for each scenario.

## How to Use
### Dependencies
- python>=3.6
- [parl](https://github.com/PaddlePaddle/PARL)
- [smac](https://github.com/oxwhirl/smac)
- Pytorch>=1.6.0

### Start Training
1. Modify the config in `qmix_config.py`.
2. Start training:
    ```bash
    python train.py
    ```
3. View the training process with tensorboard:
    ```bash
    tensorboard --logdir ./
    ```
