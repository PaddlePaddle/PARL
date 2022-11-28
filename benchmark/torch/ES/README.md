## Reproduce ES with PARL
Based on PARL, we have implemented the Evolution Strategies (ES) algorithm and evaluate it in Mujoco environments. Its performance reaches the same level of indicators as the paper.

+ ES in
[Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864)

### Mujoco games introduction
Please see [here](https://github.com/openai/mujoco-py) to know more about Mujoco games.

### Benchmark result

<p align="center">
<img src="https://github.com/benchmarking-rl/PARL-experiments/blob/master/ES/torch/result.png" alt="result"/>
</p>

## How to use
### Dependencies
+ python3.7+
+ torch==1.8.1
+ [parl>=2.1.1](https://github.com/PaddlePaddle/PARL)
+ gym>=0.26.0
+ mujoco>=2.2.2


### Distributed training

To replicate the performance reported above, we encourage you to train with 24 or 48 CPUs.  
If you haven't created a cluster before, enter the following command to create a cluster. For more information about the cluster, please refer to our [documentation](https://parl.readthedocs.io/en/latest/parallel_training/setup.html).

```bash
xparl start --port 8010 --cpu_num 24
```

Then we can start the distributed training by running:


```bash
python train.py
```

Training result will be saved in `train_log` with the training curve.

### Reference
+ [Ray](https://github.com/ray-project/ray)
+ [evolution-strategies-starter](https://github.com/openai/evolution-strategies-starter)
