## PyTorch benchmark Quick Start
Train an agent with PARL to solve the CartPole problem, a classical benchmark in RL.

## How to use
### Dependencies:

+ [parl](https://github.com/PaddlePaddle/PARL)
+ torch
+ gym

### Start Training:
```
# Install dependencies
pip install torch torchvision gym

git clone https://github.com/PaddlePaddle/PARL.git
cd PARL
pip install .

# Train model
cd benchmark/torch/QuickStart
python train.py  
```

### Expected Result
<img src="https://github.com/PaddlePaddle/PARL/blob/develop/examples/QuickStart/performance.gif" width = "300" height ="200" alt="result"/>

The agent can get around 200 points in a few minutes.
