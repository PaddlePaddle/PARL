## Dygraph Quick Start
Train an agent with PARL to solve the CartPole problem, a classical benchmark in RL. Dygraph version of [QuickStart][origin]

## How to use
### Dependencies:

+ [paddlepaddle>=1.8.5](https://github.com/PaddlePaddle/Paddle)
+ [parl](https://github.com/PaddlePaddle/PARL)
+ gym

### Start Training:
```
# Install dependencies
pip install paddlepaddle  
# Or use Cuda: pip install paddlepaddle-gpu

pip install gym
git clone https://github.com/PaddlePaddle/PARL.git
cd PARL
pip install .

# Train model
cd examples/EagerMode/QuickStart/
python train.py  
```

### Expected Result
<img src="https://github.com/PaddlePaddle/PARL/blob/develop/examples/QuickStart/performance.gif" width = "300" height ="200" alt="result"/>

The agent can get around 200 points in a few minutes.

[origin]: https://github.com/PaddlePaddle/PARL/tree/develop/examples/QuickStart
