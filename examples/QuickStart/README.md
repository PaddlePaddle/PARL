## Quick Start
Train an agent with PARL to solve the CartPole problem, a classical benchmark in RL.

## How to use
### Dependencies:

+ python2.7 or python3.5+
+ [paddlepaddle>=1.0.0](https://github.com/PaddlePaddle/Paddle)
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
cd examples/QuickStart/
python train.py  
```

### Expected Result
<img src="performance.gif" width = "300" height ="200" alt="result"/>

The agent can get around 200 points in a few minutes.

