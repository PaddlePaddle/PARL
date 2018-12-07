## Quick Start Example
Based on PARL, train a agent to play CartPole game with policy gradient algorithm in a few minutes.

## How to use
### Dependencies:

+ python2.7 or python3.5+
+ [PARL](https://github.com/PaddlePaddle/PARL)
+ [paddlepaddle>=1.0.0](https://github.com/PaddlePaddle/Paddle)
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
# Or visualize when evaluating: python train.py --eval_vis

### Result
After training, you will see the agent get the best score (200 points).
