<p align="center">
<img src=".github/PARL-logo.png" alt="PARL" width="500"/>
</p>

# Features
**Reproducible**. We provide algorithms that stably reproduce the result of many influential reinforcement learning algorithms

**Large Scale**. Ability to support high performance parallelization of training with thousands of CPUs and multi-GPUs 

**Reusable**.  Algorithms provided in repository could be directly adapted to a new task by defining a forward network and training mechanism will be built automatically.

**Extensible**. Build new algorithms quickly by inheriting the abstract class in the framework.


# Abstractions
<img src=".github/abstractions.png" alt="abstractions" width="400"/>  
PARL aims to build an agent for training algorithms to perform complex tasks.   
The main abstractions introduced by PARL that are used to build an agent recursively are the following:

### Model
`Model` is abstracted to construct the forward network which defines a policy network or critic network given state as input.

### Algorithm
`Algorithm` describes the mechanism to update parameters in `Model` and often contains at least one model.

### Agent
`Agent` is a data bridge between environment and algorithm. It is responsible for data I/O with outside and describes data preprocessing before feeding into the training process.

Here is an example of building an agent with DQN algorithm for atari games.
```python
import parl
from parl.algorithms import DQN, DDQN

class CriticModel(parl.Model):
""" define specific forward model for environment ..."""
# three steps to build an agent
#   1.  define a forward model which is critic_model is this example
#   2.  a. to build a DQN algorithm, just pass the critic_model to `DQN`
#       b. to build a DDQN algorithm, just replace DQN in following line with DDQN
#   3.  define the I/O part in your AtariAgent so that it could update the algorithm based on the interactive data 
# a. to build a DQN algorithm, just pass the critic_model to `DQN`

critic_model = CriticModel(act_dim=2)
algorithm = DQN(critic_model)
agent = AtariAgent(aglrotihm)
```

# Install:
### Dependencies
- Python 2.7 or 3.3+. 
- PaddlePaddle >=1.0 (We try to make our repository always compatible with newest version PaddlePaddle)  


```
pip install --upgrade git+https://github.com/PaddlePaddle/PARL.git
```

# Examples

- DQN 
- DDPG
- PPO
- Winning Solution for NIPS2018: AI for Prosthetics Challenge
