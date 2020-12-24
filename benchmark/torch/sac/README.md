## Reproduce SAC with PARL
Based on PARL, the SAC algorithm of deep reinforcement learning has been reproduced, reaching the same level of indicators as the paper in Mujoco benchmarks.

> Paper: SAC in [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)

### Mujoco games introduction
Please see [here](https://github.com/openai/mujoco-py) to know more about Mujoco games.

### Benchmark result

<img src=".benchmark/SAC_results.png" width = "800" height ="400" alt="SAC_HalfCheetah-v1"/>

+ We trained our model in a Mujoco environment: "HalfCheetah-v1".
+ We trained the model 3 times(for 3 different seeds).

## How to use
### Dependencies:
+ python3.5+
+ [parl](https://github.com/PaddlePaddle/PARL)
+ gym
+ torch
+ mujoco-py>=1.50.1.0

### Start Training:
#### Arguments
```arguments
--env, default="HalfCheetah-v1"   # Mujoco gym environment name
--seed, default=0                 # Sets Gym, PyTorch and Numpy seeds
--start_timesteps, default=1e4    # Time steps initial random policy is used
--eval_freq, default=5e3          # How often (time steps) we evaluate
--eval_episodes, default=5        # How many episodes during evaluation
--max_timesteps, default=1e6      # Max time steps to run environment
--alpha, default=0.2              # Temperature parameter, determines the relative importance of entropy term against the reward
--batch_size, default=256         # Batch size for learning
--discount, default=0.99          # Discount factor
--tau, default=5e-3               # Target network update rate
--actor_lr, default=3e-4          # Learning rate of actor network
--critic_lr, default=3e-4         # Learning rate of critic network
--policy_freq, default=1          # Frequency to train actor and update params
--automatic_entropy_tuning, default=False  # Automatically adjust α
--entropy_lr, default=3e-4        # Learning rate of entropy
```
#### Train
```
# To train an agent for HalfCheetah-v1 game
python train.py

# To train for other game & for automatic entropy tuning
# python train.py --env [ENV_NAME] ---automatic_entropy_tuning True
