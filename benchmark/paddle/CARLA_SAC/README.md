## SAC in Carla simulator
Based on [parl](https://github.com/PaddlePaddle/PARL) and [PaddlePaddle](https://github.com/PaddlePaddle/Paddle), 
the SAC algorithm of deep reinforcement learning has been used in Carla simulator environment.
> Paper: SAC in [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)

### Carla simulator introduction
Please see [Carla simulator](https://github.com/carla-simulator/carla/releases/tag/0.9.6) to know more about Carla simulator.

### Benchmark result
<img src=".benchmark/carla_sac.png" width = "800" height ="400" alt="carla_sac"/>
<img src=".benchmark/Lane_bend.gif" width = "300" height ="200" alt="result"/>

+ Result was run with seed `0`, mode `Lane`

## How to use
### Dependencies:
+ System: Ubuntu 16.04
+ [parl>=1.4.2](https://github.com/PaddlePaddle/PARL)
+ [paddlepaddle>=2.0.0](https://github.com/PaddlePaddle/Paddle)
+ [CARLA_0.9.6](https://github.com/carla-simulator/carla/releases/tag/0.9.6)
  ```CARLA
  Download CARLA_0.9.6, extract it to some folder, 
  and add CARLA to `PYTHONPATH` environment variable
  
  # add python path
  export PYTHONPATH="SOMEFOLDER/CARLA_0.9.6/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg:$PYTHONPATH"
  ```
+ [gym_carla](https://github.com/ShuaibinLi/gym_carla.git)
  ```gym_carla
  $ git clone https://github.com/ShuaibinLi/gym_carla.git
  $ cd gym_carla
  $ pip install -r requirements.txt
  $ pip install -e .
  ```

### Start Training
1. Enter the CARLA root folder, launch the CARLA server in different terminals 
   with non-display mode
    ```start env
    $ DISPLAY= ./CarlaUE4.sh -opengl -carla-port=2021
    ```
   or with display mode
   ```start_env
   $ ./CarlaUE4.sh -windowed -carla-port=2021
   ```
   + Start three environments (ports: 2021,2023,2025) for collecting data and training, 
     one environment (2027) for evaluating.
   
2. Open a new terminal and start [parl](https://github.com/PaddlePaddle/PARL) port for parallelization by
   ```Parallelization
   $ xparl start --port 8765
   ```
   checkout xparl connect --address in the terminal

3. Enter the cloned repository
   ```train
   $ python train.py ----localhost [xparl address]
   
   # Train for other settings
   $ python train.py ----localhost [xparl address] --seed [int] --task_mode [mode]
   ```
#### Rerun trained agent
```load
$ python evaluate.py
```