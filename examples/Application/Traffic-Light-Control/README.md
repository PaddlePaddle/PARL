## Reproduce Some Baselines of Traffic Light Control
Based on PARL, we use the DDQN algorithm of deep RL to reproduce some baselines of the Traffic Light Control(TLC), reaching the same level of indicators as the papers in TLC benchmarks.

### Traffic Light Control Simulator Introduction

Please see [sumo](https://github.com/eclipse/sumo) or [cityflow](https://github.com/cityflow-project/CityFlow) to know more about the TLC simulator.
And we use the cityflow simuator in the experiments, as for how to install the cityflow, please refer [here](https://cityflow.readthedocs.io/en/latest/index.html) for more informations.

### Benchmark Result
Note that we set the yellow signal time to 5 seconds to clear the intersection, and the action intervals is set to 10 seconds as the papers, you can refer the `config.py` for details, you also can change the time as what you want. The different values of the times above may cause different results of the experiments.
You can download the data from [here](https://traffic-signal-control.github.io/) and [MPLight data](https://github.com/Chacha-Chen/MPLight/tree/master/data).
We use the average travel time of all vehicles to evaluate the performance of the signal control method in transportation.
Performances of presslight and FRAP on cityflow envrionments in training process after 300 episodes are shown below.

| average travel time| hz_1x1_tms-<br>xy_18041608| hz_1x1_bc-<br>tyc_18041608|syn_1x3_<br>gaussian|syn_2x2_<br>gaussian|anon_4_4_<br>750_0.6| anon_4_4<br>_750_0.3| anon_4_4<br>_700_0.6|anon_4_4<br>_700_0.3|
| :-----| :----: | :----: |:----: | :----: |:----: | :----: |:----: | :----: |
| max_pressure | 284.02 | 445.62 | 240.08 |316.67|589.03 | 536.89 |545.29 | 483.08 |
| presslight |110.62 | 189.97| 127.83| 184.58| 437.86| 357.10 |410.34 | 434.33|
| FRAP | 113.79 | 135.88 | 123.97| 166.45| 374.73 | 331.43 | 343.79| 300.77 |
| presslight* |  236.29|  244.87 |149.40| 953.78| -- | --| --| -- |
| FRAP* | 130.53| 159.54| 750.68| 713.48|--| -- |-- | -- |


Note that for the method `sotl`, different `t_min`, `min_green_vehicle` and `max_red_vehicle` configs may cause huge different results, which may not fair for sotl to compare its result with others, so we don't list the result of the `sotl` above.

And results of the last two rows of the table ,`presslight*` and `FRAP*`, they are the results of the code [tlc-baselines](https://github.com/gjzheng93/tlc-baselines) provided from the paper authors' team. We run the [code](https://github.com/gjzheng93/tlc-baselines) just changing the yellow time and the action intervals to keep them same as our cinfig as the papers without changing any other parameters. `--` in the table means the origins code doesn't performs well in the last four `anon_4X4` datas, the average travel time results of it will be more than 1000, maybe it will performs better than the `max_pressure`if you modify the other hyperparameters of the DQN agents, such as the buffer size, update_model_freq, the gamma or others.

## How to use
### Dependencies
+ [parl>=1.4.3](https://github.com/PaddlePaddle/PARL)
+ torch==1.8.1+cu102
+ cityflow==0.1

### Training 
Run the training script, the `train_presslight.py `for the presslight, each intersection has its own model as default(also you can choose to train with that all the intersections share one model in the script, just as what the paper MPLight used, it is suggested when the number of the intersections is large, just setting the `--is_share_model` to `True`).
```bash
python train_presslight.py --is_share_model False
```

If you want the train the `FRAR`, you can run the script below:
```bash
python train_FRAP.py 
```

If you want to compare the different results, you can load the right model path in the `config.py` and the right data path in the `config.json`, then run:
```bash
python test.py 
```

### Contents
+ agent 
    + `agent.py`
    The agent that uses the PARL agent mode, it will be used when training the RL methods such as `presslight` or `FRAP` and so on.
    + `max_pressure_agent.py` and `sotl_agnet.py`.The classic methods of the TLC. 
+ data
    + You can get the data of the from here or download other data and put them here.
+ example
    + Put the `config.json` here, need to change the path of the roadnet the flow data in the `json` file.
+ model
    + Different algorithms have different models.
+ obs_reward
    + Different algorithms have different obs and rewards generators.


### Something about the Distributed Training

We don't use the distributed traing or the parallel actors for collect the datas from the cityflow env, if you want to use the parallel actors with the cluster, you can refer to [here](https://github.com/PaddlePaddle/PARL/tree/develop/examples/A2C) or our [documentation](https://parl.readthedocs.io/en/latest/parallel_training/setup.html) for details. 

### Some Suggestions and Conclusions
+ The classic method `max_pressure`, `solt` or `greedy`(just set green lights to the roads with the most vehicles) can get the not bad baselines, when you use the RL method, you can compare to those baselines to make sure there is no mistakes in the RL code and the training process.
+ As for the just one intersection roadnet data, from our experiences:
    + `presslight` can get the high baselines results, if you want to get better results, you can try `FRAP` in your own data, if the flow data and the roadnet is easy without so many vehicles, `presslight` maybe better.
+ If your roadnet contains hundreds intersections, it is unrealistic to make each model to each agent(intersection), you can choose to train with that all the intersections share one common model and one buffer. As for the complicated scene, the complicated model `FRAR`, `Colight`,`GAT` or `multi-agents` methods may be better.
+ The replay memory size and the gamma doesn't matter much from our experiences.
+ As the reward is hard or inappropriate to design, we suggest that the `ES` maybe a better choice, and we also have tested same data with the [ES](https://github.com/PaddlePaddle/PARL/tree/develop/benchmark/torch/ES), just use the negative average travel time as the fitness(rewards), it can get the better results when we create enough actors in the [cluster](https://parl.readthedocs.io/en/latest/parallel_training/setup.html).   
+ The RL methods is just overfitting the env with the specific flow and roadnet data, maybe when evaluating the results we can test the model with different flow or roadnet data?


### Reference
+ [Parl](https://parl.readthedocs.io/en/latest/parallel_training/setup.html)
+ [Reinforcement Learning for Traffic Signal Control](https://traffic-signal-control.github.io/)
+ [Toward A Thousand Lights: Decentralized Deep Reinforcement Learning for Large-Scale Traffic Signal Control](https://chacha-chen.github.io/papers/chacha-AAAI2020.pdf)
+ [Traffic Light Control Baselines](https://github.com/zhc134/tlc-baselines)
+ [PressLight: Learning Max Pressure Control to Coordinate Traffic Signals in Arterial Network](http://personal.psu.edu/hzw77/publications/presslight-kdd19.pdf)
+ [PressLight](https://github.com/wingsweihua/presslight)
+ [Learning Phase Competition for Traffic Signal Control](http://www.personal.psu.edu/~gjz5038/paper/cikm2019_frap/cikm2019_frap_paper.pdf)
+ [frap-pub](https://github.com/gjzheng93/frap-pub)
