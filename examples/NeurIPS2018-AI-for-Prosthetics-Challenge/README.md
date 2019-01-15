# The Winning Solution for the NeurIPS 2018: AI for Prosthetics Challenge

This folder contains the competitive solution of team `Firework`, who have won the NeurIPS 2018: AI for Prosthetics Challenge. It consists of three parts. The first part is our final submitted model, a sensible controller that can follow random target velocity. The second part is used for curriculum learning, to learn a natural and efficient gait at low-speed walking. The last part learns the final agent in the random velocity environment for round2 evaluation.

For more technical details about our solution, we provide:
1. [[Link]](https://youtu.be/RT4JdMsZaTE) An interesting video demonstrating the training process visually.
2. [[Link]](https://docs.google.com/presentation/d/1n9nTfn3EAuw2Z7JichqMMHB1VzNKMgExLJHtS4VwMJg/edit?usp=sharing) A PowerPoint Presentation briefly introducing our solution in NeurIPS2018 competition workshop.
3. (coming soon)A full academic paper detailing our solution, including entire training pipline, related work and experiments that analyze the importance of each key ingredient.

**Note**: Reproducibility is a long-standing issue in reinforcement learning field. We have tried to guarantee that our code is reproducible, testing each training sub-task three times. However, there are still some factors that prevent us from achieving the same performance. One problem is the choice time of a convergence model during curriculum learning. Choosing a sensible and natural gait visually is crucial for subsequent training, but the definition of what is a good gait varies from different people.


## Dependencies
- python3.6
- [paddlepaddle>=1.2.1](https://github.com/PaddlePaddle/Paddle)
- [osim-rl](https://github.com/stanfordnmbl/osim-rl)
- [grpcio==1.12.1](https://grpc.io/docs/quickstart/python.html)
- tqdm
- tensorflow (To use tensorboard)

## Part1: Final submited model
### Result
For final submission, we test our model in 500 CPUs, running 10 episodes per CPU with different random seeds.

| Avg reward of all episodes | Avg reward of complete episodes | Falldown % | Evaluate episodes |
|----------------------------|---------------------------------|------------|-------------------|
| 9968.5404                  | 9980.3952                       | 0.0026     | 5000              |

### Start test our submit models
- How to Run

  1. Enter the sub-folder `final_submit`
  2. Download the model file from online stroage service, [Baidu Pan](https://pan.baidu.com/s/1NN1auY2eDblGzUiqR8Bfqw) or [Google Drive](https://drive.google.com/open?id=1DQHrwtXzgFbl9dE7jGOe9ZbY0G9-qfq3) 
  3. Unpack the file by using: 
           `tar zxvf saved_model.tar.gz`
  4. Launch test scription: 
           `python test.py`

## Part2: Curriculum learning

#### 1. Run Fastest

```bash
# server
python simulator_server.py --port [PORT] --ensemble_num 1 

# client (Suggest: 200+ clients)
python simulator_client.py --port [PORT] --ip [IP] --reward_type RunFastest
```

#### 2. target speed 3.0 m/s

```bash
# server
python simulator_server.py --port [PORT] --ensemble_num 1 --warm_start_batchs 1000 \
           --restore_model_path [RunFastest model]

# client (Suggest: 200+ clients)
python simulator_client.py --port [PORT] --ip [IP] --reward_type FixedTargetSpeed --target_v 3.0 \
           --act_penalty_lowerbound 1.5 
```

#### 3. target speed 2.0 m/s

```bash
# server
python simulator_server.py --port [PORT] --ensemble_num 1 --warm_start_batchs 1000 \
           --restore_model_path [FixedTargetSpeed 3.0m/s model]

# client (Suggest: 200+ clients)
python simulator_client.py --port [PORT] --ip [IP] --reward_type FixedTargetSpeed --target_v 2.0 \
           --act_penalty_lowerbound 0.75 
```

#### 4. target speed 1.25 m/s

```bash
# server
python simulator_server.py --port [PORT] --ensemble_num 1 --warm_start_batchs 1000 \
           --restore_model_path [FixedTargetSpeed 2.0m/s model]  

# client (Suggest: 200+ clients)
python simulator_client.py --port [PORT] --ip [IP] --reward_type FixedTargetSpeed --target_v 1.25 \
           --act_penalty_lowerbound 0.6
```

### Part3: Training in random velocity environment in round2
As mentioned before, the selection of model that used to fine-tune influence later training. For those who can not obtain expected performance by former steps, a pre-trained model that walk naturally at 1.25m/s is provided. ([Baidu Pan](https://pan.baidu.com/s/1PVDgIe3NuLB-4qI5iSxtKA) or [Google Drive](https://drive.google.com/open?id=1jWzs3wvq7_ierIwGZXc-M92bv1X5eqs7))

```bash
# server
python simulator_server.py --port [PORT] --ensemble_num 12 --warm_start_batchs 1000 \
           --restore_model_path [FixedTargetSpeed 1.25m/s] --restore_from_one_head 

# client (Suggest: 100+ clients)
python simulator_client.py --port [PORT] --ip [IP] --reward_type Round2 --act_penalty_lowerbound 0.75 \
           --act_penalty_coeff 7.0 --vel_penalty_coeff 20.0 --discrete_data --stage 3
```

### Test trained model

```bash
python test.py --restore_model_path [MODEL_PATH] --ensemble_num [ENSEMBLE_NUM]
```

### Other implementation details
Following the above steps correctly, you can get an agent that can scores around 9960 in round2. Its performance is slightly poorer than our final submitted model. The score gap results from optimizing launch action. To get better performance, an independent model called launch model, is only trained to walk at 1.25m/s from a standing position. Then we fix the launch model and train a new controller for the subsequent chanllege. We do not provide this part of the code, since it reduces the readability of the code. Feel free to post issue if you have any problem:)

### Acknowledgments
We would like to thank Jingzhou He, Kai Zeng for providing computation resources and other colleagues on the Online Learning team for insightful discussions. We are grateful to Tingru Hong, Wenxia Zheng and others for creating a vivid and popular demonstration video.
