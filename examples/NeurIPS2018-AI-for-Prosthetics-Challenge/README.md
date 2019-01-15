# The Winning Solution for the NeurIPS 2018: AI for Prosthetics Challenge

This folder contains the competitive solution of team `Firework`, who have won the NeurIPS 2018: AI for Prosthetics Challenge. It consists of three parts. The first part is our final submited model, a sensible controller that is able to follow random target velocity. The second part is used for curriculum learning, to learn a natural and efficient gait at low-speed walking. The last part learns the final agent in the random velocity environment for round2 evaluation.

For more technical details about our solution, we provide:
1. [[Link]](https://youtu.be/RT4JdMsZaTE) An interesting video demonstrating the training process visually.
2. [[Link]](https://docs.google.com/presentation/d/1n9nTfn3EAuw2Z7JichqMMHB1VzNKMgExLJHtS4VwMJg/edit?usp=sharing) A PowerPoint Presentation to briefly introduce our solution in NeurIPS2018 competition workshop.
3. (coming soon)A full academic paper to describe our solution in detail.

## Dependencies
- python3.6
- [paddlepaddle>=1.2.1](https://github.com/PaddlePaddle/Paddle)
- [osim-rl](https://github.com/stanfordnmbl/osim-rl)
- [grpcio==1.12.1](https://grpc.io/docs/quickstart/python.html)
- tqdm
- tensorflow (To use tensorboard)

## Result

| Avg reward of all episodes | Avg reward of complete episodes | Falldown % | Evaluate episodes |
|----------------------------|---------------------------------|------------|-------------------|
| 9968.5404                  | 9980.3952                       | 0.0026     | 500 CPUs * 10 episodes              |

## Start test our submit models
- How to Run

```bash
# cd current directory
# cd final_submit/
# download submit models file (saved_model.tar.gz)
tar zxvf saved_model.tar.gz
python test.py
```
> You can download models file from [Baidu Pan](https://pan.baidu.com/s/1NN1auY2eDblGzUiqR8Bfqw) or [Google Drive](https://drive.google.com/open?id=1DQHrwtXzgFbl9dE7jGOe9ZbY0G9-qfq3)


## Start train

### Stage I: Curriculum learning

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

### Stage II: Round2

> You can download resulting 1.25m/s model in Stage I from [Baidu Pan](https://pan.baidu.com/s/1PVDgIe3NuLB-4qI5iSxtKA) or [Google Drive](https://drive.google.com/open?id=1jWzs3wvq7_ierIwGZXc-M92bv1X5eqs7)

```bash
# server
python simulator_server.py --port [PORT] --ensemble_num 12 --warm_start_batchs 1000 \
           --restore_model_path [FixedTargetSpeed 1.25m/s] --restore_from_one_head 

# client (Suggest: 100+ clients)
python simulator_client.py --port [PORT] --ip [IP] --reward_type Round2 --act_penalty_lowerbound 0.75 \
           --act_penalty_coeff 7.0 --vel_penalty_coeff 20.0 --discrete_data --stage 3
```

> To get a higher score, you need train a seperate model for every stage (target_v change times), and fix trained model of previous stage. It's omitted here.

### Test trained model

```bash
python test.py --restore_model_path [MODEL_PATH] --ensemble_num [ENSEMBLE_NUM]
```
