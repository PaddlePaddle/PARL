# The Winning Solution for the NeurIPS 2018: AI for Prosthetics Challenge

This folder contains the code used to train the winning models for the [NeurIPS 2018: AI for Prosthetics Challenge](https://www.crowdai.org/challenges/neurips-2018-ai-for-prosthetics-challenge) along with the resulting models.
## Dependencies
- python3.6
- [paddlepaddle>=1.2.1](https://github.com/PaddlePaddle/Paddle)
- [osim-rl](https://github.com/stanfordnmbl/osim-rl)
- grpcio==1.12.1
- tensorflow (To use tensorboard)

## Result

| Avg reward of all episodes | Avg reward of complete episodes | Falldown % | Evaluate episodes |
|----------------------------|---------------------------------|------------|-------------------|
| 9968.5404                  | 9980.3952                       | 0.0026     | 5000              |

## Start test our submit models
- How to Run

```bash
# cd current directory
# cd final_submit/
# install submit models file (saved_model.tar.gz)
tar zxvf saved_model.tar.gz
python test.py
```
> You can install models file from [Baidu Pan](https://pan.baidu.com/s/1NN1auY2eDblGzUiqR8Bfqw) or [Google Drive](https://drive.google.com/open?id=1DQHrwtXzgFbl9dE7jGOe9ZbY0G9-qfq3)


## Start train

### Stage I: Curriculum learning

#### 1. Run Fastest

```bash
# server
python .simulator_server.py --port [PORT] --ensemble_num 1 

# client (Suggest: 200+ clients)
python simulator_client.py --port [PORT] --ip [IP] --reward_type RunFastest
```

#### 2. target speed 3.0 m/s

```bash
# server
python .simulator_server.py --port [PORT] --ensemble_num 1 --restore_model_path [RunFastest model] --warm_start_batchs 1000

# client (Suggest: 200+ clients)
python simulator_client.py --port [PORT] --ip [IP] --reward_type FixedTargetSpeed --target_v 3.0 --act_penalty_lowerbound 1.5 
```

#### 3. target speed 2.0 m/s

```bash
# server
python .simulator_server.py --port [PORT] --ensemble_num 1 --restore_model_path [FixedTargetSpeed 3.0m/s model] --warm_start_batchs 1000

# client (Suggest: 200+ clients)
python simulator_client.py --port [PORT] --ip [IP] --reward_type FixedTargetSpeed --target_v 2.0 --act_penalty_lowerbound 0.75 
```

#### 4. target speed 1.25 m/s

```bash
# server
python .simulator_server.py --port [PORT] --ensemble_num 1 --restore_model_path [FixedTargetSpeed 2.0m/s model] --warm_start_batchs 1000 

# client (Suggest: 200+ clients)
python simulator_client.py --port [PORT] --ip [IP] --reward_type FixedTargetSpeed --target_v 1.25 --act_penalty_lowerbound 0.6
```

### Stage II: Round2

> You can install resulting 1.25m/s model in Stage I from [Baidu Pan](https://pan.baidu.com/s/1PVDgIe3NuLB-4qI5iSxtKA) or [Google Drive](https://drive.google.com/open?id=1jWzs3wvq7_ierIwGZXc-M92bv1X5eqs7)

```bash
# server
python .simulator_server.py --port [PORT] --ensemble_num 12 --restore_model_path [FixedTargetSpeed 1.25m/s] --restore_from_one_head --warm_start_batchs 1000

# client (Suggest: 100+ clients)
python simulator_client.py --port [PORT] --ip [IP] --reward_type Round2 --act_penalty_lowerbound 0.75 --act_penalty_coeff 7.0 --vel_penalty_coeff 20.0 --discrete_data --stage 3
```

> To get a higher score, you need train a seperate model for every stage (target_v change times), and fix trained model of previous stage. It's omitted here.

### Test trained model

```bash
python test.py --restore_model_path [MODEL_PATH] --ensemble_num [ENSEMBLE_NUM]
```
