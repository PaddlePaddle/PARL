# The Winning Solution for the NeurIPS 2019: Learn to Move Challenge

## Dependencies
- python3.6
- [paddlepaddle>=1.5.2](https://github.com/PaddlePaddle/Paddle)
- [parl>=1.2.1](https://github.com/PaddlePaddle/PARL)
- [osim-rl==3.0.11](https://github.com/stanfordnmbl/osim-rl)


## Part1: Final submitted model
### Test
- How to Run

  1. Enter the sub-folder `final_submit`
  2. Download the model file from online stroage service: [Baidu Pan](https://pan.baidu.com/s/12LIPspckCT8-Q5U1QX69Fg) (password: `b5ck`) or [Google Drive](https://drive.google.com/file/d/1jJtOcOVJ6auz3s-TyWgUJvofPXI94yxy/view?usp=sharing)
  3. Unpack the file:
           `tar zxvf saved_models.tar.gz`
  4. Launch the test script:
           `python test.py`


## Part2: Curriculum learning

#### 1. Run as fast as possible -> run at 3.0 m/s -> walk at 2.0 m/s -> walk slowly at 1.3 m/s
The curriculum learning pipeline to get a walking slowly model is the same pipeline in [our winning solution in NeurIPS 2018: AI for Prosthetics Challenge](https://github.com/PaddlePaddle/PARL/tree/develop/examples/NeurIPS2018-AI-for-Prosthetics-Challenge). You can get a walking slowly model by following the [guide](https://github.com/PaddlePaddle/PARL/tree/develop/examples/NeurIPS2018-AI-for-Prosthetics-Challenge#part2-curriculum-learning).

We also provide a pre-trained model that walk naturally at ~1.3m/s. You can download the model file (naming `low_speed_model`) from online stroage service: [Baidu Pan](https://pan.baidu.com/s/1Mi_6bD4QxLWLdyLYe2GRFw) (password: `q9vj`) or [Google Drive](https://drive.google.com/file/d/1_cz6Cg3DAT4u2a5mxk2vP9u8nDWOE7rW/view?usp=sharing).

#### 2. difficulty=1
At first, start a local cluster with 300 CPUs:

```bash
xparl start --port 8010 --cpu_num 300
```

Note: If there are not enough CPUs in your local machine, you can add CPUs resource of other machines to the cluster by running the command:
```bash
xparl connect --address [CLUSTER_IP]:8081
```

For more information of xparl, please visit the [documentation](https://parl.readthedocs.io/en/latest/parallel_training/setup.html).

Then, we can start the distributed training by running:
```bash
python train.py --actor_num 300 \
           --difficulty 1 \
           --penalty_coeff 3.0 \
           --logdir ./output/difficulty1 \
           --restore_model_path ./low_speed_model
```

Optionally, you can start the distributed evaluating by running:
```bash
python evaluate.py --actor_num 160 \
           --difficulty 1 \
           --penalty_coeff 3.0 \
           --saved_models_dir ./output/difficulty1/model_every_100_episodes \
           --evaluate_times 300
```

#### 3. difficulty=2
```bash
python train.py --actor_num 300 \
           --difficulty 2 \
           --penalty_coeff 5.0 \
           --logdir ./output/difficulty2 \
           --restore_model_path [DIFFICULTY=1 MODEL]
```

#### 4. difficulty=3, first target
```bash
python train.py --actor_num 300 \
           --difficulty 3 \
           --vel_penalty_coeff 3.0 \
           --penalty_coeff 3.0 \
           --only_first_target \
           --logdir ./output/difficulty3_first_target \
           --restore_model_path [DIFFICULTY=2 MODEL]
```

#### 5. difficulty=3
```bash
python train.py --actor_num 300 \
           --difficulty 3 \
           --vel_penalty_coeff 3.0 \
           --penalty_coeff 2.0 \
           --rpm_size 6e6 \
           --train_times 250 \
           --logdir ./output/difficulty3 \
           --restore_model_path [DIFFICULTY=3 FIRST TARGET MODEL]
```
