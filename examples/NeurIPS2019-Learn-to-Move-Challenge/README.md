The **PARL** team gets the first place in NeurIPS reinforcement learning competition, again! This folder contains our final submitted model and the code relative to the training process.

<p align="center">
<img src="image/performance.gif" alt="PARL" height="300" />
</p>

## Dependencies
- python3.6
- [parl==1.2.1](https://github.com/PaddlePaddle/PARL)
- [paddlepaddle==1.5.1](https://github.com/PaddlePaddle/Paddle)
- [parl>=1.2.1](https://github.com/PaddlePaddle/PARL)
- [osim-rl==3.0.11](https://github.com/stanfordnmbl/osim-rl)


## Part1: Final submitted model
### Test
- How to Run

  1. Enter the sub-folder `final_submit`
  2. Download the model file from online storage service: [Baidu Pan](https://pan.baidu.com/s/12LIPspckCT8-Q5U1QX69Fg) (password: `b5ck`) or [Google Drive](https://drive.google.com/file/d/1jJtOcOVJ6auz3s-TyWgUJvofPXI94yxy/view?usp=sharing)
  3. Unpack the file:
           `tar zxvf saved_models.tar.gz`
  4. Launch the test script:
           `python test.py`


## Part2: Curriculum learning

#### 1. Run as fast as possible -> run at 3.0 m/s -> walk at 2.0 m/s -> walk slowly at 1.3 m/s
The curriculum learning pipeline to get a walking slowly model is the same pipeline in [our winning solution in NeurIPS 2018: AI for Prosthetics Challenge](https://github.com/PaddlePaddle/PARL/tree/develop/examples/NeurIPS2018-AI-for-Prosthetics-Challenge). You can get a walking slowly model by following the [guide](https://github.com/PaddlePaddle/PARL/tree/develop/examples/NeurIPS2018-AI-for-Prosthetics-Challenge#part2-curriculum-learning).

We also provide a pre-trained model that walk naturally at ~1.3m/s. You can download the model file (naming `low_speed_model`) from online storage service: [Baidu Pan](https://pan.baidu.com/s/1Mi_6bD4QxLWLdyLYe2GRFw) (password: `q9vj`) or [Google Drive](https://drive.google.com/file/d/1_cz6Cg3DAT4u2a5mxk2vP9u8nDWOE7rW/view?usp=sharing).

#### 2. difficulty=1
> We built our distributed training agent based on PARL cluster. To start a PARL cluster, we can execute the following two xparl commands:
>
>
>```bash
># starts a master node to manage computation resources and adds the local CPUs to the cluster.
>xparl start --port 8010 
>```
>
>```bash
># if necessary, adds more CPUs (computation resources) in other machine to the cluster.
>xparl connect --address [CLUSTER_IP]:8010 
>```
>
> For more information of xparl, please visit the [documentation](https://parl.readthedocs.io/en/latest/parallel_training/setup.html).

In this example, we can start a local cluster with 300 CPUs by running:

```bash
xparl start --port 8010 --cpu_num 300
```

Then, we can start the distributed training by running:
```bash
# NOTE: You need provide a self-trained model, or download the `low_speed_model` as mentioned above.
sh scripts/train_difficulty1.sh ./low_speed_model
```

Optionally, you can start the distributed evaluating by running:
```bash
sh scripts/eval_difficulty1.sh
```

#### 3. difficulty=2
```bash
sh scripts/train_difficulty2.sh [TRAINED DIFFICULTY=1 MODEL]
```

#### 4. difficulty=3, first target
```bash
sh scripts/train_difficulty3_first_target.sh [TRAINED DIFFICULTY=2 MODEL]
```

#### 5. difficulty=3
```bash
sh scripts/train_difficulty3.sh [TRAINED DIFFICULTY=3 FIRST TARGET MODEL]
```
