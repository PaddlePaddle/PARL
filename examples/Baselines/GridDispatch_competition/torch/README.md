## SAC baseline for grid dispatching competition

In this example, we provide a distributed SAC baseline based on PARL and torch for the [grid dispatching competition](https://aistudio.baidu.com/aistudio/competition/detail/111) task.

### Dependencies
* Linux
* python3.6+
* torch == 1.6.0
* parl >= 2.0.0

### Computing resource requirements
* 1 GPU + 6 CPUs

### Training

1. Download the pretrained model (trained with fixed first 288 timesteps data) in the current directory. (filename: `torch_pretrain_model`)

[Baidu Pan](https://pan.baidu.com/s/1Pqv9i9byOzqStcHdttOlRA) (password: n9qc)

2. Copy all files of `gridsim` (the competition package) to the current directory.
```bash
# For example:
cp -r /XXX/gridsim/* .
```

2. Update the data path for distributed training (Using an absoluate path).
```bash
export PWD=`pwd`
python yml_creator.py --dataset_path $PWD/data
```


3. Set the environment variable of PARL and gridsim.
```bash
export PARL_BACKEND=torch
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib64
```

4. Start xparl cluster

```bash
# You can change following `cpu_num` and `args.actor_num` in the train.py based on the CPU number of your machine.
# Note that you only need to start the cluster once.

xparl start --port 8010 --cpu_num 6
```

5. start training. 

```bash
python train.py --actor_num 6
```

6. Visualize the training curve and other information.
```
tensorboard --logdir .
```

### Performance
The result after training one hour with 1 GPU and 6 CPUs.
![learning curve](https://raw.githubusercontent.com/benchmarking-rl/PARL-experiments/master/Baselines/GridDispatch_competition/torch/result.png)
