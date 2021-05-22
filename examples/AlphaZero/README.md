## AlphaZero baseline for Connect4 game

- In this example, we provide a paddle-based AlphaZero baseline to solve the Connect4 game, based on the code of [alpha-zero-general](https://github.com/suragnair/alpha-zero-general) repo.
- We take advantage of the parallelism capacity of [PARL](https://github.com/PaddlePaddle/PARL) to support running self-play and evaluating tasks in parallel.
- We also provide a script to play Connect4 game against AI trained by yourself.

### Dependencies
* python3
* paddlepaddle >= 2.0.0
* parl == 1.4.3
* tqdm

### Training 

1. Download the [1k connect4 validation set](https://www.kaggle.com/petercnudde/1k-connect4-validation-set) to the current directory. (filename: `refmoves1k_kaggle`)

2. Start xparl cluster

```bash
# You can change following `cpu_num` and `args.actor_nums` in the main.py 
# based on the CPU number of your machine.

xparl start --port 8010 --cpu_num 25
```

```bash
# [OPTIONAL] You can also run the following script in other machines to add more CPU resource 
#            to the xparl cluster, so you can increase the parallelism (args.actor_nums).

xparl connect --address MASTER_IP:8010 --cpu_num [CPU_NUM]
```

3. Run training script

```bash
python main.py
```

4. Visualize (good moves rate and perfect moves rate)

```bash
visualdl --logdir .
```

### Performance

- Following are `good moves rate` and `perfect moves rate` indicators in visualdl, please refer to the [link](https://www.kaggle.com/petercnudde/scoring-connect-x-agents) for specific meaning.

<img src=".pic/good_moves_rate.png" width = "300" alt="good moves rate"/> <img src=".pic/perfect_moves_rate.png" width = "300" alt="perfect moves rate"/>

> It takes about 2 day to run 25 iterations on the machine with 5 cpus.

### Play with AI

Pit the best model against human being.

```bash
python connect4_aiplayer.py 
```

### Reference

- [suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general)
