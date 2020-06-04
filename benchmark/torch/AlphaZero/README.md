## AlphaZero baseline for Connect4 game (distributed version)
Based on [suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general)

### Dependencies
- python3
- [parl==1.3](https://github.com/PaddlePaddle/PARL)
- torch
- tqdm

### Training 
1. Download the [1k connect4 validation set](https://www.kaggle.com/petercnudde/1k-connect4-validation-set) to the current directory. (filname: refmoves1k_kaggle)

2. Start xparl cluster
```bash
# You can change following `cpu_num` and `args.actor_nums` in the main.py based on the CPU number of your machine.
xparl start --port 8010 --cpu_num 25
```

```
# [OPTIONAL] You can also run the following script in other machines to add more CPU resource to the xparl cluster, so you can increase the parallelism (args.actor_nums).
xparl connect --address MASTER_IP:8010 --cpu_num [CPU_NUM]
```

3. Run training script
```bash
python main.py
```

### Submitting
To submit the well-trained model to the Kaggle, you can use our provided script to generate `submission.py`, for example:
```bash
python gen_submission.py saved_model/best.pth.tar
```

### Performance
- Following are `good move rate` and `perfect move rate` indicators, please refer to the [link](https://www.kaggle.com/petercnudde/scoring-connect-x-agents) for specific meaning.

- It can reach about score 1368 in the Kaggle [Connect X](https://www.kaggle.com/c/connectx/leaderboard) competition now.


### Reference
- [suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general)
- [Scoring connect-x agents](https://www.kaggle.com/petercnudde/scoring-connect-x-agents)
