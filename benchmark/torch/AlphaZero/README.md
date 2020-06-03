## AlphaZero baseline for Connect4 game (distributed version)

### Dependencies
- python3
- [parl==1.3](https://github.com/PaddlePaddle/PARL)
- torch
- tqdm

### Training 
1. Download the [1k connect4 validation set](https://www.kaggle.com/petercnudde/1k-connect4-validation-set) to the current directory. (filname: refmoves1k_kaggle)

2. Start xparl cluster
```bash
xparl start --port 8010 --cpu_num 25
```

3. Run training script
```bash
python main.py
```

### Submitting
Generate `submission.py` by provided script, for example:
```bash
python gen_submission.py saved_model/best.pth.tar
```
