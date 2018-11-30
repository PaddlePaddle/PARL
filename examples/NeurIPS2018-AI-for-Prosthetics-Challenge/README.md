## The Winning Solution for the NeurIPS 2018: AI for Prosthetics Challenge

This folder will contains the code used to train the winning models for the [NeurIPS 2018: AI for Prosthetics Challenge](https://www.crowdai.org/challenges/neurips-2018-ai-for-prosthetics-challenge) along with the resulting models. (Codes of training part is organizing, but the resulting models is available now.)

### Dependencies
- python3.6
- [paddlepaddle==1.2.0](https://github.com/PaddlePaddle/Paddle)
- [PARL](https://github.com/PaddlePaddle/PARL)
- [osim-rl](https://github.com/stanfordnmbl/osim-rl)

### Start Testing best models
- How to Run
```bash
# cd current directory
# install best models file (saved_model.tar.gz) 
tar zxvf saved_model.tar.gz
python test.py
```
> You can install models file from [Baidu Pan](https://pan.baidu.com/s/1NN1auY2eDblGzUiqR8Bfqw) or [Google Drive](https://drive.google.com/open?id=1DQHrwtXzgFbl9dE7jGOe9ZbY0G9-qfq3)

- More arguments
```bash
# Run with GPU
python test.py --use_cuda 

# Visulize the game
python test.py --vis

# Set the random seed 
python test.py --seed 1024

# Set the episode number to run
python test.py --episode_num 2
```

### Start Training
- [ ] To be Done
