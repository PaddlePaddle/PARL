## Parallel Training with PARL

Use parl.compile to train the model parallelly. When applying offline training or dataset is too large to train on a single GPU, we can use parallel computing to accelerate training.
```python
# Set CUDA_VISIBLE_DEVICES to select which GPUs to train 

import parl
import paddle.fluid as fluid

learn_program = fluid.Program()
with fluid.program_guard(learn_program):
    # Define your learn program and training loss
    pass

learn_program = parl.compile(learn_program, loss=training_loss)  
# Pass the training loss to parl.compile. Distribute the model and data to GPUs.
```

## Demonstration

We provide a demonstration of offline Q-learning with parallel executing, in which we seperate the procedures of collecting data and training the model. First we collect data by interacting with the environment and save them to a replay memory file, and then fit and evaluate the Q network with the collected data. Repeat these two steps to improve the performance gradually.

### Dependencies:
+ [paddlepaddle==1.8.5](https://github.com/PaddlePaddle/Paddle)
+ [parl<2.0.0](https://github.com/PaddlePaddle/PARL)
+ gym
+ tqdm
+ atari-py

### How to Run:
```shell
# Collect training data
python parallel_run.py --rom rom_files/pong.bin

# Train the model offline with multi-GPU
python parallel_run.py --rom rom_files/pong.bin --train
```
