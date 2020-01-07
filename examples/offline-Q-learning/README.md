## Parallel Training with PARL

Use parl.compile to train the model parallelly.
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

## Run this demonstration
### Dependencies:
+ [paddlepaddle>=1.5.1](https://github.com/PaddlePaddle/Paddle)
+ [parl](https://github.com/PaddlePaddle/PARL)
+ gym
+ tqdm
+ atari-py

### Offline Training:
```shell
# Collect training data
python parallel_run.py --rom rom_file/pong.bin

# Train the model offline with multi-GPU
python parallel_run.py --rom rom_file/pong.bin --train
```
