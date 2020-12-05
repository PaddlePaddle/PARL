## Dependencies
- python3.6
- [parl==1.3.2](https://github.com/PaddlePaddle/PARL)
- [paddlepaddle==1.6.1](https://github.com/PaddlePaddle/Paddle)
- [grid2op==1.2.2](https://github.com/rte-france/Grid2Op)
- [lightsim2grid==0.2.4](https://github.com/BDonnot/lightsim2grid)

## How to evaluate
  1. Clone the repository.
  2. Download the saved models from online storage service: [Baidu Pan](https://pan.baidu.com/s/1nqrIDomycy3D4OINSQV-8w) (password: `4801`) or [Google Drive](https://drive.google.com/file/d/1hq4Xf_xywrm3I-1bJNQt_QKrOi8HJrrr/view?usp=sharing)
  3. Unpack the file:  
    ```
    tar -xvzf saved_files.tar.gz
    ```
  4. evaluate the result:  
    ```
    python evaluate.py --num_episodes=10
    ```
