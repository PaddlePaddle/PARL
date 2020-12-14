## Dependencies
- python3.6
- [parl==1.3.2](https://github.com/PaddlePaddle/PARL)
- Pytorch==1.6.0
- [grid2op==1.2.2](https://github.com/rte-france/Grid2Op)
- [lightsim2grid==0.2.4](https://github.com/BDonnot/lightsim2grid)

## How to evaluate
  1. Clone the repository.
  2. Download the saved models from online storage service: [Baidu Pan](https://pan.baidu.com/s/1qpylN5QJA-h6EcaoUC1sgg) (password: `0r7v`) or [Google Drive](https://drive.google.com/file/d/1FuPz5bEeMSTM9QMR3cpbzH69TLMhklr4/view?usp=sharing)
  3. Unpack the file:  
	```
	tar -zxvf saved_files.tar.gz
	```
  4. evaluate the result:  
	```
	python evaluate.py --nb_episode=10
	```
