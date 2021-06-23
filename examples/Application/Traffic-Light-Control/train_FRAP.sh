#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python train_FRAP.py  --config_path_name './examples/config_hz_1.json' --save_dir 'save_model_frap/hz_1'&
# CUDA_VISIBLE_DEVICES=1 python train_FRAP.py  --config_path_name './examples/config_hz_2.json' --save_dir 'save_model_frap/hz_2'&
wait
