#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_presslight.py  --config_path_name './examples/config_hz_1.json' --save_dir 'save_model/hz_1' --is_share_model False&
# CUDA_VISIBLE_DEVICES=1 python train_presslight.py  --config_path_name './examples/config_hz_2.json' --save_dir 'save_model/hz_2' --is_share_model False&

wait
