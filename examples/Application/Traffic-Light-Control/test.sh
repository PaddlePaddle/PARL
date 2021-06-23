#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python test.py  --config_path_name './examples/config_hz_1.json' --result_name 'hz_1' --save_dir 'save_model/presslight'&
wait
