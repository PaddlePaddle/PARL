#!/bin/bash
cd "$(dirname "$0")"

if [ $# != 1 ]; then
  echo "You must specify which gpu to use"
  exit 0
fi

source ~/.bashrc
source activate parl
export CUDA_VISIBLE_DEVICES="$1"
export FLAGS_fraction_of_gpu_memory_to_use=0.2

unset http_proxy
unset https_proxy

FLAGS_enforce_when_check_program_=0 GLOG_vmodule=operator=1,computation_op_handle=1 python ./simulator_server.py \
          --port 9000 \
          --ensemble_num 1 \
          --warm_start_batchs 3 \
