#!/bin/bash
source test_tipc/common_func.sh

FILENAME=$1

# MODE be one of ['lite_train_lite_infer' 'lite_train_whole_infer' 'whole_train_whole_infer',  
#                 'whole_infer', 'klquant_whole_infer',
#                 'cpp_infer', 'serving_infer']

MODE=$2

dataline=$(cat ${FILENAME})

# parser params
IFS=$'\n'
lines=(${dataline})



# The training params
model_name=$(func_parser_value "${lines[1]}")

trainer_list=$(func_parser_value "${lines[14]}")

python_name_list=$(func_parser_value "${lines[2]}")
array=(${python_name_list})
python_name=${array[0]}

pip3 install -r ./examples/${model_name}requirements.txt

if [[ ${model_name} == "A2C" ]];then
#  ${python_name} -m pip install paddlepaddle>=2.0.0
#  ${python_name} -m pip install atari-py==0.1.7
#  ${python_name} -m pip install parl>=1.4.3
#  ${python_name} -m pip gym==0.12.1
  xparl start --port 8010 --cpu_num 2
#elif [ ${model_name} = "DQN" ];then
#  ${python_name} -m pip install paddlepaddle>=2.0.0
#  ${python_name} -m pip install gym
#  ${python_name} -m pip install parl>=2.0.0
#elif [ ${model_name} = "MADDPG" ];then
#  ${python_name} -m pip install paddlepaddle>=2.0.0
#  ${python_name} -m pip install parl>=2.0.4
#  ${python_name} -m pip install PettingZoo==1.17.0
#  ${python_name} -m pip install gym==0.23.1
#elif [ ${model_name} = "QuickStart" ];then
#  ${python_name} -m pip install gym
#  ${python_name} -m pip install paddlepaddle>=2.0.0
#  ${python_name} -m pip install parl>=2.0.0
fi