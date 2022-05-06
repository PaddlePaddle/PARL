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

echo $model_name

#${python_name} -m pip install parl
${python_name} -m pip install --upgrade pip
sed -i '/paddlepaddle/d' ./examples/${model_name}/requirements.txt
sed -i '/parl/d' ./examples/${model_name}/requirements.txt
${python_name} -m pip install -r ./examples/${model_name}/requirements.txt
sed '$ a paddlepaddle' ./examples/${model_name}/requirements.txt
sed '$ a parl' ./examples/${model_name}/requirements.txt
${python_name} -m pip install -e .

mojuco_envs="DDPG TD3 SAC PPO CQL"
if [[ ${model_name} == "A2C" ]];then
  xparl stop
  xparl start --port 8010 --cpu_num 5
fi

if [[ $mojuco_envs =~ $model_name ]]; then
  # Get the prereqs
  if [ ! -d "~/.mujoco/" ]; then
    apt-get -qq update
    apt-get -qq install -y libosmesa6-dev libgl1-mesa-glx libglfw3 libgl1-mesa-dev libglew-dev patchelf
    # Get Mujoco
    mkdir ~/.mujoco
    wget -q https://roboti.us/download/mjpro131_linux.zip -O mjpro131_linux.zip
    unzip mjpro131_linux.zip -d "$HOME/.mujoco"
    rm mjpro131_linux.zip
    wget https://roboti.us/file/mjkey.txt
    cp mjkey.txt ~/.mujoco
    cp mjkey.txt ~/.mujoco/mjpro131/bin
  #  echo 'export LD_LIBRARY_PATH=~/.mujoco/mjpro131/bin${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
  #  echo 'export MUJOCO_KEY_PATH=~/.mujoco${MUJOCO_KEY_PATH}' >> ~/.bashrc
  #  echo 'export LD_PRELOAD=$LD_PRELOAD:/usr/lib/x86_64-linux-gnu/libGLEW.so' >> ~/.bashrc
  #  echo "/root/.mujoco/mjpro131/bin" > /etc/ld.so.conf.d/mujoco_ld_lib_path.conf
  #  ldconfig
#    ${python_name} -m pip install 'mujoco-py==0.5.7'
  #  os.environ['LD_LIBRARY_PATH']=os.environ[''] + ':/root/.mujoco/mjpro131/bin'
  #  os.environ['LD_PRELOAD']=os.environ['LD_PRELOAD'] + ':/usr/lib/x86_64-linux-gnu/libGLEW.so'
  #  os.environ['LD_LIBRARY_PATH']=os.environ['LD_LIBRARY_PATH']+':/usr/lib/nvidia'
  fi
fi
