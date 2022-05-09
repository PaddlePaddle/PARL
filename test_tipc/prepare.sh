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

mujoco_envs="DDPG TD3 SAC PPO CQL ES OAC"
echo $model_name



if [[ $mujoco_envs =~ $model_name ]]; then

  # Get the prereqs
    if [ ! -d ~/.mujoco/mjpro131/ ]; then
        apt-get -qq update
        apt-get -qq install -y libosmesa6-dev libgl1-mesa-glx libglfw3 libgl1-mesa-dev libglew-dev patchelf
        # Get Mujoco
    if [ ! -d ~/.mujoco/ ];then
        mkdir ~/.mujoco
    fi
        cd ~/.mujoco
        wget -q https://roboti.us/download/mjpro131_linux.zip -O mjpro131_linux.zip
        unzip mjpro131_linux.zip
        rm mjpro131_linux.zip
    if [ ! -f "mjkey.txt" ];then
        wget https://roboti.us/file/mjkey.txt
    fi
        cp mjkey.txt ~/.mujoco/mjpro131/bin
    fi

    if [ ! -d ~/.mujoco/mujoco210/ ]; then
        apt-get -qq update
        apt-get -qq install -y libosmesa6-dev libgl1-mesa-glx libglfw3 libgl1-mesa-dev libglew-dev patchelf
    # Get Mujoco
    if [ ! -d ~/.mujoco/ ];then
        mkdir ~/.mujoco
    fi
    wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O ~/.mujoco/mujoco.tar.gz
    cd ~/.mujoco/
    tar zxvf ~/.mujoco/mujoco.tar.gz
    rm ~/.mujoco/mujoco.tar.gz
    if [ ! -f "mjkey.txt" ];then
        wget https://roboti.us/file/mjkey.txt
    fi
    cp mjkey.txt ~/.mujoco/mujoco210/bin
    fi

    if [ ! -d ~/.mujoco/mujoco200/ ]; then
        apt-get -qq update
        apt-get -qq install -y libosmesa6-dev libgl1-mesa-glx libglfw3 libgl1-mesa-dev libglew-dev patchelf
    # Get Mujoco
    if [ ! -d ~/.mujoco/ ];then
        mkdir ~/.mujoco
    fi
    cd ~/.mujoco/
    wget -q https://roboti.us/download/mujoco200_linux.zip
    unzip mujoco200_linux.zip
    mv mujoco200_linux mujoco200
    rm mujoco200_linux.zip
    if [ ! -f "mjkey.txt" ];then
        wget https://roboti.us/file/mjkey.txt
    fi
    cp mjkey.txt ~/.mujoco/mujoco200/bin

    fi
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mjpro131/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco200/bin
fi

# update pip
${python_name} -m pip install --upgrade pip

# python package
if [[ ${model_name} == "CQL" ]];then
    apt install openssl
    ${python_name} -m pip install gym==0.20.0
    ${python_name} -m pip install mujoco_py==2.0.2.8
    apt-get install gnutls-bin
    git config --global http.sslVerify false
    git config --global http.postBuffer 1048576000
    ${python_name} -m pip install git+http://gitlab.baidu.com/liuyixin04/dm_control@main#egg=dm_control
    ${python_name} -m pip install git+http://gitlab.baidu.com/liuyixin04/mjrl@main#egg=mjrl
    ${python_name} -m pip install pybullet
    ${python_name} -m pip install git+http://gitlab.baidu.com/liuyixin04/d4rl@main#egg=d4rl --no-deps
else
    sed -i '/paddlepaddle/d' ./examples/${model_name}/requirements.txt
    sed -i '/parl/d' ./examples/${model_name}/requirements.txt
    ${python_name} -m pip install -r ./examples/${model_name}/requirements.txt
    sed '$ a paddlepaddle' ./examples/${model_name}/requirements.txt
    sed '$ a parl' ./examples/${model_name}/requirements.txt
fi

# parl install
${python_name} -m pip install -e .

# prepare xparl for distributed training
if [[ ${model_name} == "A2C" ]];then
    xparl stop
    xparl start --port 8010 --cpu_num 5
elif [[ ${model_name} == "ES" ]];then
    xparl stop
    xparl start --port 8037 --cpu_num 2
fi
