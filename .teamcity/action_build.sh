#!/usr/bin/env bash

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -ex

function init() {
    RED='\033[0;31m'
    BLUE='\033[0;34m'
    BOLD='\033[1m'
    NONE='\033[0m'

    REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../" && pwd )"

    ls -l /usr/local/
    export LC_ALL=C.UTF-8
    export LANG=C.UTF-8

    which python
    python -c "import platform;print(platform.architecture()[0]);print(platform.machine())"
}

function run_example_test {
    for exp in QuickStart DQN DQN_variant PPO SAC TD3 OAC DDPG MADDPG ES A2C
    do
        sed -i '/paddlepaddle/d' ./examples/${exp}/requirements*.txt
        sed -i '/parl/d' ./examples/${exp}/requirements*.txt
    done
    
    python -m pip install -r ./examples/QuickStart/requirements.txt
    python examples/QuickStart/train.py
    python -m pip uninstall -r ./examples/QuickStart/requirements.txt -y

    # TODO: raise Error in Atari env
    # python -m pip install -r ./examples/DQN/requirements.txt
    # python examples/DQN/train.py
    # python -m pip uninstall -r ./examples/DQN/requirements.txt -y
    
    # python -m pip install -r ./examples/DQN_variant/requirements.txt
    # python examples/DQN_variant/train.py --train_total_steps 200 --warmup_size 100 --test_every_steps 50 --dueling True --env PongNoFrameskip-v4
    # python -m pip uninstall -r ./examples/DQN_variant/requirements.txt -y
    
    # python -m pip install -r ./examples/PPO/requirements_atari.txt
    # python examples/PPO/train.py --train_total_steps 5000 --env PongNoFrameskip-v4
    # python -m pip uninstall -r ./examples/PPO/requirements_atari.txt -y

    python -m pip install -r ./examples/PPO/requirements_mujoco.txt
    python examples/PPO/train.py --train_total_steps 5000 --env HalfCheetah-v4 --continuous_action
    python -m pip uninstall -r ./examples/PPO/requirements_mujoco.txt -y

    python -m pip install -r ./examples/SAC/requirements.txt
    python examples/SAC/train.py --train_total_steps 5000 --env HalfCheetah-v4
    python -m pip uninstall -r ./examples/SAC/requirements.txt -y
   
    python -m pip install -r ./examples/TD3/requirements.txt
    python examples/TD3/train.py --train_total_steps 5000 --env HalfCheetah-v4
    python -m pip uninstall -r ./examples/TD3/requirements.txt -y

    python -m pip install -r ./examples/OAC/requirements.txt
    python examples/OAC/train.py --train_total_steps 5000 --env HalfCheetah-v4
    python -m pip uninstall -r ./examples/OAC/requirements.txt -y
    
    python -m pip install -r ./examples/DDPG/requirements.txt
    python examples/DDPG/train.py --train_total_steps 5000 --env HalfCheetah-v4
    python -m pip uninstall -r ./examples/DDPG/requirements.txt -y
    
    xparl start --port 8837 --cpu_num 4
    python -m pip install -r ./examples/ES/requirements.txt
    sed -i 's/24/4/g' ./examples/ES/es_config.py
    cat ./examples/ES/es_config.py
    python ./examples/ES/train.py --train_steps 2 --actor_num 4
    python -m pip uninstall -r ./examples/ES/requirements.txt -y
    xparl stop

    # TODO: raise Error while in Atari env
    # xparl start --port 8110 --cpu_num 5
    # python -m pip install -r ./examples/A2C/requirements.txt
    # python ./examples/A2C/train.py --max_sample_steps 50000
    # python -m pip uninstall -r ./examples/A2C/requirements.txt -y
    # xparl stop
    
    python -m pip install -r ./examples/MADDPG/requirements.txt
    python examples/MADDPG/train.py --max_episodes 21 --test_every_episodes 10
    python -m pip uninstall -r ./examples/MADDPG/requirements.txt -y
}

function print_usage() {
    echo -e "\n${RED}Usage${NONE}:
    ${BOLD}$0${NONE} [OPTION]"

    echo -e "\n${RED}Options${NONE}:
    ${BLUE}test_paddle${NONE}: run all unit tests with paddlepaddle
    ${BLUE}test_torch${NONE}: run all unit tests with torch
    ${BLUE}check_style${NONE}: run check for code style
    ${BLUE}example${NONE}: run examples
    "
}

function abort(){
    echo "Your change doesn't follow PaddlePaddle's code style." 1>&2
    echo "Please use pre-commit to check what is wrong." 1>&2
    exit 1
}

function check_style() {
    trap 'abort' 0
    set -e

    python -m pip install pre-commit
    pre-commit install
    # clang-format --version

    if ! pre-commit run -a ; then
        git diff
        exit 1
    fi

    trap : 0
}

function run_test_with_cpu() {
    export CUDA_VISIBLE_DEVICES=""

    mkdir -p ${REPO_ROOT}/build
    cd ${REPO_ROOT}/build
    if [ $# -eq 0 ];then
        cmake ..
    else
        cmake .. -$1=ON
    fi
    cat <<EOF
    =====================================================
    Running unit tests with CPU in the environment: `python --version`
    =====================================================
EOF
    if [ "$#" == 1 ] && [ "$1" == "DIS_TESTING_SERIALLY" ]
    then
        ctest --output-on-failure 
    else
        ctest --output-on-failure -j10
    fi
    cd ${REPO_ROOT}
    rm -rf ${REPO_ROOT}/build
}

function run_import_test {
    export CUDA_VISIBLE_DEVICES=""

    mkdir -p ${REPO_ROOT}/build
    cd ${REPO_ROOT}/build

    cmake .. -DIS_TESTING_IMPORT=ON

    cat <<EOF
    ========================================
    Running import test...
    ========================================
EOF
    ctest --output-on-failure
    cd ${REPO_ROOT}
    rm -rf ${REPO_ROOT}/build
}

function run_all_test_with_paddle {
    # pip config set global.index-url https://mirror.baidu.com/pypi/simple
    python -m pip install --upgrade pip
    echo ========================================
    echo Running tests in `python --version` with paddlepaddle
    echo `which pip`
    echo ========================================
    pip install .
    run_import_test # import parl test

    xparl stop
    pip install -r .teamcity/requirements.txt
    pip install paddlepaddle==2.3.1
    # pip install paddlepaddle==2.3.1 -f https://www.paddlepaddle.org.cn/whl/linux/openblas/noavx/stable.html --no-index --no-deps

    run_test_with_cpu
    # run_test_with_cpu "DIS_TESTING_SERIALLY" # TODO: raise Timeout Error
    # run_test_with_cpu "DIS_TESTING_REMOTE"  # TODO: raise Timeout Error
    xparl stop
    python -m pip uninstall -r .teamcity/requirements.txt -y
}

function run_all_test_with_torch {
    # pip config set global.index-url https://mirror.baidu.com/pypi/simple
    python -m pip install --upgrade pip
    echo ========================================
    echo Running tests in `python --version` with torch
    echo `which pip`
    echo ========================================
    pip install .

    xparl stop
    # test with torch installed

    # install torch
    pip uninstall -y paddlepaddle
    python -m pip uninstall -r .teamcity/requirements.txt -y
    echo ========================================
    echo "in torch environment"
    echo ========================================
    pip install -r .teamcity/requirements_torch.txt
    pip install torch
    pip install decorator

    run_test_with_cpu "DIS_TESTING_TORCH"
    # run_test_with_cpu "DIS_TESTING_SERIALLY" # TODO: raise Timeout Error
    # run_test_with_cpu "DIS_TESTING_REMOTE" # TODO: raise Timeout Error
    python -m pip uninstall -r .teamcity/requirements_torch.txt -y
    xparl stop
}

function main() {
    set -e
    local CMD=$1
    echo $CMD
    
    init
    case $CMD in
        check_style)
            check_style
            ;;
        test_paddle)
                run_all_test_with_paddle
            ;;
        test_torch)
                run_all_test_with_torch
            ;;
        example)
            # run example test in env test_example(python 3.8)
            # pip config set global.index-url https://mirror.baidu.com/pypi/simple
            pip install .
            pip install paddlepaddle==2.3.1
            # pip install paddlepaddle==2.3.1 -f https://www.paddlepaddle.org.cn/whl/linux/openblas/noavx/stable.html --no-index --no-deps
            run_example_test
            ;;
        *)
            print_usage
            exit 0
            ;;
    esac
    echo "finished: ${CMD}"
}

main $@
