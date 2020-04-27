#!/usr/bin/env bash

# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

# NOTE: You need install mingw-cmake.

function init() {
    RED='\033[0;31m'
    BLUE='\033[0;34m'
    BOLD='\033[1m'
    NONE='\033[0m'

    REPO_ROOT=`pwd`
}


function abort(){
    echo "Your change doesn't follow PaddlePaddle's code style." 1>&2
    echo "Please use pre-commit to check what is wrong." 1>&2
    exit 1
}

function run_test_with_cpu() {
    export CUDA_VISIBLE_DEVICES="-1"

    mkdir -p ${REPO_ROOT}/build
    cd ${REPO_ROOT}/build
    if [ $# -eq 1 ];then
        cmake  -G "MinGW Makefiles" ..
    else
        cmake  -G "MinGW Makefiles" .. -$2=ON
    fi
    cat <<EOF
    =====================================================
    Running unit tests with CPU in the environment: $1
    =====================================================
EOF
    if [ $# -eq 1 ];then
      ctest --output-on-failure -j10
    else
      ctest --output-on-failure 
    fi
    cd ${REPO_ROOT}
    rm -rf ${REPO_ROOT}/build
}

function main() {
    set -e
    local CMD=$1
    
    init
    env="unused_variable"
    # run unittest in windows (used in local machine)
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple .
    pip uninstall -y torch torchvision
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple paddlepaddle==1.6.1 gym details parameterized
    run_test_with_cpu $env
    run_test_with_cpu $env "DIS_TESTING_SERIALLY"
    pip uninstall -y paddlepaddle
    pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html  
    run_test_with_cpu $env "DIS_TESTING_TORCH"
}

main $@
