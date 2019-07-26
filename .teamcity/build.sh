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

function init() {
    RED='\033[0;31m'
    BLUE='\033[0;34m'
    BOLD='\033[1m'
    NONE='\033[0m'

    REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../" && pwd )"
}

function print_usage() {
    echo -e "\n${RED}Usage${NONE}:
    ${BOLD}$0${NONE} [OPTION]"

    echo -e "\n${RED}Options${NONE}:
    ${BLUE}test${NONE}: run all unit tests
    ${BLUE}check_style${NONE}: run code style check
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

    export PATH=/usr/bin:$PATH
    pre-commit install
    clang-format --version

    if ! pre-commit run -a ; then
        git diff
        exit 1
    fi

    trap : 0
}

function run_test_with_gpu() {
    export FLAGS_fraction_of_gpu_memory_to_use=0.5
    
    mkdir -p ${REPO_ROOT}/build
    cd ${REPO_ROOT}/build
    cmake ..
    cat <<EOF
    ========================================
    Running unit tests with GPU...
    ========================================
EOF
    ctest --output-on-failure
    rm -rf ${REPO_ROOT}/build
}

function run_test_with_cpu() {
    export CUDA_VISIBLE_DEVICES=""

    mkdir -p ${REPO_ROOT}/build
    cd ${REPO_ROOT}/build
    cmake ..
    cat <<EOF
    ========================================
    Running unit tests with CPU...
    ========================================
EOF
    ctest --output-on-failure
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
    rm -rf ${REPO_ROOT}/build
}

function run_docs_test {
    #export CUDA_VISIBLE_DEVICES=""

    mkdir -p ${REPO_ROOT}/build
    cd ${REPO_ROOT}/build

    cmake .. -DIS_TESTING_DOCS=ON 

    cat <<EOF
    ========================================
    Running docs test...
    ========================================
EOF
    ctest --output-on-failure
    rm -rf ${REPO_ROOT}/build
}

function main() {
    set -e
    local CMD=$1
    
    init
    case $CMD in
        check_style)
          check_style
          ;;
        test)
          pip install -i https://pypi.tuna.tsinghua.edu.cn/simple .
          pip3.6 install -i https://pypi.tuna.tsinghua.edu.cn/simple .
          /root/miniconda3/envs/empty_env/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple .
          run_test_with_gpu
          run_test_with_cpu
          run_import_test
          run_docs_test
          ;;
        *)
          print_usage
          exit 0
          ;;
    esac
}

main $@
