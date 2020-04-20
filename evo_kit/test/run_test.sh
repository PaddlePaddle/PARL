#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

#---------------libtorch-------------#
if [ ! -d "./libtorch" ];then
echo "Cannot find the torch library: ../libtorch"
  echo "Downloading Torch library"
  wget -q https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.4.0%2Bcpu.zip
  unzip -q libtorch-cxx11-abi-shared-with-deps-1.4.0+cpu.zip
  rm -rf libtorch-cxx11-abi-shared-with-deps-1.4.0+cpu.zip
  echo "Torch library Downloaded"
fi

#----------------protobuf-------------#
cd core/proto/
protoc evo_kit/evo_kit.proto --cpp_out . 
cd -

#----------------build---------------#
sh scripts/lib_install.sh torch

#----------------build test---------------#
cd test

cp -r ../libevokit ./
if [ ! -d "./libevokit" ];then
  echo "Cannot find the EvoKit library: ./libevokit"
  echo "Please put the EvoKit libraray to current folder according the instruction in README" # TODO: readme
  exit 1
fi

rm -rf build
mkdir build
cd build
cmake ../
make -j10

#-----------------run----------------#
./unit_test_main

cd ..
