#!/bin/bash

if [ $# != 1 ]; then
  echo "You must choose one framework (paddle/torch) to compile DeepES."
  exit 0
fi

if [ $1 = "paddle" ]; then
  #---------------paddlelite-------------#
  if [ ! -d "./inference_lite_lib" ];then
    echo "Cannot find the PaddleLite library: ./inference_lite_lib"
    echo "Please put the PaddleLite libraray to current folder according the instruction in README"
    exit 1
  fi
  
  # Initialization model
  if [ ! -d ./demo/paddle/cartpole_init_model ]; then
    unzip ./demo/paddle/cartpole_init_model.zip -d ./demo/paddle/
  fi

  FLAGS=" -DWITH_PADDLE=ON"
elif [ $1 = "torch" ]; then
  #---------------libtorch-------------#
  if [ ! -d "./libtorch" ];then
    echo "Cannot find the torch library: ./libtorch"
      echo "Downloading Torch library"
      wget -q https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.4.0%2Bcpu.zip
      unzip -q libtorch-cxx11-abi-shared-with-deps-1.4.0+cpu.zip
      rm -rf libtorch-cxx11-abi-shared-with-deps-1.4.0+cpu.zip
      echo "Torch library Downloaded"
  fi
  FLAGS=" -DWITH_TORCH=ON"
else
  echo "Invalid arguments. [paddle/torch]"
  exit 0
fi

#----------------protobuf-------------#
cp ./core/src/proto/deepes.proto ./
protoc deepes.proto --cpp_out ./
mv deepes.pb.h core/include
mv deepes.pb.cc core/src
rm deepes.proto

#----------------build---------------#
echo ${FLAGS}
rm -rf build
mkdir build
cd build
cmake ../ ${FLAGS}
make -j10

#-----------------run----------------#
./parallel_main
