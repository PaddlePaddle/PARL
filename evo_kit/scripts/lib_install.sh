#!/bin/bash

if [ $# != 1 ]; then
  echo "You must choose one framework (paddle/torch) to compile EvoKit."
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
  FLAGS=" -DWITH_TORCH=ON"
else
  echo "Invalid arguments. [paddle/torch]"
  exit 0
fi


#----------------protobuf-------------#
cd core/proto/
protoc evo_kit/evo_kit.proto --cpp_out . 
cd -

#----------------build---------------#
echo ${FLAGS}
rm -rf build
mkdir build
cd build
cmake ../ ${FLAGS}
make -j10
make install
cd -
