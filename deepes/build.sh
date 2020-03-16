#!/bin/bash
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

#----------------protobuf-------------#
cp ./src/proto/deepes.proto ./
protoc deepes.proto --cpp_out ./
mv deepes.pb.h ./include
mv deepes.pb.cc ./src

#---------------libtorch-------------#
if [ ! -d "./libtorch" ];then
  echo "Cannot find the torch library: ./libtorch"
  echo "Please put the torch libraray to current folder according the instruction in README"
  exit 1
fi

#----------------build---------------#
rm -rf build
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=./libtorch ../
make -j10
./parallel_main
