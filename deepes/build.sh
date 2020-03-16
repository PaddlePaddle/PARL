#!/bin/bash
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

#----------------protobuf-------------#
cp ./src/proto/deepes.proto ./
protoc deepes.proto --cpp_out ./
mv deepes.pb.h ./include
mv deepes.pb.cc ./src

#----------------build---------------#
rm -rf build
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/home/zhoubo01/Firework/cpptorch/libtorch ../
make -j10
./parallel_main
