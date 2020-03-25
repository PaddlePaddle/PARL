#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

#----------------protobuf-------------#
rm deepes.proto
cp ./src/proto/deepes.proto ./
protoc deepes.proto --cpp_out ./
mv deepes.pb.h ./include
mv deepes.pb.cc ./src

#----------------build---------------#
rm -rf build
mkdir build
cd build
cmake ../test  # -DWITH_TORCH=ON
make -j10
./unit_test_main
