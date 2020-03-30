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
cp ./src/proto/deepes.proto ./
protoc deepes.proto --cpp_out ./
mv deepes.pb.h ./include
mv deepes.pb.cc ./src


#----------------build---------------#
rm -rf build
mkdir build
cd build
cmake ../test
make -j10

#-----------------run----------------#
./unit_test_main
