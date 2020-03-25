#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

#----------------protobuf-------------#
rm deepes.proto
cp ./src/proto/deepes.proto ./
protoc deepes.proto --cpp_out ./
mv deepes.pb.h ./include
mv deepes.pb.cc ./src

#---------------libtorch-------------#
# if [ ! -d "./libtorch" ];then
#  	echo "Cannot find the torch library: ./libtorch"
# 	echo "Downloading Torch library"
# 	wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.4.0%2Bcpu.zip
# 	tar zxvf libtorch-cxx11-abi-shared-with-deps-1.4.0+cpu.zip
#  	rm libtorch-cxx11-abi-shared-with-deps-1.4.0+cpu.zip
#  	echo "Torch library Downloaded"
# fi

#----------------build---------------#
rm -rf build
mkdir build
cd build
cmake ../test  #  -DWITH_TORCH=ON
make -j10
./unit_test_main
