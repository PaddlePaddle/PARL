#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

#---------------libtorch-------------#
if [ ! -d "./libtorch" ] && [  "$1" = "torch"  ];then
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


if [ "$1" = "torch" ]; then
    #----------------build---------------#
    rm -rf build
    mkdir build
    cd build
    cmake ../test/torch_agent
    make -j10
    #-----------------run----------------#
    ./torch_agent_test_main > info
    #-----------------test---------------#
    loss=`tail -1 info`
    test_loss=${loss%%|*}
    train_loss=${loss#*|}
    echo "test_loss: ${test_loss}"
    echo "train_loss: ${train_loss}"
    threshold=0.1
    if [ `echo "$test_loss > $threshold"|bc` -eq 1 ]; then
        echo "[warning] test_loss > $threshold"
        exit 1
    fi
    if [ `echo "$test_loss-$train_loss > $threshold"|bc` -eq 1 ] || [ `echo "$train_loss-$test_loss > $threshold"|bc` -eq 1 ]; then
        echo "[warning] abs(test_loss - train_loss) > $threshold"
        exit 1
    fi
else
    #----------------build---------------#
    rm -rf build
    mkdir build
    cd build
    cmake ../test
    make -j10
    #-----------------run----------------#
    ./unit_test_main
fi