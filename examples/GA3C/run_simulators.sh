#!/bin/bash

export CUDA_VISIBLE_DEVICES=""

for i in $(seq 1 24); do
    python simulator.py &
done;
wait
