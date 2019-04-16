#!/bin/bash

export CUDA_VISIBLE_DEVICES=""

for i in $(seq 1 5); do
    python actor.py &
done;
wait
