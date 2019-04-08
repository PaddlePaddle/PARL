#!/bin/bash

export CUDA_VISIBLE_DEVICES=""

for index in {1..5}; do
    python actor.py &
done;
wait
